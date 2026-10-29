"""Monthly past-only, equal-count pair-selection comparison.

All directed ADF candidates are newly calculated from each 504-row raw-price
window using the shared numerics module. No full-sample universe or p-value
filter is applied.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import platform
import time
from pathlib import Path
import networkx as nx
import rustworkx as rx
import numpy as np
import pandas as pd
import numerics as _numeric

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICES = ROOT/'data/prices.parquet.gz'
ADF_SOURCE = ROOT/'src/numerics.py'
EXPECTED_PRICE_SHA256 = '7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'
NUMERIC_COLUMNS = ['BETA', 'ALPHA', 'T_STAT', 'P_VALUE', 'USEDLAG', 'AIC']


def boundaries(prices):
    """Actual final trading dates at monthly boundaries, including end-only May."""
    calendar = pd.date_range('2016-12-01', '2023-05-31', freq='BME')
    return [prices.loc[:date].index[-1] for date in calendar]


def collapse_directions(directed):
    """Keep lower t; exact ties choose alphabetical explanatory/dependent names."""
    data = directed.copy()
    data['NODE_1'] = np.where(data.ASSET_1 < data.ASSET_2, data.ASSET_1, data.ASSET_2)
    data['NODE_2'] = np.where(data.ASSET_1 < data.ASSET_2, data.ASSET_2, data.ASSET_1)
    data = data.sort_values(['T_STAT', 'ASSET_1', 'ASSET_2'], kind='stable')
    data = data.drop_duplicates(['NODE_1', 'NODE_2'], keep='first')
    data['SCORE'] = -data.T_STAT
    return data.sort_values(['NODE_1', 'NODE_2'], kind='stable').reset_index(drop=True)


def edge_digest(edges, directed=False):
    if directed:
        ordered = edges.sort_values(['ASSET_1', 'ASSET_2'], kind='stable')
        columns = ['ASSET_1', 'ASSET_2']+NUMERIC_COLUMNS
    else:
        ordered = edges.sort_values(['NODE_1', 'NODE_2'], kind='stable')
        columns = ['NODE_1', 'NODE_2', 'ASSET_1', 'ASSET_2']+NUMERIC_COLUMNS
    encoded = ordered[columns].to_csv(index=False, float_format='%.17g', lineterminator='\n').encode()
    return hashlib.sha256(encoded).hexdigest()


def solve_graph(edges, eligible):
    """M optimizes -t without cardinality priority; B/U are then limited to K."""
    edges = edges.sort_values(['NODE_1', 'NODE_2'], kind='stable').reset_index(drop=True)
    ratios = [float(w).as_integer_ratio() for w in edges.SCORE]
    denominator = max((q for _, q in ratios), default=1)
    weights = [p*(denominator//q) for p, q in ratios]
    assert all(w*q == p*denominator for w, (p, q) in zip(weights, ratios))
    graph = nx.Graph()
    graph.add_nodes_from(sorted(eligible))
    graph.add_weighted_edges_from((x, y, w) for x, y, w in zip(edges.NODE_1, edges.NODE_2, weights))
    started = time.monotonic()
    names = sorted(eligible)
    lookup = {name:i for i,name in enumerate(names)}
    assert max((abs(w).bit_length() for w in weights), default=0)+20 < 120
    fast_graph = rx.PyGraph(multigraph=False)
    fast_graph.add_nodes_from(names)
    fast_graph.add_edges_from([(lookup[x],lookup[y],w) for x,y,w in zip(edges.NODE_1,edges.NODE_2,weights)])
    result = rx.max_weight_matching(fast_graph, max_cardinality=False,
                                    weight_fn=lambda w:w, verify_optimum=True)
    m = {tuple(sorted((names[i],names[j]))) for i,j in result}
    weighted_seconds = time.monotonic()-started
    k = len(m)
    bdata = edges.sort_values(['T_STAT', 'NODE_1', 'NODE_2'], kind='stable').head(k)
    b = set(zip(bdata.NODE_1, bdata.NODE_2))
    # Missing attribute is deliberately constant one. Stable alphabetical graph
    # insertion order and truncation use no cointegration score.
    started = time.monotonic()
    u_full = sorted(tuple(sorted(pair)) for pair in nx.max_weight_matching(
        graph, maxcardinality=True, weight='_unweighted_constant_one'))
    if len(u_full) < k:
        raise AssertionError('Maximum cardinality cannot be below M cardinality')
    u = set(u_full[:k])
    unweighted_seconds = time.monotonic()-started
    memberships = {'M': m, 'B': b, 'U': u}
    keys = list(zip(edges.NODE_1, edges.NODE_2))
    for style, chosen in memberships.items():
        edges['is_'+style] = [pair in chosen for pair in keys]
    selected = edges.loc[edges[['is_M', 'is_B', 'is_U']].any(axis=1)].copy()
    degree, objective = {}, {}
    for style in memberships:
        d = selected.loc[selected['is_'+style]]
        deg = pd.concat([d.NODE_1, d.NODE_2]).value_counts()
        degree[style] = {'max': int(deg.max()) if len(deg) else 0,
                         'used_stocks': int(len(deg)),
                         'histogram': {str(int(v)): int(n) for v, n in deg.value_counts().sort_index().items()}}
        objective[style] = float(d.SCORE.sum())
        assert len(d) == k
        if style in ('M', 'U'):
            assert len(deg) == 2*k and (deg <= 1).all()
    info = {'K': k, 'unweighted_max_cardinality': len(u_full), 'U_lexicographic_truncation': len(u_full)-k,
            'objective': objective, 'degree': degree,
            'binary64_weight_scale_denominator': str(denominator),
            'weighted_matching_seconds': weighted_seconds, 'unweighted_matching_seconds': unweighted_seconds,
            'weighted_backend': 'rustworkx max_weight_matching; verify_optimum=True; binary64-exact integer weights',
            'unweighted_backend': 'networkx max_weight_matching with missing weight attribute (constant1)'}
    return selected, info


def generate_month(cutoff, prices_path=DEFAULT_PRICES, output_dir=None, force=False):
    import statsmodels
    started = time.monotonic()
    cutoff = pd.Timestamp(cutoff)
    dest = Path(output_dir or ROOT/'data/selection')
    dest.mkdir(parents=True, exist_ok=True)
    key = cutoff.strftime('%Y%m%d')
    parquet, meta = dest/(key+'.parquet'), dest/(key+'.json')
    if parquet.exists() and meta.exists() and not force:
        return json.loads(meta.read_text())
    prices_path = Path(prices_path)
    source_hash = hashlib.sha256(prices_path.read_bytes()).hexdigest()
    if source_hash != EXPECTED_PRICE_SHA256:
        raise ValueError('Unexpected price input SHA256')
    prices = pd.read_parquet(prices_path).sort_index()
    if not prices.index.is_unique:
        raise ValueError('Duplicate source price dates')
    hist = prices.loc[:cutoff].tail(504)
    if len(hist) != 504 or hist.index[-1] != cutoff:
        raise ValueError('Invalid formation endpoint/window')
    eligible_mask = np.isfinite(hist.to_numpy()).all(axis=0) & (hist.to_numpy() > 0).all(axis=0)
    hist = hist.loc[:, eligible_mask].sort_index(axis=1)
    directed, fallbacks = _numeric.calculate_edges(np.log(hist), directed=True)
    directed_hash = edge_digest(directed, directed=True)
    graph_edges = collapse_directions(directed)
    graph_hash = edge_digest(graph_edges)
    adf_seconds = time.monotonic()-started
    selected_main, graph_info = solve_graph(graph_edges, hist.columns)
    excluded = {'SRE', 'CPRT'}
    clean_edges = graph_edges.loc[~graph_edges.NODE_1.isin(excluded) & ~graph_edges.NODE_2.isin(excluded)].copy()
    clean_eligible = [x for x in hist.columns if x not in excluded]
    selected_clean, clean_info = solve_graph(clean_edges, clean_eligible)
    selected = graph_edges.copy()
    all_keys = list(zip(selected.NODE_1, selected.NODE_2))
    for suffix, frame in [('', selected_main), ('_clean', selected_clean)]:
        for style in ('M', 'B', 'U'):
            d = frame.loc[frame['is_'+style]]
            chosen = set(zip(d.NODE_1, d.NODE_2))
            selected['is_'+style+suffix] = [pair in chosen for pair in all_keys]
    flags = ['is_'+style+suffix for suffix in ('', '_clean') for style in ('M', 'B', 'U')]
    selected = selected.loc[selected[flags].any(axis=1)].copy()
    selected.insert(0, 'CUTOFF', cutoff)
    selected.insert(1, 'WINDOW_START', hist.index[0])
    selected['K'] = graph_info['K']
    selected['Kclean'] = clean_info['K']
    selected.to_parquet(parquet.with_suffix('.tmp.parquet'), index=False)
    parquet.with_suffix('.tmp.parquet').replace(parquet)
    info = {'schema': 1, 'cutoff': str(cutoff.date()), 'window_start': str(hist.index[0].date()),
            'window_rows': 504, 'eligible_n': hist.shape[1], 'eligible': list(hist.columns),
            'raw_columns': prices.shape[1], 'source_sha256': source_hash,
            'numeric_adf_source': 'src/numerics.py',
            'numeric_adf_source_sha256': hashlib.sha256(ADF_SOURCE.read_bytes()).hexdigest(),
            'selection_algorithm_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'directed_candidates_attempted': hist.shape[1]*(hist.shape[1]-1),
            'directed_finite_candidates': len(directed), 'undirected_edges': len(graph_edges),
            'directed_edges_sha256': directed_hash, 'graph_edges_sha256': graph_hash,
            'edge_hash_format': 'canonical sorted UTF-8 CSV; numeric float_format %.17g; LF; columns in edge_digest',
            'lag_counts': {str(int(lag)): int(count) for lag, count in directed.USEDLAG.value_counts().sort_index().items()},
            'failed_adf_pairs_skipped': int(directed.attrs.get('adf_failed_pairs_skipped', 0)),
            'adf_fallbacks': fallbacks, 'exact_sort_tie_refits': int(directed.attrs.get('exact_sort_tie_refits', 0)),
            'direction': 'minimum t across both directions; exact ties alphabetical ASSET_1 then ASSET_2',
            'p_value_filter': None,
            'M_rule': 'maximum total -t; maxcardinality=False; binary64-exact integer scaling',
            'B_rule': 'top K by -t; canonical alphabetical exact-tie rule; K=M count',
            'U_rule': 'alphabetical graph, unweighted maximum cardinality; alphabetical first K pairs if larger',
            'selection_cache_read': False, 'full_sample_universe_used': False,
            'selected_union_rows': len(selected), 'selection_file': parquet.name,
            'clean_excluded_tickers': sorted(excluded), 'clean_eligible_n': len(clean_eligible),
            'clean_graph_edges': len(clean_edges), 'clean_graph_edges_sha256': edge_digest(clean_edges),
            'Kclean': clean_info['K'], 'clean': clean_info,
            'clean_protocol': 'SRE/CPRT excluded for all dates; same candidate ADF values, predeclared before new performance outcomes',
            'adf_seconds': adf_seconds, 'total_seconds': time.monotonic()-started,
            'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'statsmodels': statsmodels.__version__, 'networkx': nx.__version__, 'rustworkx':rx.__version__, **graph_info}
    temporary_meta = meta.with_suffix('.tmp')
    temporary_meta.write_text(json.dumps(info, indent=2))
    temporary_meta.replace(meta)
    return info


def brute_matching_objective(nodes, weights):
    """Enumerate every feasible matching independently of graph libraries."""
    nodes = tuple(nodes)
    def visit(remaining):
        if not remaining:
            return 0.
        first, rest = remaining[0], remaining[1:]
        best = visit(rest)
        for position, partner in enumerate(rest):
            edge = tuple(sorted((first, partner)))
            if edge in weights:
                best = max(best, weights[edge]+visit(rest[:position]+rest[position+1:]))
        return best
    return visit(nodes)


def validate(prices_path=DEFAULT_PRICES, checks_dir=None):
    from statsmodels.tsa.stattools import adfuller
    checks = Path(checks_dir or ROOT/'checks')
    checks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261029)
    graph_checks = []
    # First case requires preserving maxcardinality=False: weight10 alone beats
    # the two-edge weight8 matching, and U must be shortened from two to one.
    examples = [('abcd', {('a','b'):10., ('c','d'):-1., ('a','c'):4., ('b','d'):4.})]
    for _ in range(8):
        nodes = tuple('abcdefgh')
        examples.append((nodes, {(nodes[i], nodes[j]): float(rng.integers(-40,81))/8
                                for i in range(len(nodes)) for j in range(i+1,len(nodes))}))
    for number, (nodes, weights) in enumerate(examples):
        edges = pd.DataFrame([{'NODE_1':x,'NODE_2':y,'T_STAT':-w,'SCORE':w} for (x,y),w in weights.items()])
        chosen, info = solve_graph(edges, nodes)
        exact = brute_matching_objective(nodes, weights)
        assert info['objective']['M'] == exact
        if number == 0:
            assert info['K'] == 1 and info['U_lexicographic_truncation'] == 1
        graph_checks.append({'graph': number, 'nodes':len(nodes), 'edges':len(weights),
                             'brute_force_optimum':exact, 'M_objective':info['objective']['M'],
                             'K':info['K'], 'U_truncated':info['U_lexicographic_truncation']})
    prices = pd.read_parquet(prices_path)
    adf_checks = []
    # Include names absent from the earlier full-sample432 universe without
    # reading any earlier selection or eligibility cache.
    for cutoff in ['2016-12-30', '2020-01-31', '2023-04-28']:
        hist = prices.loc[:cutoff].tail(504)
        hist = hist.loc[:, np.isfinite(hist).all() & (hist>0).all()].sort_index(axis=1)
        names = sorted(set(list(hist.columns[::max(1,hist.shape[1]//8)][:8]) +
                           [x for x in ['ABBV','AMCR','ANET','CDW','COTY','ETSY','META','MRNA','VICI'] if x in hist.columns]))
        small = np.log(hist[names])
        fast, _ = _numeric.calculate_edges(small, directed=True)
        sampled = fast.sample(n=min(56,len(fast)), random_state=44)
        max_t, max_p, max_beta, mismatches = 0., 0., 0., 0
        for row in sampled.itertuples():
            x, y = small[row.ASSET_1].to_numpy(), small[row.ASSET_2].to_numpy()
            alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
            t,p,lag,*_ = adfuller(y-alpha-beta*x)
            max_t, max_p = max(max_t,abs(t-row.T_STAT)), max(max_p,abs(p-row.P_VALUE))
            max_beta = max(max_beta,abs(beta-row.BETA));mismatches += int(lag != row.USEDLAG)
        assert max_t < 1e-8 and max_p < 1e-8 and mismatches == 0
        adf_checks.append({'cutoff':cutoff,'samples':len(sampled),'max_t_error':max_t,
                           'max_p_error':max_p,'max_beta_error':max_beta,'lag_mismatches':mismatches,
                           'subset_names':names})
    # Ensure exact direction ties are alphabetical rather than hash-order based.
    ties = pd.DataFrame([{'ASSET_1':'b','ASSET_2':'a','T_STAT':-3.,'P_VALUE':.02},
                         {'ASSET_1':'a','ASSET_2':'b','T_STAT':-3.,'P_VALUE':.02}])
    assert collapse_directions(ties).iloc[0].ASSET_1 == 'a'
    result = {'passed':True,'small_graph_brute_force':graph_checks,'independent_OLS_ADF_samples':adf_checks,
              'exact_direction_tie_alphabetical':True,'scope':'nine exhaustive small graphs; independent OLS+statsmodels for168 sampled directed pairs'}
    (checks/'selection_validation.json').write_text(json.dumps(result,indent=2))
    return result


def generate_all(prices_path=DEFAULT_PRICES, output_dir=None, checks_dir=None, workers=2, force=False):
    if workers < 1 or workers > 2:
        raise ValueError('CPU limit: workers must be1 or2')
    prices = pd.read_parquet(prices_path)
    cutoffs = boundaries(prices)
    destination = Path(output_dir or ROOT/'data/selection')
    reused_cutoffs = [str(c.date()) for c in cutoffs[:-1] if not force
                      and (destination/(c.strftime('%Y%m%d')+'.parquet')).exists()
                      and (destination/(c.strftime('%Y%m%d')+'.json')).exists()]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(generate_month,c,prices_path,output_dir,force):c for c in cutoffs[:-1]}
        for future in as_completed(pending):
            info = future.result();results.append(info)
            print(json.dumps({k:info[k] for k in ['cutoff','eligible_n','K','undirected_edges','adf_seconds','weighted_matching_seconds','total_seconds']}),flush=True)
    results.sort(key=lambda x:x['cutoff'])
    assert len(results)==77
    manifest = {'passed':True,'selection_months':77,'first_cutoff':results[0]['cutoff'],
                'last_selection_cutoff':results[-1]['cutoff'],'last_interval_endpoint':str(cutoffs[-1].date()),
                'workers':workers,'prior_selection_cache_used':bool(reused_cutoffs),
                'cached_months_reused':reused_cutoffs,'months':results,
                'monthly_metadata_provenance': 'Month entries retain their creation-time source identifiers and hashes. Historical identifiers do not declare current runtime dependencies; arithmetic-preserving migration is recorded in checks/numeric_migration.json.',
                'historical_source_months': [r['cutoff'] for r in results if r.get('numeric_adf_source') != 'src/numerics.py'],
                'current_runtime_sources': {
                    'prices': {'path':'data/prices.parquet.gz','sha256':hashlib.sha256(Path(prices_path).read_bytes()).hexdigest()},
                    'selection': {'path':'src/selection.py','sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                    'numerics': {'path':'src/numerics.py','sha256':hashlib.sha256(ADF_SOURCE.read_bytes()).hexdigest()}}}
    checks=Path(checks_dir or ROOT/'checks');checks.mkdir(parents=True,exist_ok=True)
    (checks/'selection_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prices',type=Path,default=DEFAULT_PRICES)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--checks-dir',type=Path)
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--month')
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    if args.validate:
        print(json.dumps(validate(args.prices,args.checks_dir),indent=2))
    if args.month:
        info=generate_month(args.month,args.prices,args.output_dir,args.force)
        print(json.dumps({k:info[k] for k in ['cutoff','eligible_n','K','undirected_edges','adf_seconds','weighted_matching_seconds','total_seconds']},indent=2))
    if args.all:
        generate_all(args.prices,args.output_dir,args.checks_dir,args.workers,args.force)


if __name__=='__main__':
    main()
