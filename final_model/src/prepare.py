"""Rebuild only the fixed model's monthly selections and 15-column features.

All candidate edges exist in memory for one month; only selected edges are saved.
Existing monthly files are reused unless --force is explicitly supplied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.special import ndtr
from statsmodels.tsa import adfvalues
from engine import load_data, rolling_beta

ROOT = Path(__file__).resolve().parents[1]
PRICE_FILE = ROOT / 'data/prices.parquet.gz'
SELECTION_DIR = ROOT / 'data/selection'
FEATURE_DIR = ROOT / 'data/features'
CAPS = ('C1', 'C2', 'C4', 'Cinf', 'M', 'B')
FEATURE_COLUMNS = ['date', 'signal_date', 'x', 'y', 'xi', 'yi', 'beta',
                   'q_fit', 'z_fit', 'is_B', 'is_C1', 'is_C2', 'is_C4',
                   'is_Cinf', 'is_M']


def adf_pvalues(t):
    t = np.asarray(t)
    small = np.polyval(adfvalues._tau_smallps['c'][0][::-1], t)
    large = np.polyval(adfvalues._tau_largeps['c'][0][::-1], t)
    p = ndtr(np.where(t <= adfvalues._tau_stars['c'][0], small, large))
    return np.where(t > adfvalues._tau_maxs['c'][0], 1.,
                    np.where(t < adfvalues._tau_mins['c'][0], 0., p))


def adf_fixed1_batch(residuals):
    """OLS equivalent of adfuller(e, maxlag=1, regression='c', autolag=None)."""
    dependent = np.diff(residuals, axis=1)[:, 1:]
    level = residuals[:, 1:-1].copy()
    lagdiff = np.diff(residuals, axis=1)[:, :-1]
    dependent -= dependent.mean(axis=1, keepdims=True)
    level -= level.mean(axis=1, keepdims=True)
    lagdiff -= lagdiff.mean(axis=1, keepdims=True)
    ee = np.einsum('ij,ij->i', level, level)
    ll = np.einsum('ij,ij->i', lagdiff, lagdiff)
    el = np.einsum('ij,ij->i', level, lagdiff)
    ed = np.einsum('ij,ij->i', level, dependent)
    ld = np.einsum('ij,ij->i', lagdiff, dependent)
    dd = np.einsum('ij,ij->i', dependent, dependent)
    det = ee*ll-el*el
    b = (ed*ll-ld*el)/det
    c = (ld*ee-ed*el)/det
    rss = np.maximum(dd-b*ed-c*ld, 0.)
    variance = rss/(dependent.shape[1]-3)
    return b/np.sqrt(variance*ll/det)


def calculate_edges(log_prices, batch_size=1024):
    """Alphabetical x->y log OLS and fixed-one-lag residual ADF."""
    a = log_prices.to_numpy(dtype=float)
    means = a.mean(axis=0)
    centered = a-means
    cov = centered.T@centered
    i, j = np.triu_indices(a.shape[1], k=1)
    beta = cov[i, j]/cov[i, i]
    alpha = means[j]-beta*means[i]
    t = np.empty(len(i))
    for start in range(0, len(i), batch_size):
        stop = start+batch_size
        e = (a[:, j[start:stop]]-a[:, i[start:stop]]*beta[start:stop]-alpha[start:stop]).T
        t[start:stop] = adf_fixed1_batch(e)
    cols = log_prices.columns.to_numpy()
    return pd.DataFrame({'x': cols[i], 'y': cols[j], 'beta': beta,
                         'alpha': alpha, 't_stat': t,
                         'p_value': adf_pvalues(t)})


def select_matching_and_baseline(edges):
    """Maximum total -ADF t, without forcing maximum cardinality; B is top250 p."""
    edges = edges.sort_values(['p_value', 'x', 'y'], kind='stable').reset_index(drop=True)
    names = sorted(set(edges.x) | set(edges.y))
    values = -edges.t_stat.to_numpy()
    try:
        import rustworkx as rx
    except ImportError:
        graph = nx.Graph()
        graph.add_nodes_from(names)
        graph.add_weighted_edges_from((x, y, float(w)) for x, y, w in zip(edges.x, edges.y, values))
        matching = nx.max_weight_matching(graph, maxcardinality=False, weight='weight')
        backend = 'networkx'
    else:
        lookup = {x: i for i, x in enumerate(names)}
        ratios = [float(x).as_integer_ratio() for x in values]
        denominator = max(q for p, q in ratios)
        weights = [p*(denominator//q) for p, q in ratios]
        assert all(w*q == p*denominator for w, (p, q) in zip(weights, ratios))
        assert max(abs(w).bit_length() for w in weights)+20 < 120
        graph = rx.PyGraph(multigraph=False)
        graph.add_nodes_from(names)
        graph.add_edges_from([(lookup[x], lookup[y], w) for x, y, w in zip(edges.x, edges.y, weights)])
        result = rx.max_weight_matching(graph, max_cardinality=False,
                                       weight_fn=lambda w: w, verify_optimum=True)
        matching = {(names[i], names[j]) for i, j in result}
        backend = 'rustworkx/binary64-exact integer weights'
    matchset = {frozenset(e) for e in matching}
    edges['is_M'] = [frozenset((x, y)) in matchset for x, y in zip(edges.x, edges.y)]
    edges['is_B'] = np.arange(len(edges)) < 250
    return edges, backend


def solve_degree(edges, cap, k):
    """Exactly K distinct edges, degree <= cap, maximizing total -ADF t."""
    names = sorted(set(edges.x) | set(edges.y))
    lookup = {v: i for i, v in enumerate(names)}
    n = len(edges)
    xi = np.array([lookup[v] for v in edges.x])
    yi = np.array([lookup[v] for v in edges.y])
    a = coo_matrix((np.ones(3*n), (np.r_[xi, yi, np.full(n, len(names))],
                    np.tile(np.arange(n), 3))), shape=(len(names)+1, n)).tocsc()
    lo = np.r_[np.full(len(names), -np.inf), k]
    hi = np.r_[np.full(len(names), cap), k]
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected')
        result = milp(c=edges.t_stat.to_numpy(), integrality=np.ones(n),
                      bounds=Bounds(0, 1), constraints=LinearConstraint(a, lo, hi),
                      options={'time_limit': 240., 'mip_rel_gap': 1e-9, 'threads': 1})
    if result.x is None:
        raise RuntimeError(f'No feasible degree-{cap} selection: {result.message}')
    choose = result.x > .5
    degree = np.bincount(np.r_[xi[choose], yi[choose]], minlength=len(names))
    assert choose.sum() == k and degree.max() <= cap
    info = {'solver': 'scipy.optimize.milp/HiGHS', 'status': int(result.status),
            'message': result.message, 'mip_gap': float(result.mip_gap),
            'objective_score': float(-result.fun), 'dual_upper_score': float(-result.mip_dual_bound),
            'seconds': time.monotonic()-started, 'n_binary_variables': n,
            'optimal': bool(result.success)}
    if result.mip_gap > 1e-7:
        raise RuntimeError(f'Uncertified material objective gap: {info}')
    return edges.loc[choose].copy(), info


def add_formation_risk(frame, prices, month):
    hist = prices.loc[prices.index < month+'-01'].tail(504)
    xi = prices.columns.get_indexer(frame.x)
    yi = prices.columns.get_indexer(frame.y)
    p = hist.to_numpy()
    x, y = np.log(p[:, xi]), np.log(p[:, yi])
    beta = frame.beta.to_numpy()
    residual = y-frame.alpha.to_numpy()-beta*x
    rx, ry = p[1:, xi]/p[:-1, xi]-1, p[1:, yi]/p[:-1, yi]-1
    hedge_return = (ry-beta*rx)/(1+np.abs(beta))
    frame['formation_pair_vol_daily'] = hedge_return.std(axis=0, ddof=1)
    frame['formation_pair_vol_annual'] = frame.formation_pair_vol_daily*np.sqrt(252)
    frame['formation_residual_sigma'] = np.sqrt((residual**2).sum(axis=0)/(504-2))
    frame['formation_cutoff'] = hist.index[-1]
    frame['score'] = -frame.t_stat
    return frame


def select_month(month, force=False):
    dest, meta_dest = SELECTION_DIR/f'{month}.parquet', SELECTION_DIR/f'{month}.json'
    if dest.exists() and not force:
        if meta_dest.exists():
            return json.loads(meta_dest.read_text())
        return {'month': month, 'reused': True, 'path': str(dest)}
    prices = load_data(PRICE_FILE)
    hist = prices.loc[prices.index < pd.Timestamp(month+'-01')].tail(504)
    ok = np.isfinite(hist.to_numpy()).all(axis=0) & (hist.to_numpy() > 0).all(axis=0)
    hist = hist.loc[:, ok].sort_index(axis=1)
    assert len(hist) == 504
    edges, backend = select_matching_and_baseline(calculate_edges(np.log(hist)))
    edges = edges.sort_values(['t_stat', 'x', 'y'], kind='stable').reset_index(drop=True)
    k = int(edges.is_M.sum())
    selected = {'C1': edges.loc[edges.is_M].copy(), 'Cinf': edges.iloc[:k].copy()}
    solver = {}
    for cap in (2, 4):
        selected[f'C{cap}'], solver[f'C{cap}'] = solve_degree(edges, cap, k)
    selected['M'], selected['B'] = selected['C1'], edges.loc[edges.is_B].copy()
    parts = []
    for cap in CAPS:
        d = selected[cap][['x', 'y', 'beta', 'alpha', 't_stat', 'p_value']].copy()
        d.insert(0, 'cap', cap)
        d.insert(0, 'month', month)
        d = add_formation_risk(d, prices, month)
        assert len(d) == (250 if cap == 'B' else k)
        parts.append(d)
    result = pd.concat(parts, ignore_index=True)
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(dest.with_suffix('.tmp.parquet'), index=False)
    dest.with_suffix('.tmp.parquet').replace(dest)
    meta = {'schema_version': 2, 'month': month, 'K': k, 'n_candidates': len(edges),
            'solver': solver, 'matching_backend': backend,
            'cutoff': str(hist.index[-1].date()), 'window_start': str(hist.index[0].date()),
            'source_sha256': hashlib.sha256(PRICE_FILE.read_bytes()).hexdigest(),
            'C1_source': 'maximum total -ADF t; no maximum-cardinality priority',
            'Cinf_source': 'top K by -ADF t statistic, deterministic x/y tie-break'}
    meta_dest.write_text(json.dumps(meta, indent=2))
    # A newly selected universe invalidates its derived features.
    (FEATURE_DIR/f'{month}.parquet').unlink(missing_ok=True)
    return meta


def calculate_features(sel, prices, month):
    """Current-fit beta/Q/Z at the previous close; all other signals were retired."""
    dates = prices.index
    ids = np.where((dates.to_period('M').astype(str) == month) &
                   (dates >= '2017-01-01') & (dates <= '2023-05-31'))[0]
    if not len(ids):
        raise ValueError(f'No evaluation dates for {month}')
    xi, yi = prices.columns.get_indexer(sel.x), prices.columns.get_indexer(sel.y)
    if (xi < 0).any() or (yi < 0).any():
        raise ValueError('Selected ticker absent from price columns')
    # Preserve the original rolling-sum start to retain floating-point results.
    start, end = max(0, ids[0]-504-260-5), ids[-1]+1
    lp = np.log(prices.to_numpy())
    x, y = lp[start:end, xi], lp[start:end, yi]
    a, b, res, sigma = rolling_beta(x, y)
    idx = ids-start-1
    qfit = []
    for js in idx:
        rr = y[js-503:js+1]-a[js]-b[js]*x[js-503:js+1]
        quant = np.quantile(rr, [.25, .5, .75], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            qfit.append((res[js]-quant[1])/(quant[2]-quant[0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        zfit = res[idx]/sigma[idx]
    return pd.DataFrame({'date': np.repeat(dates[ids].to_numpy(), len(sel)),
                         'signal_date': np.repeat(dates[ids-1].to_numpy(), len(sel)),
                         'x': np.tile(sel.x, len(ids)), 'y': np.tile(sel.y, len(ids)),
                         'xi': np.tile(xi, len(ids)), 'yi': np.tile(yi, len(ids)),
                         'beta': b[idx].ravel(), 'q_fit': np.array(qfit).ravel(),
                         'z_fit': zfit.ravel()})


def selection_flags(selected):
    flags = selected.assign(value=True).pivot(index=['x', 'y'], columns='cap', values='value').eq(True)
    flags.columns = ['is_'+c for c in flags.columns]
    flags = flags.reset_index()
    for cap in CAPS:
        if 'is_'+cap not in flags:
            raise ValueError(f'Missing selection group {cap}')
        flags['is_'+cap] = flags['is_'+cap].astype(bool)
    return flags


def feature_month(month, force=False):
    dest = FEATURE_DIR/f'{month}.parquet'
    if dest.exists() and not force:
        existing = pd.read_parquet(dest, columns=FEATURE_COLUMNS)
        return {'month': month, 'rows': len(existing), 'reused': True}
    selected = pd.read_parquet(SELECTION_DIR/f'{month}.parquet')
    flags = selection_flags(selected)
    result = calculate_features(flags, load_data(PRICE_FILE), month).merge(flags, on=['x', 'y'], how='left', validate='many_to_one')
    result = result.sort_values(['date', 'x', 'y']).reset_index(drop=True)[FEATURE_COLUMNS]
    assert not result.duplicated(['date', 'x', 'y']).any()
    assert (result.is_C1 == result.is_M).all()
    for cap in CAPS:
        k = len(selected[selected.cap == cap])
        assert result[result['is_'+cap]].groupby('date').size().eq(k).all()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(dest.with_suffix('.tmp.parquet'), index=False)
    dest.with_suffix('.tmp.parquet').replace(dest)
    return {'month': month, 'rows': len(result), 'union_pairs': len(flags), 'reused': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['selection', 'features', 'all'], default='all')
    parser.add_argument('--months', nargs='+', help='Explicit months, e.g. 2017-01 2023-05')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--force', action='store_true', help='Rebuild requested monthly outputs instead of reusing them')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    months = args.months or pd.period_range('2017-01', '2023-05', freq='M').astype(str).tolist()
    if any(str(pd.Period(m, freq='M')) != m or m < '2017-01' or m > '2023-05' for m in months):
        parser.error('Months must use YYYY-MM within 2017-01 through 2023-05')
    if len(set(months)) != len(months):
        parser.error('Duplicate months')
    for stage, fun in [('selection', select_month), ('features', feature_month)]:
        if args.stage not in (stage, 'all'):
            continue
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fun, month, args.force) for month in months]
            for future in as_completed(futures):
                print(stage.upper(), json.dumps(future.result()), flush=True)


if __name__ == '__main__':
    main()
