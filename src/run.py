"""Run the completed public Q-code route, preserving original arithmetic."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['MAX_WEIGHT', 'BASELINE']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trade_one(start, end, prices_path, selection_dir, interval_dir, force=False):
    from selection import environment
    from trading import trade_interval
    key = pd.Timestamp(start).strftime('%Y%m%d')
    selection_path = Path(selection_dir)/(key+'.parquet')
    out = Path(interval_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt = out/(key+'.json')
    paths = {method: out/f'{key}_{method}.parquet' for method in METHODS}
    identity = {'start': str(pd.Timestamp(start).date()), 'end': str(pd.Timestamp(end).date()),
                'price_sha256': digest(prices_path), 'selection_sha256': digest(selection_path),
                'run_sha256': digest(__file__), 'trading_sha256': digest(ROOT/'src/trading.py'),
                'source_performance_sha256': digest(ROOT/'source/performance.py'), 'environment': environment()}
    if receipt.exists() and not force:
        try:
            old = json.loads(receipt.read_text())
            if old.get('cache_identity') == identity and all(p.exists() and digest(p) == old['output_sha256'][m] for m,p in paths.items()):
                return {**old, 'reused_verified_cache': True}
        except (OSError, ValueError, KeyError):
            pass
    begun = time.monotonic()
    prices = pd.read_parquet(prices_path)
    selected = pd.read_parquet(selection_path)
    result = {'cutoff': identity['start'], 'end': identity['end'], 'cache_identity': identity,
              'created_at': datetime.now(timezone.utc).isoformat(), 'methods': {}, 'output_sha256': {}}
    for method in METHODS:
        pairs = selected[selected.METHOD.eq(method)].sort_values('POSITION')
        daily, meta = trade_interval(prices, pairs, start, end)
        if any(meta[k] for k in ['infinite_returns', 'infinite_pair_pnl_elements', 'infinite_pair_capital_elements']):
            raise ValueError(f'Infinite original arithmetic: {key} {method}')
        daily['formation_cutoff'] = pd.Timestamp(start)
        tmp = paths[method].with_suffix('.tmp.parquet')
        daily.to_parquet(tmp)
        tmp.replace(paths[method])
        result['methods'][method] = meta
        result['output_sha256'][method] = digest(paths[method])
    result['seconds'] = time.monotonic()-begun
    tmp = receipt.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(receipt)
    return {**result, 'reused_verified_cache': False}


def trade_all(boundaries, prices_path, selection_dir, interval_dir, workers=2, wait_selection=False, force=False):
    remaining = {pd.Timestamp(s).strftime('%Y%m%d'): (s,e) for s,e in zip(boundaries[:-1], boundaries[1:])}
    pending, completed = {}, []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while remaining or pending:
            for key in list(remaining):
                if len(pending) >= workers:
                    break
                if not (Path(selection_dir)/(key+'.json')).exists():
                    continue
                start, end = remaining.pop(key)
                future = pool.submit(trade_one, start, end, str(prices_path), str(selection_dir), str(interval_dir), force)
                pending[future] = key
            if pending:
                finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in finished:
                    key = pending.pop(future)
                    result = future.result()
                    completed.append(result)
                    print(json.dumps({'stage': 'trading', 'completed': len(completed), 'total': len(boundaries)-1,
                                      'cutoff': key, 'verified_cache_reused': result['reused_verified_cache']}), flush=True)
            elif remaining:
                if not wait_selection:
                    raise FileNotFoundError(f'Missing selection files: {list(remaining)}')
                if time.monotonic()-started > 3600:
                    raise TimeoutError('Selection did not finish within one hour')
                time.sleep(1)
    return completed


def summarize(boundaries, interval_dir, output_dir):
    from trading import summarize_original_style
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = []
    for scope, (start,end) in {'source_default': ('2020-01-31','2023-05-31'), 'extended_2017': ('2017-01-03','2023-05-31')}.items():
        starts = [s for s in boundaries[:-1] if scope == 'extended_2017' or pd.Timestamp(s).year >= 2020]
        for method in METHODS:
            frames = [pd.read_parquet(Path(interval_dir)/f'{pd.Timestamp(s):%Y%m%d}_{method}.parquet') for s in starts]
            daily = pd.concat(frames).sort_index().loc[start:end]
            daily.index.name = 'date'
            daily.to_csv(out/f'{scope}_{method}.csv')
            result = summarize_original_style(daily.returns)
            wealth = np.r_[1.,np.cumprod(1+daily.returns.fillna(0).to_numpy())]
            result.update(scope=scope, method=method, start=str(daily.index.min().date()), end=str(daily.index.max().date()),
                          unique_dates=daily.index.nunique(), duplicate_rows=int(daily.index.duplicated().sum()),
                          max_drawdown_supplement=float((wealth/np.maximum.accumulate(wealth)-1).min()),
                          finite_returns=bool(np.isfinite(daily.returns.dropna()).all()))
            metrics.append(result)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=ROOT/'data/prices.parquet.gz')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--selection-workers', type=int, default=4)
    parser.add_argument('--skip-selection', action='store_true')
    parser.add_argument('--wait-for-selection', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        parser.error('Run through reproduce.py or set PYTHONHASHSEED=0')
    if min(args.workers, args.selection_workers) < 1:
        parser.error('Worker counts must be positive')
    from selection import EXPECTED_SHA256, selection_cutoffs, generate_all
    if digest(args.prices) != EXPECTED_SHA256:
        raise ValueError('Unexpected price input')
    boundaries = selection_cutoffs(pd.read_parquet(args.prices))
    if not args.skip_selection:
        generate_all(args.prices, ROOT/'data/selection', boundaries, args.selection_workers, args.force)
    reports = trade_all(boundaries, args.prices, ROOT/'data/selection', ROOT/'results/intervals',
                        args.workers, args.wait_for_selection, args.force)
    metrics = summarize(boundaries, ROOT/'results/intervals', ROOT/'results')
    result = {'model_id': 'author_code_aaac6d9_q_replay', 'completed_at': datetime.now(timezone.utc).isoformat(),
              'source_commit': 'aaac6d999d6b077304a887caff2aab514d7ea6a2', 'input_sha256': digest(args.prices),
              'intervals': len(reports), 'strategies': METHODS, 'signal': 'Q only',
              'recomputed_intervals': sum(not r['reused_verified_cache'] for r in reports),
              'verified_cached_intervals': sum(r['reused_verified_cache'] for r in reports),
              'source_default': '202*.csv date filter, 2020-01-31 onward',
              'extension': 'Date range only extended to 2017; original duplicate endpoints retained',
              'completion_assumption': 'Absent daily SPREAD/BETA writer supplied with trailing504 log-OLS endpoints',
              'net_costs': 'Not computed; missing executable cost route in original source', 'metrics': metrics}
    (ROOT/'checks').mkdir(exist_ok=True)
    (ROOT/'checks/execution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
