"""Independently refit daily OLS, then compare unchanged public trade execution.

Run with OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1. This validates the
chosen completion mathematically; it cannot recover the missing author writer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

from trading import original_performance, trade_interval, verify_original_sample

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = '1ec5807f13b9c0944833b79cf3aa1ae36700ef3c801dd9c024a44c00e08e07a3'
PRICE_SHA256 = '7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'
INTERVALS = [
    ('2017-12-29', '2018-01-31'),
    ('2020-02-28', '2020-03-31'),
    ('2023-03-31', '2023-04-28'),
]
PAIR_GROUPS = {
    'group_a': [
        ('GPC', 'MMM'), ('CMS', 'SRE'), ('AAPL', 'MSFT'), ('XEL', 'CMS'),
        ('PFE', 'MCK'), ('NUE', 'MCD'), ('T', 'CPB'), ('YUM', 'BK'),
    ],
    'group_b': [
        ('PAYX', 'LIN'), ('ULTA', 'IP'), ('CCI', 'SWK'), ('BA', 'MNST'),
        ('BRO', 'ITW'), ('JKHY', 'CVX'), ('HST', 'CCI'), ('OXY', 'AEE'),
    ],
}


def independent_pair_frame(prices, x, y, start, end):
    """Reference uses sklearn fits, not production rolling/make_pair_frame."""
    left = prices.index.searchsorted(pd.Timestamp(start), side='left')
    right = prices.index.searchsorted(pd.Timestamp(end), side='right')
    warm = max(0, left - 524)
    frame = np.log(prices.iloc[warm:right][[x, y]]).copy()
    betas = np.full(len(frame), np.nan)
    spreads = np.full(len(frame), np.nan)
    fitted = missing = negative_beta = 0
    for i in range(503, len(frame)):
        history = frame.iloc[i - 503:i + 1]
        if not np.isfinite(history.to_numpy()).all():
            missing += 1
            continue
        design = np.column_stack([history[x].to_numpy(), np.ones(504)])
        dependent = history[y].to_numpy()
        regression = LinearRegression().fit(design, dependent)
        betas[i] = regression.coef_[0]
        spreads[i] = dependent[-1] - regression.intercept_ - betas[i] * design[-1, 0]
        fitted += 1
        negative_beta += int(betas[i] < 0)
    frame['BETA'], frame['SPREAD'] = betas, spreads
    return frame, fitted, missing, negative_beta


def validate(prices_path, source_path, output_path):
    prices_path, source_path, output_path = map(Path, (prices_path, source_path, output_path))
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    price_hash = hashlib.sha256(prices_path.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256 or price_hash != PRICE_SHA256:
        raise ValueError('Pinned public source or raw-price hash differs')
    prices = pd.read_parquet(prices_path)
    checks = []
    home_before = os.environ.get('HOME')
    for start, end in INTERVALS:
        for group, pairs in PAIR_GROUPS.items():
            fitted = missing = negative_beta = 0
            with TemporaryDirectory(prefix='independent-ols-source-') as tmp:
                home = Path(tmp)
                (home / 'spread/pairs').mkdir(parents=True)
                for x, y in pairs:
                    frame, n_fit, n_missing, n_negative = independent_pair_frame(prices, x, y, start, end)
                    fitted += n_fit
                    missing += n_missing
                    negative_beta += n_negative
                    frame.to_csv(home / 'spread/pairs' / f'{x}_{y}.csv', date_format='%Y-%m-%d')
                with original_performance(source_path, home, [x + '-' + y for x, y in pairs]) as original:
                    reference = original.Performance.run_trade((start, end, 'MAX_WEIGHT'))
                vector, metadata = trade_interval(prices, pairs, start, end)
                candidate = vector.returns
                dates_equal = np.array_equal(
                    reference.index.to_numpy(dtype='datetime64[ns]'),
                    candidate.index.to_numpy(dtype='datetime64[ns]'),
                )
                nan_equal = np.array_equal(reference.isna(), candidate.isna())
                allclose = np.allclose(reference, candidate, rtol=0, atol=1e-10, equal_nan=True)
                error = float(np.nanmax(np.abs(reference.to_numpy() - candidate.to_numpy()))) if reference.notna().any() else 0.
                checks.append({
                    'start': start, 'end': end, 'sample_group': group,
                    'pairs': pairs, 'pair_count': len(pairs), 'rows': len(reference),
                    'independent_daily_regressions': fitted,
                    'missing_windows_preserved': missing,
                    'negative_beta_windows': negative_beta,
                    'max_abs_return_error': error,
                    'date_values_equal_after_ns_normalization': bool(dates_equal),
                    'source_index_dtype': str(reference.index.dtype),
                    'vector_index_dtype': str(candidate.index.dtype),
                    'nan_mask_equal': bool(nan_equal),
                    'passed': bool(dates_equal and nan_equal and allclose),
                    'vector_metadata': metadata,
                })
    shared_input_check = verify_original_sample(
        prices, [('GPC', 'MMM'), ('CMS', 'SRE'), ('AAPL', 'MSFT')],
        '2017-12-29', '2018-01-31', source_path,
    )
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_repository': 'https://github.com/jingmouren/kai-trading-bot-pair',
        'source_commit': 'aaac6d999d6b077304a887caff2aab514d7ea6a2',
        'source_sha256': source_hash, 'price_sha256': price_hash,
        'trading_sha256': hashlib.sha256((ROOT / 'src/trading.py').read_bytes()).hexdigest(),
        'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference': 'Independent sklearn trailing-504 daily log-OLS with constant column and intercept; temporary pair CSVs passed to unedited public Performance.run_trade. No production rolling_endpoint or make_pair_frame is used by these six independent cases.',
        'sampling': 'Fixed 16 oriented pairs across 3 intervals, including actual missing-price windows and negative beta estimates. Pair samples are independent of portfolio selection; this is not a selection validation.',
        'pair_months': sum(c['pair_count'] for c in checks),
        'independent_daily_regressions': sum(c['independent_daily_regressions'] for c in checks),
        'missing_windows_preserved': sum(c['missing_windows_preserved'] for c in checks),
        'negative_beta_windows': sum(c['negative_beta_windows'] for c in checks),
        'absolute_tolerance': 1e-10, 'checks': checks,
        'max_abs_return_error': max(c['max_abs_return_error'] for c in checks),
        'shared_input_arithmetic_check': shared_input_check,
        'environment_HOME_unchanged': os.environ.get('HOME') == home_before,
        'temporary_pair_csvs': 'Automatically removed; operating-system default temporary directory.',
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'thread_environment': {key: os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']},
        'limitations': [
            'The daily pair CSV writer is absent. Rolling-504 endpoint OLS is a declared completion assumption, not recovered author code.',
            'Same-day beta and capital, same-day month-end selection, and duplicated inclusive month ends remain original source behavior. Numerical equivalence does not establish an ex-ante backtest.',
            'The public active run_trade uses Q only. Neither a Z strategy nor transaction costs are invented.',
            'These are bounded independent samples, not an exhaustive proof of all pair-day arithmetic or original author cache equivalence.',
        ],
    }
    report['passed'] = all(c['passed'] for c in checks) and shared_input_check['passed'] and report['environment_HOME_unchanged']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise AssertionError(f'Trading validation failed; see {output_path}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=ROOT / 'data/prices.parquet.gz')
    parser.add_argument('--source', type=Path, default=ROOT / 'source/performance.py')
    parser.add_argument('--output', type=Path, default=ROOT / 'checks/trading_validation.json')
    args = parser.parse_args()
    result = validate(args.prices, args.source, args.output)
    print(json.dumps({key: result[key] for key in [
        'passed', 'pair_months', 'independent_daily_regressions',
        'missing_windows_preserved', 'negative_beta_windows', 'max_abs_return_error',
    ]}, indent=2))
