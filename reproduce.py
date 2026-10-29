"""Reproduce the available author Q-code route and verify its outputs.

Default: reuse only dependency-and-output-hash-verified caches.
--fresh: recompute all 77 selections and 154 strategy intervals from prices.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--selection-workers', type=int, default=4)
    args = parser.parse_args()
    if min(args.workers, args.selection_workers) < 1:
        parser.error('Worker counts must be positive')
    env = os.environ.copy()
    env.update(PYTHONHASHSEED='0', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    checks = ROOT/'checks'
    checks.mkdir(exist_ok=True)
    selection_count = len(list((ROOT/'data/selection').glob('*.parquet')))
    trading_count = len(list((ROOT/'results/intervals').glob('*.parquet')))
    empty = selection_count == trading_count == 0
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'src/selection.py', ROOT/'src/trading.py', ROOT/'src/run.py']}
    start = {'created_at': datetime.now(timezone.utc).isoformat(), 'model_id': 'author_code_aaac6d9_q_replay',
             'fresh_selection': args.fresh or empty, 'fresh_trading': args.fresh or empty,
             'preexisting_caches_copied': False, 'selection_files_before_initial_launch': selection_count,
             'trading_files_before_initial_launch': trading_count, 'force_recompute': args.fresh,
             'input_price_sha256': hashlib.sha256((ROOT/'data/prices.parquet.gz').read_bytes()).hexdigest(),
             'calculation_files': code}
    (checks/'run_start.json').write_text(json.dumps(start, indent=2))
    cmd = [sys.executable, '-u', str(ROOT/'src/run.py'), '--workers', str(args.workers),
           '--selection-workers', str(args.selection_workers)]
    if args.fresh:
        cmd.append('--force')
    with (checks/'reproduction.log').open('w') as log:
        commands = [[sys.executable, str(ROOT/'src/selection.py'), '--validate'],
                    [sys.executable, str(ROOT/'src/validate_trading.py')],
                    cmd, [sys.executable, str(ROOT/'src/verify.py')],
                    [sys.executable, str(ROOT/'src/report.py')]]
        for command in commands:
            print('Running:', ' '.join(command), flush=True)
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed. Read README.md and reports/REPORT_KO.html', flush=True)


if __name__ == '__main__':
    main()
