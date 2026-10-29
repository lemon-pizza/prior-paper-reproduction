"""Reproduce the fixed research baseline, its controls, and independent checks."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
PRICE_SHA = '7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-selection', action='store_true',
                        help='Recalculate all 77 monthly graphs from raw prices instead of using retained selections')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if sha(ROOT/'data/prices.parquet.gz') != PRICE_SHA:
        raise ValueError('The fixed baseline price input has changed')
    (ROOT/'checks').mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(Path(tempfile.gettempdir())/'matching-baseline-matplotlib'))
    stages = []
    with (ROOT/'checks/reproduction.log').open('w') as log:
        def run(name, *extra):
            print('RUN', name, flush=True)
            log.write('RUN '+name+' '+' '.join(extra)+'\n');log.flush()
            result = subprocess.run([sys.executable, str(ROOT/'src'/name), *extra], cwd=ROOT,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print((ROOT/'checks/reproduction.log').read_text()[-6000:], file=sys.stderr)
                raise SystemExit(result.returncode)
            stages.append(name)
        selection_args = ['--all', '--validate', '--workers', str(args.workers)]
        if args.rebuild_selection: selection_args.append('--force')
        run('selection.py', *selection_args)
        run('backtest.py')
        run('verify.py')
        run('analyze.py', '--require-clean')
    # Promote only an arithmetic-preserving result. This reference contains
    # checksums, not copies of the previous data or code.
    reference = json.loads((ROOT/'checks/migration_reference.json').read_text())
    compared = []; mismatches = []
    core = {'results/metrics.csv', 'results/comparisons.csv', 'results/bootstrap.csv',
            'results/annual_primary.csv', 'results/signal_summary.csv'}
    for name, expected in reference['result_sha256'].items():
        if name in core or name.startswith(('results/daily/', 'results/weights/')):
            compared.append(name)
            if sha(ROOT/name) != expected: mismatches.append(name)
    migration = dict(passed=not mismatches, compared_files=len(compared),
                     byte_identical=compared, mismatches=mismatches,
                     scope='Retained fixed-baseline daily paths, weights, statistics and signal summaries')
    (ROOT/'checks/migration_comparison.json').write_text(json.dumps(migration, indent=2)+'\n')
    if mismatches: raise RuntimeError('Baseline outputs changed: '+', '.join(mismatches))
    versions = {name: version(name) for name in
                ['numpy', 'pandas', 'scipy', 'statsmodels', 'networkx', 'rustworkx', 'pyarrow', 'scikit-learn', 'matplotlib']}
    files = []
    for folder in ['src', 'config', 'data', 'results']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                files.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)))
    record = dict(passed=True, baseline_id='matching_q_v1', completed_at=datetime.now(timezone.utc).isoformat(),
                  stages=stages, rebuilt_selection=args.rebuild_selection, price_sha256=PRICE_SHA,
                  trading_days=1613, monthly_selections=77, execution_paths=45,
                  primary='weighted M versus same-K top-score B, stock gross 1, delay 1, cost 5bp',
                  packages=versions, python=sys.version, unchanged_reference_outputs=len(compared), files=files)
    (ROOT/'checks/reproduction_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print('COMPLETE: 77 selections, 45 execution paths, independent verification, report and baseline equality checks.', flush=True)


if __name__ == '__main__': main()
