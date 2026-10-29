"""Audit original-source identity, all interval outputs, and reproduction checks."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    from selection import cache_identity, environment, selection_cutoffs
    source = json.loads((ROOT/'checks/source_manifest.json').read_text())
    for record in source['verified_files']:
        data = (ROOT/record['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == record['git_blob_sha1']
    assert len(source['verified_files']) == 11
    prices_path = ROOT/'data/prices.parquet.gz'
    price_hash = digest(prices_path)
    boundaries = selection_cutoffs(pd.read_parquet(prices_path))
    selection_receipts = sorted((ROOT/'data/selection').glob('*.json'))
    trade_receipts = sorted((ROOT/'results/intervals').glob('*.json'))
    assert len(selection_receipts) == len(trade_receipts) == 77
    assert len(list((ROOT/'results/intervals').glob('*.parquet'))) == 154
    expected_keys = {pd.Timestamp(d).strftime('%Y%m%d') for d in boundaries[:-1]}
    assert {p.stem for p in selection_receipts} == {p.stem for p in trade_receipts} == expected_keys
    monthly = []
    max_accounting_error = 0.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        key = pd.Timestamp(start).strftime('%Y%m%d')
        selected_path = ROOT/'data/selection'/f'{key}.parquet'
        selected = pd.read_parquet(selected_path)
        meta = json.loads((ROOT/'data/selection'/f'{key}.json').read_text())
        assert meta['cache_identity'] == cache_identity(start, prices_path), key
        assert meta['selection_sha256'] == digest(selected_path), key
        assert meta['window_rows'] == 504
        b = selected[selected.METHOD.eq('BASELINE')]
        m = selected[selected.METHOD.eq('MAX_WEIGHT')]
        assert len(b) == meta['baseline_n'] and len(m) == meta['matching_n']
        assert b.P_VALUE.le(.05).all()
        assert pd.concat([m.ASSET_1,m.ASSET_2]).is_unique
        trade = json.loads((ROOT/'results/intervals'/f'{key}.json').read_text())
        identity = trade['cache_identity']
        assert identity == {'start': str(pd.Timestamp(start).date()), 'end': str(pd.Timestamp(end).date()),
                            'price_sha256': price_hash, 'selection_sha256': digest(selected_path),
                            'run_sha256': digest(ROOT/'src/run.py'), 'trading_sha256': digest(ROOT/'src/trading.py'),
                            'source_performance_sha256': digest(ROOT/'source/performance.py'), 'environment': environment()}
        for method in ['MAX_WEIGHT','BASELINE']:
            path = ROOT/'results/intervals'/f'{key}_{method}.parquet'
            assert digest(path) == trade['output_sha256'][method]
            daily = pd.read_parquet(path)
            assert not daily.index.has_duplicates
            assert daily.index.min() == start and daily.index.max() == end
            assert np.isfinite(daily.returns.dropna()).all()
            np.testing.assert_allclose(daily.returns, daily.pnl/daily.capital, rtol=0, atol=1e-14, equal_nan=True)
            max_accounting_error = max(max_accounting_error, float(np.nanmax(np.abs(daily.returns-daily.pnl/daily.capital))))
        monthly.append({'cutoff': key, 'baseline_pairs': len(b), 'matching_pairs': len(m), 'passed': True})
    metrics = pd.read_csv(ROOT/'results/metrics.csv')
    assert len(metrics) == 4
    references = {('source_default','MAX_WEIGHT'): (.7741785180404984,.20112160396484624),
                  ('source_default','BASELINE'): (.32170907570191076,.08104642870162193),
                  ('extended_2017','MAX_WEIGHT'): (.5982393540892392,.2538646891106251),
                  ('extended_2017','BASELINE'): (.29981357424436084,.12384592698052588)}
    comparison = []
    for row in metrics.to_dict('records'):
        daily = pd.read_csv(ROOT/'results'/f"{row['scope']}_{row['method']}.csv", parse_dates=['date'])
        starts = [s for s in boundaries[:-1] if row['scope']=='extended_2017' or pd.Timestamp(s).year>=2020]
        frames = [pd.read_parquet(ROOT/'results/intervals'/f"{pd.Timestamp(s):%Y%m%d}_{row['method']}.parquet") for s in starts]
        joined = pd.concat(frames).sort_index().loc[row['start']:row['end']].reset_index()
        for col in ['date','formation_cutoff']:
            daily[col] = pd.to_datetime(daily[col]).astype('datetime64[ns]')
            joined[col] = pd.to_datetime(joined[col]).astype('datetime64[ns]')
        pd.testing.assert_frame_equal(daily, joined, check_exact=False, rtol=1e-12, atol=1e-10)
        r = daily.returns
        expected_unique, expected_duplicate = {'source_default': (839,39), 'extended_2017': (1613,76)}[row['scope']]
        assert daily.date.nunique() == expected_unique and daily.date.duplicated().sum() == expected_duplicate
        assert len(daily) == row['rows_including_duplicate_dates']
        np.testing.assert_allclose(row['sharpe'], np.sqrt(252)*r.mean()/r.std(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(row['cumulative_return'], (1+r.fillna(0)).prod()-1, rtol=0, atol=1e-12)
        reference_sharpe, reference_return = references[(row['scope'],row['method'])]
        np.testing.assert_allclose([row['sharpe'],row['cumulative_return']], [reference_sharpe,reference_return], rtol=0, atol=1e-10)
        comparison.append({'scope':row['scope'],'method':row['method'],
                           'sharpe_difference':row['sharpe']-reference_sharpe,
                           'cumulative_return_difference':row['cumulative_return']-reference_return})
    sv = json.loads((ROOT/'checks/selection_validation.json').read_text())
    tv = json.loads((ROOT/'checks/trading_validation.json').read_text())
    assert sv['passed'] and tv['passed']
    assert sv['generator_sha256'] == digest(ROOT/'src/selection.py')
    assert sv['environment'] == environment()
    assert tv['trading_sha256'] == digest(ROOT/'src/trading.py')
    assert tv['validator_sha256'] == digest(ROOT/'src/validate_trading.py')
    assert sv['price_sha256'] == tv['price_sha256'] == price_hash
    import sklearn
    assert tv['versions'] == {'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}
    start = json.loads((ROOT/'checks/run_start.json').read_text())
    for filename, sha in start['calculation_files'].items():
        assert digest(ROOT/filename) == sha, filename
    execution = json.loads((ROOT/'checks/execution.json').read_text())
    assert execution['intervals'] == 77
    result = {'passed':True,'status':'completed','model_id':'author_code_aaac6d9_q_replay',
              'completed_at':datetime.now(timezone.utc).isoformat(), 'source_commit':source['commit'],
              'selection_months':77,'intervals':77,'monthly_strategy_paths':154,
              'source_files_verified':11,'price_sha256':price_hash,
              'fresh_selection':start['fresh_selection'], 'fresh_trading':execution['recomputed_intervals']==77,
              'preexisting_caches_copied':False,'input_caches_and_output_hashes_verified':True,
              'aggregate_csv_all_rows_match_verified_intervals':True,
              'maximum_pnl_div_capital_error':max_accounting_error,'selection_reference_samples':sv['samples'],
              'independent_selection_validation_passed':True,'independent_trading_validation_passed':True,
              'previous_reproduction_metric_comparison':comparison,
              'reference_note':'Four metric values transcribed from the earlier aaac6d9 conditional replay; old files were removed before this task. They are regression checks, not paper targets or parameter-tuning targets.',
              'monthly_checks':monthly,'scope':'Completed public-code arithmetic under declared missing daily-cache assumption; not exact reproduction of paper tables.'}
    (ROOT/'checks/fresh_reproduction.json').write_text(json.dumps(result,indent=2))
    if result['fresh_selection'] and result['fresh_trading']:
        (ROOT/'checks/full_recomputation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='monthly_checks'},indent=2))
    return result


if __name__ == '__main__':
    verify()
