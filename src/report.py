"""Generate the single-model README, HTML report and figures from saved outputs.

Run at the repository root: python src/report.py. Never executes notebook cells
or the backtest. Historical notebook outputs remain separate references.
"""
from __future__ import annotations
import hashlib
import html
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'author_code_aaac6d9_q_replay'
COMMIT = 'aaac6d999d6b077304a887caff2aab514d7ea6a2'
REPO = 'https://github.com/jingmouren/kai-trading-bot-pair'
PAPER = 'https://arxiv.org/html/2403.07998v1#S5.SS2'
SCOPES = {'source_default': '원코드 기본 기간', 'extended_2017': '2017년까지 확장한 보조 실행'}
METHODS = {'MAX_WEIGHT': 'M: 실제 무가중 매칭', 'BASELINE': 'B: p≤0.05 전체'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def notebook_evidence():
    evidence = {'repository': REPO, 'commit': COMMIT,
                'evidence_type': 'historical notebook saved output; no cells executed',
                'comparability': 'Exact input/cache/date range unknown; not verified as output of the visible Python source at this commit.',
                'notebooks': {}}
    for name in ['QSCORE.ipynb', 'clean_zscore.ipynb', 'report.ipynb']:
        raw = (ROOT / 'source' / name).read_bytes()
        nb = json.loads(raw)
        plain, tables = [], []
        for cell in nb['cells']:
            for output in cell.get('outputs', []):
                data = output.get('data', {})
                text = data.get('text/plain', [])
                plain.append(''.join(text) if isinstance(text, list) else text)
                text = data.get('text/html', [])
                text = ''.join(text) if isinstance(text, list) else text
                for table in re.findall(r'<table\b.*?</table>', text, re.S):
                    rows = []
                    for row in re.findall(r'<tr\b.*?</tr>', table, re.S):
                        cells = [html.unescape(re.sub(r'<[^>]*>', '', c)).strip()
                                 for c in re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', row, re.S)]
                        if cells:
                            rows.append(cells)
                    tables.append(rows)
        stats = {}
        for key in ['SHARPE', 'SORTINO', 'CUMULATIVE_RETURN', 'SIMPLE_RETURN']:
            match = re.search(r'\b' + key + r'\s+([-+]?\d+(?:\.\d+)?)', '\n'.join(plain))
            if match:
                stats[key] = float(match.group(1))
        evidence['notebooks'][name] = {
            'url': f'{REPO}/blob/{COMMIT}/{name}', 'sha256': hashlib.sha256(raw).hexdigest(),
            'python_metadata': nb.get('metadata', {}).get('language_info', {}).get('version'),
            'cells_source': [''.join(c.get('source', [])) for c in nb['cells']],
            'saved_statistics': stats, 'saved_tables': tables,
            'date_extent': 'Exact dates unknown. Saved plot ticks near 2020-01 to 2023-05.'
                           if name != 'clean_zscore.ipynb' else 'Not shown in saved output'}
    evidence['incompatibilities_with_visible_source'] = [
        'clean_zscore calls fetch_performance, absent from performance.py.',
        'run_history references undefined run_trade/perf and does not return returns.',
        'QSCORE contains a saved plot although current simple_report only returns a DataFrame.',
        'QSCORE prose uses Q75-Q25; active Python code uses Q25-Q75.',
        'report compares four selection styles, not paper MQ/MZ/BQ/BZ.']
    write_json(ROOT / 'checks/notebook_reference.json', evidence)
    return evidence


def comparison_rows(metrics, evidence):
    rows = []
    for row in metrics.to_dict('records'):
        rows.append(dict(dataset='conditional_source_execution', evidence_type='computed_outputs',
                         strategy=row['method'], scope=row['scope'], start=row['start'], end=row['end'],
                         sharpe=row['sharpe'], sortino=row['sortino'], cumulative_return=row['cumulative_return'],
                         simple_return=row['simple_return'], unique_dates=row['unique_dates'],
                         calculation_rows=row['rows_including_duplicate_dates'], duplicate_rows=row['duplicate_rows'],
                         source_url='', local_source='results/metrics.csv',
                         comparison_limit='Unweighted M; all significant B pairs; duplicate endpoints; assumed daily OLS cache; Q only; gross.'))
    for strategy, sharpe, sortino, cumulative in [('MQ', 1.23, 1.78, .7365), ('BQ', .34, .44, .1796)]:
        rows.append(dict(dataset='paper_table2_gross', evidence_type='published_reference', strategy=strategy,
                         scope='paper_2017_2023', start='2017-01 (month only)', end='2023-05 (month only)',
                         sharpe=sharpe, sortino=sortino, cumulative_return=cumulative, source_url=PAPER,
                         local_source='', comparison_limit='Paper weighted M / top-250 B; different trading rules; not equivalent conditions.'))
    for name, label in [('QSCORE.ipynb', 'MAX_WEIGHT_Q_saved'), ('clean_zscore.ipynb', 'MAX_WEIGHT_Z_saved')]:
        record = evidence['notebooks'][name]
        stats = record['saved_statistics']
        rows.append(dict(dataset='notebook_saved_output', evidence_type='historical_saved_output', strategy=label,
                         scope='exact_dates_unknown', start='', end='', sharpe=stats.get('SHARPE'),
                         sortino=stats.get('SORTINO'), cumulative_return=stats.get('CUMULATIVE_RETURN'),
                         simple_return=stats.get('SIMPLE_RETURN'), source_url=record['url'],
                         local_source=f'source/{name}', comparison_limit='Saved output not generated by visible commit alone; input/cache/exact dates unknown.'))
    frame = pd.DataFrame(rows)
    frame.insert(0, 'model', MODEL)
    frame.to_csv(ROOT / 'results/reference_comparison.csv', index=False)
    return frame


def make_chart(metrics):
    with TemporaryDirectory(prefix='author-reproduction-mpl-') as temp:
        os.environ['MPLCONFIGDIR'] = temp
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import dates as mdates
        from matplotlib.ticker import PercentFormatter
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'text.color': '#202a36', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
        colors = {'MAX_WEIGHT': '#0072B2', 'BASELINE': '#D55E00'}
        for ax, scope, title in zip(axes, SCOPES, ['A  Source-default period', 'B  Extended period (supplement)']):
            for method in METHODS:
                daily = pd.read_csv(ROOT / f'results/{scope}_{method}.csv', parse_dates=['date'])
                wealth = (1 + daily['returns'].fillna(0)).cumprod() - 1
                expected = metrics.loc[(metrics.scope == scope) & (metrics.method == method)].iloc[0]
                if len(daily) != expected.rows_including_duplicate_dates or not np.isclose(
                        wealth.iloc[-1], expected.cumulative_return, rtol=1e-10, atol=1e-12):
                    raise ValueError(f'Daily CSV and metrics disagree: {scope} / {method}')
                label = 'M: unweighted matching' if method == 'MAX_WEIGHT' else 'B: all p <= 0.05'
                ax.plot(daily.date, wealth, label=label, color=colors[method], lw=1.55)
                ax.annotate(f'{wealth.iloc[-1]:.1%}', (daily.date.iloc[-1], wealth.iloc[-1]),
                            xytext=(-5, 7), textcoords='offset points', ha='right', color=colors[method],
                            fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
            ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=10)
            ax.axhline(0, color='#8c939c', lw=.65)
            ax.grid(axis='y', color='#dce1e7', lw=.65)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.margins(x=.035)
            row = metrics.loc[metrics.scope == scope].iloc[0]
            ax.text(0, -.16, f'{row.start} to {row.end} | {int(row.unique_dates):,} unique dates; '
                    f'{int(row.duplicate_rows)} repeated month-end rows', transform=ax.transAxes,
                    fontsize=8.5, color='#59616c')
        axes[0].set_ylabel('Gross cumulative return')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045), ncol=2, frameon=False, fontsize=10)
        fig.suptitle('Available author code: conditional Q-signal reproduction', x=.075, ha='left', fontsize=15, fontweight='bold')
        fig.text(.075, .018, 'Missing daily cache supplied by rolling 504-observation OLS. Original duplicate endpoints retained. No transaction costs.',
                 fontsize=8, color='#59616c')
        fig.subplots_adjust(left=.075, right=.985, top=.85, bottom=.27, wspace=.13)
        fig.savefig(ROOT / 'reports/cumulative_returns.png', dpi=240)
        plt.close(fig)


def check_table():
    start_name = 'run_start.json' if (ROOT / 'checks/run_start.json').exists() else 'initial_state.json'
    start = read_json(ROOT / 'checks' / start_name)
    receipt = read_json(ROOT / 'checks/fresh_reproduction.json')
    execution = read_json(ROOT / 'checks/execution.json')
    selection_n = len(list((ROOT / 'data/selection').glob('*.json')))
    interval_n = len(list((ROOT / 'results/intervals').glob('*.json')))
    strategy_n = len(list((ROOT / 'results/intervals').glob('*.parquet')))
    declared = ((start.get('fresh_selection') is True and start.get('fresh_trading') is True and start.get('preexisting_caches_copied') is False)
                or start.get('previous_selection_or_trading_cache_copied') is False)
    receipt_counts = (receipt.get('selection_months'), receipt.get('intervals'), receipt.get('monthly_strategy_paths'))
    complete = receipt.get('passed') is True and receipt_counts == (77, 77, 154) and (selection_n, interval_n, strategy_n) == (77, 77, 154)
    mode = '선정·거래 전체 재계산' if declared else '검증된 캐시 재사용 가능; 실제 재계산 수는 execution.json 참조'
    rows = [('최근 실행 방식', mode, start_name),
            ('월별 선정 파일', f'{selection_n} / 77개', ''),
            ('월별 거래 완료 기록', f'{interval_n} / 77개', ''),
            ('월별 두 전략 수익 파일', f'{strategy_n} / 154개', ''),
            ('실행 종료 기록', str(execution.get('completed_at', '미확인')), 'execution.json'),
            ('독립 완료 확인서', '통과 확인' if complete else '대기 또는 조건 미충족', 'fresh_reproduction.json')]
    full = read_json(ROOT / 'checks/full_recomputation.json')
    if full.get('passed') and full.get('fresh_selection') and full.get('fresh_trading'):
        rows.append(('원가격 전체 재계산 기록', str(full.get('completed_at')), 'full_recomputation.json'))
    table = ['| 확인 항목 | 상태 | 근거 |', '|---|---|---|']
    for name, state, filename in rows:
        link = (f'[{filename}](checks/{filename})' if filename and (ROOT / 'checks' / filename).exists()
                else '생성 파일 개수' if not filename else '확인서 없음')
        table.append(f'| {name} | {state} | {link} |')
    return '\n'.join(table), complete


def independent_checks():
    lines = []
    excluded = {'notebook_reference.json', 'run_start.json', 'initial_state.json', 'execution.json', 'source_manifest.json',
                'fresh_reproduction.json', 'upstream_tree.json'}
    for path in sorted((ROOT / 'checks').glob('*.json')):
        if path.name in excluded:
            continue
        value = read_json(path)
        state = ('통과 표시' if isinstance(value, dict) and value.get('passed') is True
                 else '실패 표시 — 세부 결과 확인 필요' if isinstance(value, dict) and value.get('passed') is False
                 else '세부 결과 확인')
        lines.append(f'- [{path.name}](checks/{path.name}): {state}.')
    return '\n'.join(lines) or '- 추가 독립 검산 JSON이 아직 없습니다. 완료 확인서와 함께 추가 확인이 필요합니다.'


def make_readme(metrics, evidence, comparisons):
    table = ['| 실행 범위 | 전략 | 비용 전 누적수익 | 샤프 | 소르티노 | 고유 날짜 / 계산행 / 중복행 |',
             '|---|---|---:|---:|---:|---:|']
    for row in metrics.itertuples(index=False):
        table.append(f'| {SCOPES[row.scope]}<br>{row.start}~{row.end} | {METHODS[row.method]} | '
                     f'{row.cumulative_return:.2%} | {row.sharpe:.4f} | {row.sortino:.4f} | '
                     f'{row.unique_dates:,} / {row.rows_including_duplicate_dates:,} / {row.duplicate_rows:,} |')
    comparison = ['| 자료 | 전략·기간 | 샤프 | 복리 누적수익 |', '|---|---|---:|---:|']
    for row in comparisons.itertuples(index=False):
        if row.strategy == 'MAX_WEIGHT_Z_saved':
            continue
        label = {'conditional_source_execution': '이번 공개 코드 실행', 'paper_table2_gross': '논문 표 2 (gross)',
                 'notebook_saved_output': 'QSCORE 과거 저장 출력'}[row.dataset]
        period = SCOPES.get(row.scope, '2017-01~2023-05' if row.dataset == 'paper_table2_gross' else '정확한 날짜 불명')
        comparison.append(f'| {label} | {row.strategy}; {period} | {row.sharpe:.4f} | {row.cumulative_return:.2%} |')
    selections = [read_json(p) for p in sorted((ROOT / 'data/selection').glob('*.json'))]
    counts = pd.DataFrame([{k: x.get(k) for k in ['cutoff', 'eligible_n', 'candidate_n', 'baseline_n', 'matching_n']} for x in selections])
    if len(counts):
        counts.to_csv(ROOT / 'results/selection_counts.csv', index=False)
    def pair_range(column):
        if column not in counts or counts[column].dropna().empty:
            return '미집계'
        return f'{int(counts[column].min()):,}~{int(counts[column].max()):,}개'
    checks, passed = check_table()
    start_name = 'run_start.json' if (ROOT / 'checks/run_start.json').exists() else 'initial_state.json'
    state = '최신 결과의 검산 통과와 77개월 산출물 개수를 확인했습니다. 원가격 전체 재계산 이력과 최근 실행의 캐시 재사용 여부는 아래 기록에서 구분합니다.' if passed else '최신 결과의 독립 완료 확인은 아직 충족되지 않았습니다. 아래 검증표를 확인하세요.'
    q = evidence['notebooks']['QSCORE.ipynb']['saved_statistics']
    z = evidence['notebooks']['clean_zscore.ipynb']['saved_statistics']
    doc = f'''# 선행논문 공개 코드 재현

이 저장소의 유일한 현재 기준은 **`{MODEL}`**입니다. [Khizar Qureshi·Tauhid Zaman의 논문](https://arxiv.org/abs/2403.07998)이 공개한 코드의 [보존본 `{COMMIT[:7]}`]({REPO}/tree/{COMMIT})을 기준으로, 남아 있는 연산을 보존하고 누락된 실행 연결부와 일별 회귀 캐시를 보완했습니다. 결과는 **Q 신호 M/B 두 전략의 조건부 실행**이며, 논문 표 2·3의 정확한 재현을 뜻하지 않습니다. 이전 모델은 Git의 부모 커밋에만 보관됩니다.

[한국어 HTML 보고서](reports/REPORT_KO.html) · [성과 원표](results/metrics.csv) · [비교 CSV](results/reference_comparison.csv) · [전체 소스 출처](checks/source_manifest.json)

## 결과

{chr(10).join(table)}

`source_default`는 공개 `run_history`의 `202*.csv` 필터를 유지한 주 실행입니다. `extended_2017`은 같은 거래 연산에서 날짜 범위를 2017년까지 넓힌 보조 실행입니다. 표본 기간이 다르므로 두 실행의 성과를 직접 우열로 해석하지 않습니다. 샤프·소르티노·복리·단순 누적수익은 원코드의 산식과 결측 처리 규칙을 사용합니다.

![Q 신호 두 전략의 비용 전 누적수익](reports/cumulative_returns.png)

월별 거래 구간의 양끝을 모두 포함하기 때문에 같은 월말이 두 번 들어갑니다. 고유 날짜와 계산행 수를 함께 보아야 합니다. `max_drawdown_supplement`는 초기자산 1을 포함한 자산곡선에서 추가 계산한 보조 지표이며 원소스 출력이 아닙니다. 활성 코드에 거래비용 계산이 없어 비용 후 성과는 생성하지 않았습니다.

## 실행 확인과 독립 검산

{state}

{checks}

시작 선언, 파일 개수, 실행 종료 기록과 검산 결과는 서로 다른 증거입니다. 완료 여부는 [fresh_reproduction.json](checks/fresh_reproduction.json)의 `passed`와 실제 산출물을 함께 확인합니다. `data/prices.parquet.gz`에서 선정·거래를 생성하며, 다른 모델의 선정·신호 캐시는 입력으로 사용하지 않습니다.

독립 검산 자료:

{independent_checks()}

원격 커밋의 텍스트 파일 10개와 가격 파일 1개의 Git blob·SHA-256 식별은 [source_manifest.json](checks/source_manifest.json)의 `verified_files`에, 시작 기록은 [{start_name}](checks/{start_name})에 있습니다. 환경 버전은 [requirements.txt](requirements.txt)와 개별 검산 JSON을 참조합니다. 검산 통과는 기록된 표본과 연산 대조의 범위이며, 누락된 저자 캐시 자체를 복구했다는 뜻은 아닙니다.

## 논문 및 노트북과 비교

{chr(10).join(comparison)}

**서로 다른 조건의 수치를 병기한 표**입니다. 정확한 재현 오차나 투자전략의 성과 순위로 해석하지 않습니다. [reference_comparison.csv](results/reference_comparison.csv)는 자료 종류·기간·출처·조건 차이를 행마다 기록합니다. CSV 수익률의 단위는 1=100%입니다.

- [논문 표 2]({PAPER})의 MQ/BQ는 2017년 1월~2023년 5월 gross 결과입니다. MQ 누적수익은 73.65%, BQ는 17.96%입니다. 논문 M은 가중 매칭, B는 상위 250개 페어여서 이번 원코드 실행의 선정 방식과 다릅니다.
- [QSCORE.ipynb](source/QSCORE.ipynb) 과거 저장 출력은 샤프 {q.get('SHARPE', float('nan')):.6f}, 누적수익 {q.get('CUMULATIVE_RETURN', float('nan')):.4%}입니다. 저장 그래프 눈금은 2020-01~2023-05 부근이지만 정확한 날짜와 입력 캐시가 없습니다. 해당 커밋의 실행 함수에는 미정의 이름·반환 누락이 있고, 저장 그래프와 현재 보고 함수도 달라 동일 환경의 정답 값으로 확정할 수 없습니다.
- [clean_zscore.ipynb](source/clean_zscore.ipynb)의 샤프 {z.get('SHARPE', float('nan')):.6f}, 누적수익 {z.get('CUMULATIVE_RETURN', float('nan')):.4%}도 과거 저장 출력입니다. 호출하는 `fetch_performance`가 공개 소스에 없어 이번 Z 재현 결과로 사용하지 않습니다.
- [report.ipynb](source/report.ipynb)의 BASELINE/MAX_WEIGHT/MAXIMAL/APPROX_MAXIMAL은 매칭 방식별 과거 캐시 보고서입니다. 논문 MQ/MZ/BQ/BZ와 구분이 다릅니다. 저장 텍스트·표·호출식은 [notebook_reference.json](checks/notebook_reference.json)에 있으며, 노트북 셀은 실행하지 않았습니다.

## 보존한 연산과 보완 가정

| 항목 | 이번 실행 | 논문 및 재현 해석 |
|---|---|---|
| M 선정 | `rating`을 매칭 함수의 `weight`로 전달하지 않는 원동작 유지; 월별 {pair_range('matching_n')} | 실제로는 무가중 최대 개수 매칭 |
| B 선정 | p≤0.05인 모든 방향별 페어; 월별 {pair_range('baseline_n')} | 상위 250개 선정·역방향 중복 제거 없음 |
| 후보·ADF | `betas.py`의 양방향 페어, 전체 기간 수익률 결측 5% 미만 유니버스, 기본 AIC 자동시차 ADF | 논문의 단일 시차 설명과 다름 |
| 일별 SPREAD·BETA | 504개 관측치의 rolling 로그 OLS에서 당일 잔차·계수 생성 | **공개되지 않은 생성기에 대한 보완 가정** |
| Q | 20일 중앙값·사분위수, Q25−Q75 분모, sign×round, 1행 지연 | 원코드 부호·반올림·shift 유지 |
| 손익·자본 | 당일 β의 주식 수 비율, 당일 가격 자본 분모, 원본 결측 처리 | 실전 체결·회계 검증과 별개 |
| 월말 | 양끝 포함 구간을 이어 붙여 중복 유지 | 같은 날 수익이 두 번 복리 반영될 수 있음 |
| Z·거래비용 | 활성 경로가 없거나 호출 함수 누락 | 임의로 추가하지 않음 |

실행 연결부 보완에는 로컬 경로, 누락 페어 reader, 미정의 함수 연결·반환값·실행 순서가 포함됩니다. 계산 가속용 벡터화는 독립 원함수 대조 자료와 함께 제공합니다. Python 해시 시드 0으로 현재 실행의 매칭 순서를 고정했지만, 저자의 당시 환경과 동점 처리 순서까지 복구한 것은 아닙니다.

월말까지의 자료로 선정한 페어가 같은 날의 전날 대비 수익에도 적용됩니다. 당일 β·자본 분모, 전체 기간 유니버스와 중복 월말도 남아 있습니다. 따라서 이 결과는 공개 소스의 동작을 확인하는 연구 자료이며, 미래정보를 배제한 투자수익이나 매칭의 일반적 우월성을 입증하지 않습니다.

## 재실행

저장소 최상위 폴더에서 실행합니다. `reproduce.py` 기본 실행은 해시 검증을 거친 기존 캐시를 재사용해 집계·검산·보고서를 갱신합니다.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

원가격부터 선정 77개월과 M/B 거래를 **전부 새로 계산**하려면 `--fresh`를 사용합니다. 이 명령은 같은 출력 위치의 생성 캐시와 결과를 갱신합니다.

```sh
python reproduce.py --fresh --workers 2 --selection-workers 4
```

기존 결과의 README·HTML·그림만 갱신하려면 `python src/report.py`를 실행합니다. `src/run.py`는 하위 실행 경로입니다. 자세한 옵션은 `python reproduce.py --help`에서 확인합니다. Python·라이브러리 환경은 `requirements.txt`와 개별 검산 JSON을 참조합니다.

| 경로 | 역할 |
|---|---|
| [reproduce.py](reproduce.py) | 기본: 검증 캐시 재사용; `--fresh`: 전체 재계산·검산·보고서 생성 |
| [src](src/) | 선정·거래·집계·검산·보고서 코드 |
| [source](source/) | 고정 커밋의 원본 Python·노트북·라이선스 10개 텍스트 파일 |
| [data/prices.parquet.gz](data/prices.parquet.gz) | 고정 원가격 입력 |
| [data/selection](data/selection/) | 새로 계산한 77개월 선정 |
| [results](results/) | 월별 두 전략의 수익, 두 기간의 일별 CSV, 성과·비교표 |
| [checks](checks/) | 출처·시작 선언·독립 검산·완료 근거 |
| [reports](reports/) | 한국어 HTML 보고서와 누적수익 PNG |

원가격 SHA-256: `7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c`.

원소스 라이선스는 [source/LICENSE](source/LICENSE)에 보존합니다. 코드와 가격 파일의 출처는 위 manifest를 따릅니다.
'''
    (ROOT / 'README.md').write_text(doc, encoding='utf-8')
    return doc


def inline_html(text):
    escaped = html.escape(text).replace('&lt;br&gt;', '<br>')
    def image_tag(match):
        alt, target = match.groups()
        target = target if target.startswith(('http:', 'https:')) else '../' + target
        return f'<img src="{target}" alt="{alt}">'
    escaped = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', image_tag, escaped)
    def link_tag(match):
        label, target = match.groups()
        target = target if target.startswith(('http:', 'https:')) else '../' + target
        return f'<a href="{target}">{label}</a>'
    escaped = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_tag, escaped)
    escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', escaped)


def make_html(markdown):
    """Render the controlled Markdown subset above without extra dependencies."""
    lines = markdown.splitlines()
    body, paragraphs = [], []
    code, bullets = None, False
    def flush():
        if paragraphs:
            body.append('<p>' + inline_html(' '.join(paragraphs)) + '</p>')
            paragraphs.clear()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            flush()
            if code is None:
                code = []
            else:
                body.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
                code = None
            i += 1
            continue
        if code is not None:
            code.append(line)
            i += 1
            continue
        if bullets and not line.startswith('- '):
            body.append('</ul>')
            bullets = False
        if not line.strip():
            flush()
        elif line.startswith('#'):
            flush()
            level = len(line) - len(line.lstrip('#'))
            body.append(f'<h{level}>' + inline_html(line[level:].strip()) + f'</h{level}>')
        elif line.startswith('|'):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                row = [c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in row):
                    rows.append(row)
                i += 1
            body.append('<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + inline_html(c) + '</th>' for c in rows[0]) + '</tr></thead><tbody>')
            for row in rows[1:]:
                body.append('<tr>' + ''.join('<td>' + inline_html(c) + '</td>' for c in row) + '</tr>')
            body.append('</tbody></table></div>')
            continue
        elif line.startswith('- '):
            flush()
            if not bullets:
                body.append('<ul>')
                bullets = True
            body.append('<li>' + inline_html(line[2:]) + '</li>')
        else:
            paragraphs.append(line)
        i += 1
    flush()
    if bullets:
        body.append('</ul>')
    css = '''
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; color: #202a36; background: #f2f4f7; }
body { margin: 0; line-height: 1.7; font-size: 15px; }
main { max-width: 1080px; margin: 36px auto; background: #fff; padding: 52px 60px; border: 1px solid #dde3ea; }
.eyebrow { color: #356493; font-size: 12px; letter-spacing: .06em; font-weight: 700; }
h1 { margin: 12px 0 22px; font-size: 32px; letter-spacing: -.04em; }
h2 { margin: 40px 0 14px; padding-top: 10px; font-size: 22px; letter-spacing: -.03em; border-top: 1px solid #e3e8ee; }
p { margin: 12px 0; } a { color: #165e9f; text-underline-offset: 3px; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .87em; background: #f1f4f8; padding: 2px 4px; overflow-wrap: anywhere; }
pre { padding: 16px 20px; background: #f1f4f8; overflow-x: auto; border-left: 3px solid #0072b2; }
pre code { padding: 0; } img { max-width: 100%; display: block; height: auto; margin: 22px auto; }
.table-wrap { overflow-x: auto; margin: 18px 0; } table { width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.55; }
th { text-align: left; background: #eef3f8; color: #183c60; font-weight: 700; }
th, td { padding: 11px 12px; border-bottom: 1px solid #e1e7ee; vertical-align: top; }
td { min-width: 80px; } ul { padding-left: 22px; } li { margin: 8px 0; }
@media (max-width: 720px) { main { margin: 0; padding: 28px 20px; border: 0; } h1 { font-size: 27px; } }
@media print { body { background: white; font-size: 10pt; } main { max-width: none; margin: 0; padding: 0; border: 0; } h2 { break-after: avoid; } table { font-size: 8pt; } tr, img { break-inside: avoid; } a { color: inherit; } .table-wrap { overflow: visible; } }
'''
    page = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>선행논문 공개 코드 재현 | ' + MODEL + '</title><style>' + css + '</style></head><body><main><div class="eyebrow">RESEARCH REPRODUCTION · ' + MODEL + '</div>' + '\n'.join(body) + '</main></body></html>'
    (ROOT / 'reports/REPORT_KO.html').write_text(page, encoding='utf-8')


def main():
    metrics_path = ROOT / 'results/metrics.csv'
    if not metrics_path.exists():
        raise SystemExit('No completed results/metrics.csv. Finish computation before reporting.')
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_path)
    if len(metrics) != 4 or set(zip(metrics.scope, metrics.method)) != {(s, m) for s in SCOPES for m in METHODS}:
        raise ValueError('Expected exactly four scope/method rows.')
    evidence = notebook_evidence()
    comparisons = comparison_rows(metrics, evidence)
    make_chart(metrics)
    make_html(make_readme(metrics, evidence, comparisons))
    print('Created README, HTML report, cumulative-return PNG, reference comparison CSV, and notebook evidence JSON.')


if __name__ == '__main__':
    main()
