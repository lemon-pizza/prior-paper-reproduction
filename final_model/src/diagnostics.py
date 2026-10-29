"""Rebuild the retained model's price/missingness diagnostics from local inputs.

No historical report aggregate is an input. Long missingness means at least 60
consecutive rows of the preserved trading-date index, clipped to the evaluation
period. First/last observation categories use the complete supplied price file;
they do not identify IPO, delisting, trading suspension or economic value.
"""
from pathlib import Path
import os
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
SOURCE = ROOT / 'data/prices.parquet.gz'
START = '2017-01-03'
END = '2023-05-31'
SOURCE_SHA256 = '7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'


def generate_missing_reports(prices):
    """Calculate all three missing-data outputs afresh; return plot inputs."""
    valid = np.isfinite(prices) & (prices > 0)
    evaluation = prices.loc[START:END]
    observed = valid.loc[evaluation.index]
    days, columns = evaluation.shape
    empty = valid.columns[~valid.any()].tolist()
    ever = [name for name in prices.columns if name not in empty]
    categories = pd.DataFrame('', index=evaluation.index, columns=evaluation.columns)
    metadata = {}
    for ticker in prices.columns:
        index = prices.index[valid[ticker]]
        first = index[0] if len(index) else None
        last = index[-1] if len(index) else None
        metadata[ticker] = {
            'source_first_observed': None if first is None else str(first.date()),
            'source_last_observed': None if last is None else str(last.date()),
            'observed_days_evaluation': int(observed[ticker].sum()),
            'missing_days_evaluation': int((~observed[ticker]).sum()),
            'source_observed_days': len(index),
        }
        if first is None:
            categories[ticker] = 'never_observed_in_source'
        else:
            absent = ~observed[ticker]
            categories.loc[absent & (evaluation.index < first), ticker] = 'before_first_observation'
            categories.loc[absent & (evaluation.index > last), ticker] = 'after_last_observation'
            categories.loc[absent & (evaluation.index >= first) & (evaluation.index <= last), ticker] = 'between_observations'
    runs = []
    for ticker in prices.columns:
        missing = (~observed[ticker]).to_numpy()
        bounds = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)))
        for lo, hi in zip(bounds[0::2], bounds[1::2]):
            labels = categories[ticker].iloc[lo:hi].unique().tolist()
            assert len(labels) == 1
            runs.append({
                'ticker': ticker, 'start': str(evaluation.index[lo].date()),
                'end': str(evaluation.index[hi-1].date()), 'trading_days': int(hi-lo),
                'category': labels[0], 'is_long_60': bool(hi-lo >= 60),
                'all_missing_source': ticker in empty, **metadata[ticker],
            })
    frame = pd.DataFrame(runs).sort_values(['trading_days','ticker','start'], ascending=[False,True,True])
    frame.to_csv(OUT / 'missing_runs.csv', index=False)
    daily = pd.DataFrame(index=evaluation.index)
    daily.index.name = 'date'
    daily['all_columns'] = columns
    daily['ever_observed_columns'] = len(ever)
    daily['missing_all505'] = (~observed).sum(axis=1)
    daily['missing_ever_observed483'] = (~observed[ever]).sum(axis=1)
    labels = ['never_observed_in_source','before_first_observation','after_last_observation','between_observations']
    for label in labels:
        daily[label] = (categories == label).sum(axis=1)
    daily['missing_rate_all505'] = daily.missing_all505 / columns
    daily['missing_rate_ever_observed483'] = daily.missing_ever_observed483 / len(ever)
    assert (daily[labels].sum(axis=1) == daily.missing_all505).all()
    daily.to_csv(OUT / 'daily_missing_counts.csv')
    long = frame[frame.is_long_60]
    partial = long[~long.all_missing_source]
    frozen = {}
    union = set()
    for style in ['MQ','MZ','BQ','BZ']:
        ledger = pd.read_parquet(ROOT / 'results/frozen_ledger' / f'paper_daily_{style}.parquet')
        assets = sorted(ledger.asset.unique())
        union.update(assets)
        ledger['abs_weight'] = ledger.weight.abs()
        terminal = ledger[ledger.date == ledger.date.max()]
        frozen[style] = {
            'assets': assets, 'asset_count': len(assets),
            'first_frozen_date': str(ledger.date.min().date()) if len(ledger) else None,
            'last_frozen_date': str(ledger.date.max().date()) if len(ledger) else None,
            'asset_days': len(ledger), 'terminal_abs_weight': float(terminal.abs_weight.sum()),
            'per_asset': [],
        }
        for ticker, group in ledger.groupby('asset'):
            frozen[style]['per_asset'].append({
                'ticker': ticker, 'frozen_start': str(group.date.min().date()),
                'frozen_end': str(group.date.max().date()), 'frozen_days': len(group),
                'max_abs_weight': float(group.abs_weight.max()),
                'terminal_abs_weight': float(group.abs_weight.iloc[-1]),
                'underlying_missing_run': frame[(frame.ticker == ticker) & frame.is_long_60].to_dict('records'),
            })
    previous, seam = pd.Timestamp('2017-12-29'), pd.Timestamp('2018-01-02')
    newly_missing = prices.columns[valid.loc[previous] & ~valid.loc[seam]].tolist()
    newly_observed = prices.columns[~valid.loc[previous] & valid.loc[seam]].tolist()
    category_counts = {label: int((categories == label).sum().sum()) for label in labels}
    summary = {
        'definition': {
            'evaluation_start': str(evaluation.index[0].date()),
            'evaluation_end': str(evaluation.index[-1].date()),
            'evaluation_trading_rows': days,
            'missing': 'non-finite or nonpositive source price; observed source has zero nonpositive prices',
            'long_missing': 'at least 60 consecutive rows of the supplied trading-date index, clipped to evaluation period; not 60 calendar days',
            'category_reference': 'first/last finite positive observation in full supplied source 2010-01-04 to 2023-06-20',
            'warning': 'missing values do not establish trading halts, delisting, IPO dates, bankruptcy or cash recovery',
        },
        'source_path': str(SOURCE.resolve()), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'all_columns': columns, 'ever_observed_columns': len(ever),
        'all_missing_source_columns': len(empty), 'all_missing_source_tickers': empty,
        'all_missing_evaluation_columns': int((~observed.any()).sum()),
        'denominators': {'all505_stock_days': days*columns, 'ever_observed483_stock_days': days*len(ever), 'all_missing22_stock_days': days*len(empty)},
        'missing_all505_stock_days': int((~observed).sum().sum()),
        'missing_all505_rate': float((~observed).sum().sum()/(days*columns)),
        'missing_ever_observed483_stock_days': int((~observed[ever]).sum().sum()),
        'missing_ever_observed483_rate': float((~observed[ever]).sum().sum()/(days*len(ever))),
        'category_stock_days': category_counts,
        'category_share_of_all505_stock_days': {k:v/(days*columns) for k,v in category_counts.items()},
        'any_missing_ticker_count': int((~observed).any().sum()), 'missing_runs_total': len(frame),
        'long_missing_all': {'ticker_count': int(long.ticker.nunique()), 'run_count': len(long), 'stock_days': int(long.trading_days.sum())},
        'long_missing_excluding_allmissing': {'ticker_count': int(partial.ticker.nunique()), 'run_count': len(partial), 'stock_days': int(partial.trading_days.sum())},
        'top15_long_runs_all': long.head(15).to_dict('records'),
        'top15_long_runs_excluding_allmissing': partial.head(15).to_dict('records'),
        'long_runs_by_category': {k:{'runs':len(g),'tickers':int(g.ticker.nunique()),'stock_days':int(g.trading_days.sum())} for k,g in long.groupby('category')},
        'common_2018_break': {
            'previous_date': str(previous.date()), 'missing_start_date': str(seam.date()),
            'new_missing_count': len(newly_missing), 'new_missing_tickers': newly_missing,
            'new_observed_count': len(newly_observed), 'new_observed_tickers': newly_observed,
            'missing_before': int(daily.loc[previous,'missing_all505']),
            'missing_after': int(daily.loc[seam,'missing_all505']),
            'runs': frame[(frame.ticker.isin(newly_missing)) & (frame.start == '2018-01-02')].to_dict('records'),
        },
        'main_strategy_frozen_ledgers': frozen,
        'main_frozen_union_tickers': sorted(union), 'main_frozen_union_count': len(union),
        'main_frozen_in_2018_break': sorted(union & set(newly_missing)),
        'daily_missing_range': {'min':int(daily.missing_all505.min()),'max':int(daily.missing_all505.max()),'first':int(daily.missing_all505.iloc[0]),'last':int(daily.missing_all505.iloc[-1])},
    }
    assert sum(category_counts.values()) == summary['missing_all505_stock_days']
    (OUT / 'missing_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return daily, summary


def render_diagnostics():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/AppleGothic.ttf')
    plt.rcParams.update({'font.family':'AppleGothic','font.size':15,'axes.titlesize':17,'axes.labelsize':14,
                        'xtick.labelsize':13,'ytick.labelsize':13,'axes.unicode_minus':False,
                        'figure.facecolor':'white','axes.facecolor':'white','text.color':'#163448',
                        'axes.labelcolor':'#163448','xtick.color':'#425968','ytick.color':'#425968',
                        'savefig.facecolor':'white','svg.fonttype':'path'})
    RAW='#c65c38'; REF='#087f8c'; INK='#163448'; GRID='#dbe4e9'
    prices=pd.read_parquet(SOURCE)
    ev=prices.loc['2017-01-03':'2023-05-31']
    disc=pd.read_csv(ROOT/'audit/price_discontinuities_independent.csv').set_index('ticker')
    seam=pd.Timestamp('2018-01-02'); anchor=pd.Timestamp('2017-12-29')
    references={};comparison=[]
    for ticker in ['SRE','CPRT','LH','BWA']:
        fp=ROOT/'data/price_reference'/f'{ticker}.json'
        assert hashlib.sha256(fp.read_bytes()).hexdigest()==disc.loc[ticker,'response_sha256']
        j=json.loads(fp.read_text())['chart']['result'][0]
        ix=pd.to_datetime(j['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        references[ticker]=pd.Series(j['indicators']['adjclose'][0]['adjclose'],index=ix)
        r=prices.loc[seam,ticker]/prices.loc[anchor,ticker]-1
        ref=references[ticker].loc[seam]/references[ticker].loc[anchor]-1
        assert np.isclose(r,disc.loc[ticker,'source_return']) and np.isclose(ref,disc.loc[ticker,'response_return'])
        comparison.append({'ticker':ticker,'raw_return_pct':100*r,'reference_return_pct':100*ref})
    pd.DataFrame(comparison).to_csv(OUT/'price_return_comparison.csv',index=False)

    def clean(ax):
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(GRID)
        ax.grid(axis='y',color=GRID,linewidth=.7)
        ax.set_axisbelow(True)

    fig=plt.figure(figsize=(12.4,9.2))
    gs=fig.add_gridspec(2,2,left=.09,right=.96,bottom=.12,top=.84,hspace=.63,height_ratios=[1,1.08],wspace=.24)
    fig.text(.06,.957,'가격 불연속: 2018년 첫 거래일에 원자료만 크게 점프',size=22,weight='bold')
    fig.text(.06,.91,'주황 = 원자료    ·    청록 점선 = 별도로 조회해 보존한 조정가격',size=15)
    for k,ticker in enumerate(['SRE','CPRT']):
        ax=fig.add_subplot(gs[0,k]);clean(ax)
        raw=prices.loc['2017-12-18':'2018-01-12',ticker]/prices.loc[anchor,ticker]*100
        ref=references[ticker].loc[raw.index]/references[ticker].loc[anchor]*100
        raw.plot(ax=ax,color=RAW,lw=2.6,marker='o',markersize=3)
        ref.plot(ax=ax,color=REF,lw=2.4,ls='--',marker='o',markersize=3)
        ax.axvline(seam,color='#81949e',lw=1,ls=':')
        ax.set_ylim(80,270);ax.set_title(ticker,loc='left',pad=12,weight='bold')
        ax.set_ylabel('가격지수 (12/29 = 100)' if k==0 else '')
        ax.set_xlabel('2017년 12월 → 2018년 1월')
        ticks=pd.to_datetime(['2017-12-18','2017-12-29','2018-01-02','2018-01-12'])
        ax.set_xticks(ticks);ax.set_xticklabels(['12/18','12/29','1/2','1/12'])
        val=float(raw.loc[seam]);refval=float(ref.loc[seam])
        ax.annotate(f'하루 +{disc.loc[ticker,"source_return"]*100:.2f}%',(seam,val),xytext=(8,14),textcoords='offset points',color=RAW,size=14,weight='bold')
        ax.annotate(f'별도 조회 {disc.loc[ticker,"response_return"]*100:+.2f}%',(seam,refval),xytext=(8,-24),textcoords='offset points',color=REF,size=13)
    ax=fig.add_subplot(gs[1,:]);clean(ax);ax.grid(False);ax.grid(axis='x',color=GRID,lw=.7)
    y=np.arange(4);vals=np.array([r['raw_return_pct'] for r in comparison]);refs=np.array([r['reference_return_pct'] for r in comparison])
    ax.barh(y-.18,vals,height=.29,color=RAW,label='원자료');ax.barh(y+.18,refs,height=.29,color=REF,label='별도 조회')
    ax.set_yticks(y);ax.set_yticklabels([r['ticker'] for r in comparison]);ax.invert_yaxis()
    ax.axvline(0,color='#81949e',lw=1);ax.set_xlim(-12,163);ax.set_xticks([0,50,100,150])
    ax.set_xlabel('2018-01-02 하루 가격 변화율 (%)')
    ax.set_title('같은 날짜에 확인된 4종목의 차이',loc='left',pad=14,weight='bold')
    for yy,v,r in zip(y,vals,refs):
        ax.text(v+2,yy-.18,f'{v:+.2f}%',va='center',size=14,color=RAW)
        ax.text(max(0,r)+2,yy+.18,f'{r:+.2f}%',va='center',size=13,color=REF)
    fig.text(.06,.047,'별도 조회값: 앞선 검토 때 저장한 Yahoo 조정가격. 두 자료의 가격 수준은 12/29=100으로 맞춰 비교했습니다.',size=12,color='#526978')
    fig.text(.06,.021,'이는 연결 불일치의 증거이며, 별도 조회자료 전체가 완전한 정답으로 검증되었다는 뜻은 아닙니다.',size=12,color='#526978')
    fig.savefig(OUT/'price_discontinuity.png',dpi=180);fig.savefig(OUT/'price_discontinuity.svg');plt.close(fig)

    counts,summary=generate_missing_reports(prices)
    frozen={}
    for st in ['MQ','BQ','BZ','MZ']:
        d=pd.read_csv(ROOT/'results/daily'/f'paper_daily_{st}.csv',parse_dates=['date']).set_index('date')
        frozen[st]=d.frozen_gross*100
        expected=summary['main_strategy_frozen_ledgers'][st]['terminal_abs_weight']*100
        assert np.isclose(frozen[st].iloc[-1],expected)
    pd.DataFrame(frozen).to_csv(OUT/'frozen_exposure_pct.csv')
    fig,axs=plt.subplots(2,1,figsize=(12.4,9.4),gridspec_kw={'left':.09,'right':.96,'bottom':.12,'top':.82,'hspace':.56})
    fig.text(.06,.957,'장기 결측: 21종목이 같은 날 끊겨 5년 넘게 지속',size=22,weight='bold')
    fig.text(.06,.91,'2018-01-02 ~ 2023-05-31, 종목당 1,362거래일 연속 결측',size=16)
    fig.text(.06,.871,'전체 505열 중 22열은 전 기간 관측 없음. 아래는 나머지 483종목 기준입니다.',size=13,color='#526978')
    ax=axs[0];clean(ax)
    count=counts['missing_ever_observed483']
    ax.fill_between(count.index,count.to_numpy(),color=RAW,alpha=.12,step='post')
    ax.step(count.index,count.to_numpy(),where='post',color=RAW,lw=2.2)
    ax.set_ylim(0,39);ax.set_yticks([0,10,20,30]);ax.set_ylabel('당일 가격이 없는 종목 수')
    ax.set_title('종목×거래일 기준 결측률: 전체 8.93% / 전부 결측 22열 제외 4.78%',loc='left',pad=12,size=16)
    ax.annotate('11 → 31종목\n21개 단절 · 1개 새 관측',(seam,31),xytext=(pd.Timestamp('2018-06-01'),34),arrowprops={'arrowstyle':'-','color':'#81949e'},size=13)
    ax.annotate('2023-05-31: 22종목',(count.index[-1],count.iloc[-1]),xytext=(-160,-30),textcoords='offset points',size=13)
    ax.set_xlim(pd.Timestamp('2017-01-01'),pd.Timestamp('2023-12-01'));ax.set_xticks(pd.date_range('2017-01-01','2023-01-01',freq='YS'));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('연도')
    ax=axs[1];clean(ax)
    colors={'MQ':REF,'BQ':'#4d79ad','BZ':RAW,'MZ':'#6f7c86'}
    for st in ['BZ','MQ','BQ','MZ']:
        s=frozen[st];ax.plot(s.index,s,color=colors[st],lw=2.1,ls='--' if st=='MZ' else '-')
        ax.text(pd.Timestamp('2023-07-10'),float(s.iloc[-1]),f'{st} {s.iloc[-1]:.2f}%',color=colors[st],va='center',size=14,weight='bold')
    ax.set_title('전략 장부에 남은 동결 보유액의 비중',loc='left',pad=12,size=17,weight='bold')
    ax.set_ylabel('전일 순자산 대비 절대금액 (%)')
    ax.set_xlabel('연도');ax.set_ylim(-.3,7.3);ax.set_yticks([0,2,4,6])
    ax.set_xlim(pd.Timestamp('2017-01-01'),pd.Timestamp('2024-04-01'));ax.set_xticks(pd.date_range('2017-01-01','2023-01-01',freq='YS'));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.text(pd.Timestamp('2019-01-01'),.55,'이 보유분은 마지막 가격으로 평가되며, 이후 가격 변화 손익은 0으로 기록',size=12,color='#526978')
    fig.text(.06,.049,'M=매칭, B=상위 250페어 / Q·Z=신호 방식. 동결 비중은 손실률이 아니라 가격을 확인하지 못한 잔여 보유 규모입니다.',size=12,color='#526978')
    fig.text(.06,.022,'결측은 자료에 가격이 없다는 뜻입니다. 실제 거래정지·상장폐지 또는 경제적 가치 0을 뜻하지 않습니다.',size=12,color='#526978')
    fig.savefig(OUT/'missing_and_frozen.png',dpi=180);fig.savefig(OUT/'missing_and_frozen.svg');plt.close(fig)
    print('Diagnostics regenerated:',OUT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256, 'Preserved price input changed'
    previous = os.environ.get('MPLCONFIGDIR')
    try:
        with tempfile.TemporaryDirectory(prefix='quant-diagnostics-mpl-') as cache:
            os.environ['MPLCONFIGDIR'] = cache
            render_diagnostics()
    finally:
        if previous is None:
            os.environ.pop('MPLCONFIGDIR', None)
        else:
            os.environ['MPLCONFIGDIR'] = previous


if __name__ == '__main__':
    main()
