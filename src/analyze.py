"""Independent descriptive statistics and paired date-block uncertainty.

Consumes completed daily cash/NAV simulations; never generates selections or
trades. No production performance/statistics functions are imported. The report
and every number in it are generated from the input CSVs.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import html
import json
import os
from pathlib import Path
import re
import platform
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'matching-baseline-matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
SEED=240307998
STYLES=('M','B','U')
FEES=(0,5,10)
PATTERN=re.compile(r'^(clean_)?(stockgross|pairgross)_d([01])_([MBU])_c(0|5|10)$')
CHECKS=[]


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean_json(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.bool_,)):return bool(value)
    return value


def check(name,actual,expected,tolerance=2e-10):
    a,b=np.broadcast_arrays(np.asarray(actual,float),np.asarray(expected,float))
    if not np.array_equal(np.isnan(a),np.isnan(b)):
        raise AssertionError((name,'NaN masks differ'))
    if np.isinf(a).any() or np.isinf(b).any():raise AssertionError((name,'Unexpected infinity'))
    finite=np.isfinite(a)
    error=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
    CHECKS.append(dict(check=name,max_absolute_error=error,tolerance=tolerance,passed=bool(error<=tolerance)))
    if error>tolerance:raise AssertionError((name,error,tolerance))


def metrics(r):
    a=np.asarray(r,float)
    if not np.isfinite(a).all():raise ValueError('Returns must be finite, including cash days')
    n=len(a)
    if n<2:raise ValueError('A statistics scope needs at least two observations')
    mean=float(a.sum()/n)
    variance=float(((a-mean)**2).sum()/(n-1))
    sd=variance**.5
    wealth=np.r_[1.,np.cumprod(1+a)]
    if (wealth<=0).any():raise ValueError('Nonpositive wealth: ordinary performance metrics invalid')
    return dict(days=n,mean_daily=mean,mean_annual=mean*252,volatility=sd*np.sqrt(252),
                sharpe=mean/sd*np.sqrt(252) if sd else np.nan,
                cumulative=float(wealth[-1]-1),cagr=float(wealth[-1]**(252/n)-1),
                max_drawdown=float((wealth/np.maximum.accumulate(wealth)-1).min()))


def manual_metrics(r):
    """Independent one-pass mean/variance and quantity-free NAV/peak loop."""
    n=0;mean=0.;m2=0.;nav=1.;peak=1.;drawdown=0.
    for value in r:
        n+=1;delta=float(value)-mean;mean+=delta/n;m2+=delta*(float(value)-mean)
        nav*=1+float(value);peak=max(peak,nav);drawdown=min(drawdown,nav/peak-1)
    sd=(m2/(n-1))**.5
    return dict(days=n,mean_daily=mean,mean_annual=mean*252,volatility=sd*np.sqrt(252),
                sharpe=mean/sd*np.sqrt(252) if sd else np.nan,
                cumulative=nav-1,cagr=nav**(252/n)-1,max_drawdown=drawdown)


def load_inputs(folder,require_clean=False):
    expected={f'{budget}_d{delay}_{style}_c{fee}' for budget in ['stockgross','pairgross']
              for delay in [0,1] for style in STYLES for fee in FEES}
    clean_expected={f'clean_stockgross_d1_{style}_c{fee}' for style in STYLES for fee in FEES}
    present={p.stem for p in folder.glob('*.csv') if PATTERN.fullmatch(p.stem)}
    if require_clean or present&clean_expected:expected|=clean_expected
    missing=expected-present
    if missing:raise FileNotFoundError('Incomplete simulation outputs: '+', '.join(sorted(missing)))
    data={};meta={};sources=[];dates=None
    required=['returns','nav','turnover','gross_exposure','net_exposure','hhi','frozen_gross']
    for stem in sorted(expected):
        path=folder/f'{stem}.csv';match=PATTERN.fullmatch(stem)
        clean,budget,delay,style,fee=match.groups()
        frame=pd.read_csv(path,index_col='date',parse_dates=True)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f'Duplicated or unsorted dates: {stem}')
        if any(c not in frame for c in required):raise ValueError((stem,'Missing required columns'))
        if not np.isfinite(frame[required].to_numpy()).all():raise ValueError((stem,'Nonfinite values'))
        if dates is None:dates=frame.index
        if not dates.equals(frame.index):raise ValueError((stem,'Dates differ across paths'))
        rebuilt=np.cumprod(1+frame.returns.to_numpy())
        check('NAV_reconstruction:'+stem,rebuilt,frame.nav.to_numpy())
        implied=frame.nav.to_numpy()/np.r_[1.,frame.nav.to_numpy()[:-1]]-1
        check('returns_from_NAV:'+stem,frame.returns.to_numpy(),implied)
        if (frame.turnover < -1e-12).any():raise ValueError((stem,'Negative turnover'))
        data[stem]=frame
        meta[stem]=dict(path=stem,dataset='clean' if clean else 'original',budget=budget,
                        delay=int(delay),style=style,cost_bp=int(fee))
        sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(frame)))
    return data,meta,dates,sources


def scope_indices(dates):
    return dict(full=np.ones(len(dates),bool),
                years_2017_2019=dates.year<=2019,
                years_2020_on=dates.year>=2020,
                exclude_2018_01_02=dates!=pd.Timestamp('2018-01-02'))


def all_metrics(data,meta,dates):
    records=[]
    for stem,frame in data.items():
        for scope,mask in scope_indices(dates).items():
            subset=frame.loc[mask]
            result=metrics(subset.returns)
            manual=manual_metrics(subset.returns)
            check(f'manual_metrics:{stem}:{scope}',list(result.values()),list(manual.values()))
            record=dict(**meta[stem],scope=scope,start=str(subset.index.min().date()),
                        end=str(subset.index.max().date()),**result,
                        turnover_daily=float(subset.turnover.mean()),
                        turnover_annual=float(subset.turnover.mean()*252),
                        gross_exposure_mean=float(subset.gross_exposure.mean()),
                        net_exposure_mean=float(subset.net_exposure.mean()),
                        hhi_mean=float(subset.hhi.mean()),frozen_gross_max=float(subset.frozen_gross.max()),
                        frozen_days=int((subset.frozen_gross.abs()>1e-12).sum()),
                        zero_gross_days=int((subset.gross_exposure.abs()<1e-12).sum()))
            if scope=='exclude_2018_01_02':
                record['diagnostic']='Remove one return observation only; retained trades/NAV paths were not rerun.'
            elif scope!='full':
                record['diagnostic']='Slice the original continuous path; carried holdings and capital state remain.'
            else:record['diagnostic']='Full continuous independently fee-funded path.'
            records.append(record)
    check('initial_NAV_drawdown_guard',metrics([-.1,.01])['max_drawdown'],-.1)
    return pd.DataFrame(records)


def compare_metrics(perf):
    rows=[]
    group_cols=['dataset','budget','delay','cost_bp','scope']
    for group,part in perf.groupby(group_cols,sort=True):
        by=part.set_index('style')
        for reference in ['B','U']:
            a,b=by.loc['M'],by.loc[reference]
            rows.append(dict(**dict(zip(group_cols,group)),comparison='M-'+reference,
                sharpe_gap=a.sharpe-b.sharpe,mean_annual_gap=a.mean_annual-b.mean_annual,
                volatility_ratio=a.volatility/b.volatility,
                cumulative_gap=a.cumulative-b.cumulative,
                max_drawdown_gap=a.max_drawdown-b.max_drawdown,
                turnover_gap=a.turnover_annual-b.turnover_annual,
                M_gross_exposure=a.gross_exposure_mean,reference_gross_exposure=b.gross_exposure_mean))
    return pd.DataFrame(rows)


def bootstrap_primary(data,meta):
    keys=[stem for stem,m in meta.items() if m['budget']=='stockgross' and m['delay']==1]
    keys=sorted(keys)
    values=np.column_stack([data[k].returns.to_numpy() for k in keys]);n=len(values)
    key_index={k:i for i,k in enumerate(keys)}
    datasets=sorted({meta[k]['dataset'] for k in keys})
    output=[]
    for length,reps in [(21,4000),(5,2000),(63,2000)]:
        # Clean is a declared additional robustness test, requiring L21 only.
        rng=np.random.default_rng(SEED+length)
        starts=rng.integers(0,n,size=(reps,(n+length-1)//length))
        means=[];sharpes=[];vols=[]
        for begin in range(0,reps,64):
            ix=((starts[begin:begin+64,:,None]+np.arange(length))%n).reshape(-1,starts.shape[1]*length)[:,:n]
            draws=values[ix]
            mean=draws.mean(axis=1)
            sd=np.sqrt(((draws-mean[:,None,:])**2).sum(axis=1)/(n-1))
            with np.errstate(divide='ignore',invalid='ignore'):
                sharpe=np.sqrt(252)*mean/sd
            means.append(mean*252);sharpes.append(sharpe);vols.append(sd*np.sqrt(252))
            if begin==0:
                for j in range(len(keys)):
                    direct=manual_metrics(draws[0,:,j])
                    check(f'bootstrap_first_draw:L{length}:{keys[j]}',
                          [mean[0,j]*252,sd[0,j]*np.sqrt(252),sharpe[0,j]],
                          [direct['mean_annual'],direct['volatility'],direct['sharpe']])
        boot=dict(mean_annual=np.concatenate(means),sharpe=np.concatenate(sharpes),volatility=np.concatenate(vols))
        for dataset in datasets:
            if dataset=='clean' and length!=21:continue
            prefix='clean_' if dataset=='clean' else ''
            for fee in FEES:
                mkey=f'{prefix}stockgross_d1_M_c{fee}'
                for reference in ['B','U']:
                    rkey=f'{prefix}stockgross_d1_{reference}_c{fee}'
                    a,b=key_index[mkey],key_index[rkey]
                    ma,mb=metrics(data[mkey].returns),metrics(data[rkey].returns)
                    distributions=dict(sharpe_gap=boot['sharpe'][:,a]-boot['sharpe'][:,b],
                        mean_annual_gap=boot['mean_annual'][:,a]-boot['mean_annual'][:,b],
                        volatility_ratio=boot['volatility'][:,a]/boot['volatility'][:,b])
                    estimates=dict(sharpe_gap=ma['sharpe']-mb['sharpe'],
                        mean_annual_gap=ma['mean_annual']-mb['mean_annual'],
                        volatility_ratio=ma['volatility']/mb['volatility'])
                    for metric,dist in distributions.items():
                        finite=np.isfinite(dist)
                        if not finite.all():raise ValueError((dataset,fee,reference,length,metric,'Nonfinite bootstrap draws'))
                        low,high=np.quantile(dist,[.025,.975]);null=1. if metric.endswith('ratio') else 0.
                        output.append(dict(dataset=dataset,budget='stockgross',delay=1,cost_bp=fee,
                            comparison='M-'+reference,metric=metric,estimate=estimates[metric],
                            ci_low=float(low),ci_high=float(high),block_length=length,reps=reps,
                            seed=SEED+length,fraction_above_null=float(np.mean(dist>null)),
                            null=null,finite_draws=int(finite.sum())))
        print(f'Completed joint date bootstrap L={length}, reps={reps}',flush=True)
    return pd.DataFrame(output)


def annual_comparisons(data,meta,dates):
    rows=[]
    datasets=sorted({m['dataset'] for m in meta.values()})
    for dataset in datasets:
        prefix='clean_' if dataset=='clean' else ''
        for year in sorted(set(dates.year)):
            mask=dates.year==year
            for fee in FEES:
                a=data[f'{prefix}stockgross_d1_M_c{fee}'].loc[mask]
                b=data[f'{prefix}stockgross_d1_B_c{fee}'].loc[mask]
                ma,mb=metrics(a.returns),metrics(b.returns)
                rows.append(dict(dataset=dataset,year=year,cost_bp=fee,start=str(a.index.min().date()),
                    end=str(a.index.max().date()),days=len(a),M_return=ma['cumulative'],B_return=mb['cumulative'],
                    return_gap=ma['cumulative']-mb['cumulative'],M_sharpe=ma['sharpe'],B_sharpe=mb['sharpe'],
                    sharpe_gap=ma['sharpe']-mb['sharpe'],volatility_ratio=ma['volatility']/mb['volatility'],
                    M_drawdown=ma['max_drawdown'],B_drawdown=mb['max_drawdown']))
    return pd.DataFrame(rows)


def plot_results(data,meta,boot,folder):
    folder.mkdir(parents=True,exist_ok=True)
    colors=dict(M='#087f8c',B='#d87530',U='#775ca6')
    datasets=sorted({m['dataset'] for m in meta.values()},key=lambda x:x!='original')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    installed_fonts={font.name for font in font_manager.fontManager.ttflist}
    korean_font=next((name for name in ['AppleGothic','Malgun Gothic','Noto Sans CJK KR','NanumGothic']
                      if name in installed_fonts),None)
    with plt.rc_context({'font.family':korean_font or 'DejaVu Sans','font.size':12,'axes.unicode_minus':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,4.9),sharey=True)
        labels=(dict(M='가중 매칭 (M)',B='같은 수 상위 페어 (B)',U='무가중 매칭 (U)') if korean_font
                else dict(M='Weighted matching (M)',B='Same-count top pairs (B)',U='Unweighted matching (U)'))
        titles=(['거래비용 반영 전','5bp 거래비용 반영 후'] if korean_font
                else ['Before transaction costs','After 5 bp transaction costs'])
        low,high=100.,100.
        for ax,fee,title in zip(axes,[0,5],titles):
            for style in STYLES:
                frame=data[f'stockgross_d1_{style}_c{fee}']
                wealth=frame.nav*100
                low=min(low,float(wealth.min()));high=max(high,float(wealth.max()))
                ax.plot(frame.index,wealth,label=labels[style],color=colors[style],linewidth=1.8)
            ax.axhline(100,color='#88939d',lw=.9,ls='--')
            ax.set_title(title,pad=14,fontsize=14);ax.grid(alpha=.17)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        padding=(high-low)*.07
        axes[0].set_ylim(low-padding,high+padding)
        axes[0].set_ylabel('초기 100원으로 시작한 자산' if korean_font else 'Account value (initial = 100)',labelpad=12)
        handles,legend_labels=axes[0].get_legend_handles_labels()
        fig.suptitle('같은 페어 수와 총노출로 비교한 결과' if korean_font else 'Equal pair count and gross exposure',fontsize=18,y=.98)
        fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
        footer=('2017.01–2023.05 · 하루 추가 지연 · 5bp = 거래대금의 0.05%' if korean_font
                else '2017.01–2023.05 · One extra session of delay · 5 bp = 0.05% of traded value')
        fig.text(.5,.025,footer,ha='center',fontsize=11,color='#526574')
        fig.subplots_adjust(left=.09,right=.985,top=.73,bottom=.15,wspace=.13)
        for ext in ['png','svg']:fig.savefig(folder/f'overview.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(len(datasets),3,figsize=(13,4.2*len(datasets)),squeeze=False,constrained_layout=True)
    for row,dataset in enumerate(datasets):
        prefix='clean_' if dataset=='clean' else ''
        for col,fee in enumerate(FEES):
            ax=axes[row,col]
            for style in STYLES:
                frame=data[f'{prefix}stockgross_d1_{style}_c{fee}']
                ax.plot(frame.index,frame.nav,label=style,color=colors[style],linewidth=1.4)
            ax.axhline(1,color='#8c959e',lw=.8,ls='--')
            ax.set_title(f'{"Original universe" if dataset=="original" else "SRE / CPRT excluded"} | cost {fee} bp')
            ax.set_ylabel('Account NAV (initial = 1)');ax.grid(alpha=.15)
            ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.legend(ncol=3,frameon=False)
    fig.suptitle('Primary paths: stock net gross budget 1, one additional session of delay',fontsize=12)
    for ext in ['png','svg']:fig.savefig(folder/f'primary_nav.{ext}',dpi=170)
    plt.close(fig)
    main=boot[boot.block_length==21]
    display=main[main.metric=='sharpe_gap'].sort_values(['dataset','cost_bp','comparison']).reset_index(drop=True)
    height=max(4.5,.42*len(display)+1.7)
    fig,axes=plt.subplots(1,2,figsize=(12,height),constrained_layout=True)
    for ax,metric,null,title in zip(axes,['sharpe_gap','volatility_ratio'],[0,1],['Sharpe difference','Volatility ratio (M / reference)']):
        for i,r in display.iterrows():
            rr=main[(main.dataset==r.dataset)&(main.cost_bp==r.cost_bp)&(main.comparison==r.comparison)&(main.metric==metric)].iloc[0]
            color='#087f8c' if r.dataset=='original' else '#775ca6'
            # A percentile interval need not contain its point estimate.
            ax.hlines(i,rr.ci_low,rr.ci_high,color=color,linewidth=1.5)
            ax.vlines([rr.ci_low,rr.ci_high],i-.09,i+.09,color=color,linewidth=1)
            ax.plot(rr.estimate,i,'o',color=color)
        labels=[f'{"Full" if r.dataset=="original" else "Clean"} | {r.cost_bp} bp | {r.comparison}' for r in display.itertuples()]
        ax.set_yticks(range(len(display)),labels);ax.invert_yaxis();ax.axvline(null,color='#6f7882',ls='--',lw=1)
        ax.set_title(title);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Conditional 95% paired circular-block intervals: 21 sessions, 4,000 draws',fontsize=12)
    for ext in ['png','svg']:fig.savefig(folder/f'primary_intervals.{ext}',dpi=170)
    plt.close(fig)


def fmt(value,kind='number'):
    if pd.isna(value):return '—'
    if kind=='percent':return f'{100*value:,.2f}%'
    if kind=='pp':return f'{100*value:+,.2f}%p'
    if kind=='signed':return f'{value:+.3f}'
    if kind=='integer':return f'{int(value):,}'
    return f'{value:.3f}'


def table(frame,columns):
    header=''.join(f'<th>{html.escape(label)}</th>' for key,label,kind in columns)
    body=[]
    for _,row in frame.iterrows():
        values=[]
        for key,label,kind in columns:
            value=str(row[key]) if kind=='text' else fmt(row[key],kind)
            values.append(f'<td>{html.escape(value)}</td>')
        body.append('<tr>'+''.join(values)+'</tr>')
    return '<div class="table-wrap"><table><thead><tr>'+header+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'


def interval_sentence(boot,fee,comparison):
    selected=boot[(boot.dataset=='original')&(boot.cost_bp==fee)&(boot.comparison==comparison)&(boot.metric=='sharpe_gap')&(boot.block_length==21)].iloc[0]
    if selected.ci_low>0:reading='조건부 구간 전체가 0보다 큽니다'
    elif selected.ci_high<0:reading='조건부 구간 전체가 0보다 작습니다'
    else:reading='조건부 구간이 0을 포함합니다'
    return f'{fee}bp에서 {comparison} 샤프 차이는 {selected.estimate:+.3f}, 95% 구간은 [{selected.ci_low:.3f}, {selected.ci_high:.3f}]로 {reading}.'


def render_report(perf,gaps,boot,annual,meta,dates,out):
    actual_audit=json.loads((out/'checks/actual_path_audit.json').read_text())
    if not actual_audit.get('passed'):raise ValueError('Independent actual-path audit has not passed')
    audited={row['scenario']:row for row in actual_audit['primary_path_checks']}
    terminal={style:audited[f'stockgross_d1_{style}_c5'] for style in STYLES}
    if not all(row['passed'] for row in terminal.values()):raise ValueError('Primary terminal audit has not passed')
    residual_text=', '.join(f'{style} {100*terminal[style]["final_unquoted_gross_fraction_of_final_NAV"]:.4f}%' for style in STYLES)
    primary=perf[(perf.scope=='full')&(perf.budget=='stockgross')&(perf.delay==1)]
    original=primary[primary.dataset=='original'].sort_values(['cost_bp','style'])
    main_ci=boot[(boot.dataset=='original')&(boot.block_length==21)].copy()
    ci_rows=[]
    for (fee,comparison),part in main_ci.groupby(['cost_bp','comparison']):
        row=dict(cost_bp=fee,comparison=comparison)
        for metric in ['sharpe_gap','mean_annual_gap','volatility_ratio']:
            r=part[part.metric==metric].iloc[0]
            scale=100 if metric=='mean_annual_gap' else 1
            row[metric]=f'{r.estimate*scale:+.3f} [{r.ci_low*scale:.3f}, {r.ci_high*scale:.3f}]'
        ci_rows.append(row)
    scope_names=dict(full='전체',years_2017_2019='2017–2019',years_2020_on='2020년 이후',exclude_2018_01_02='2018-01-02 수익만 제외')
    period=gaps[(gaps.dataset=='original')&(gaps.budget=='stockgross')&(gaps.delay==1)&(gaps.comparison=='M-B')].copy()
    period['scope_name']=period.scope.map(scope_names)
    sensitivities=gaps[(gaps.dataset=='original')&(gaps.scope=='full')].copy()
    yearly=annual[annual.dataset=='original'].copy()
    years=[f'{r.year} ({r.start[5:]}~{r.end[5:]})' for r in yearly.itertuples()]
    yearly['year_label']=years
    pieces=[]
    add=pieces.append
    add('<header><p class="eyebrow">기준 모델 · 사전 고정 비교</p><h1>매칭 전략 기준 모델 검증</h1>')
    add(f'<p class="lead">{dates.min():%Y-%m-%d}–{dates.max():%Y-%m-%d} · 고유 거래일 {len(dates):,}일 · {len(meta)}개 독립 현금·수량·순자산 경로</p></header>')
    headline=original[original.cost_bp==5].set_index('style')
    hm,hb=headline.loc['M'],headline.loc['B']
    add('<section><h2>주된 결과부터 보기</h2>')
    add(f'<p>한 거래일 추가 지연·종목 총노출 1·거래대금당 5bp 비용에서 <b>M의 누적수익은 {fmt(hm.cumulative,"percent")}, 샤프는 {fmt(hm.sharpe)}</b>입니다. 같은 개수의 상위 점수 페어를 고른 B는 누적수익 {fmt(hb.cumulative,"percent")}, 샤프 {fmt(hb.sharpe)}입니다.</p>')
    add('<div class="callout"><b>'+interval_sentence(boot,5,'M-B')+'</b></div>')
    add('<figure><img src="../results/figures/overview.svg" alt="초기 100원 기준 가중 매칭과 두 비교군의 비용 전후 자산 추이"><figcaption>원 유니버스의 주 분석입니다. 왼쪽은 비용 전, 오른쪽은 같은 거래 규칙에 5bp 비용을 실제로 지불한 경로입니다.</figcaption></figure>')
    add('<p class="caption"><b>최종 평가액은 전액 현금화된 금액이 아닙니다.</b> 주 분석의 5bp 경로에서 말일 가격이 없는 미청산 보유분의 절댓값 합은 최종 순자산 대비 '+residual_text+'입니다. 말일 가격이 관측된 보유분은 모두 청산했지만, 미관측 보유분은 정해진 수량 동결 규칙에 따라 마지막 관측가격으로 평가했습니다. 따라서 이 잔여분의 실제 청산 가능 가격과 최종 실현수익은 확인되지 않은 데이터 한계로 남습니다. 수치는 <a href="../checks/actual_path_audit.json">독립 말일 장부 검산</a>에서 가져왔습니다.</p>')
    add('<p class="caption">이 판단은 같은 역사 가격과 고정 실험 규칙에 조건부이며, 순수한 중복 제거의 인과효과나 새로운 표본에서의 실전 수익성을 입증하지 않습니다. 아래에 비용·기간·실행과 가격 품질 민감도를 모두 함께 표시합니다.</p></section>')
    add('<section><h2>이번 검증이 답하는 질문</h2><p>같은 가격, 후보, 신호와 자금 조건에서 <b>선정 정책 전체</b>를 바꾸었을 때 성과가 달라지는지 살펴봅니다. M은 공적분 후보 점수의 합을 최대화하는 비중복 매칭, B는 M과 같은 개수 K의 상위 점수 페어, U는 점수를 쓰지 않는 하나의 고정된 비중복 조합입니다. 이 비교는 순수한 중복 제거의 인과효과나 탐욕적 중복 제거 대비 전역 최적화만의 효과를 분리하지 않습니다.</p>')
    add('<div class="callout"><b>주된 통계 판단을 결과 전에 하나로 고정했습니다.</b> 종목 주문을 합산한 총노출 한도 1, 전체 주문을 한 거래일 더 늦춘 실행, 거래대금당 <b>5bp에서 M−B 샤프 차이의 21거래일 블록 95% 구간</b>입니다. 비용 0·10bp, 다른 실행·배분·기간·U 비교와 두 종목 제외는 보조 민감도입니다. 각 비용 경로는 비용을 실제 현금에서 지불하고 남은 자기 자산으로 다음 거래를 합니다. 가장 좋은 설정을 최종 결과로 고르지 않습니다.</div>')
    add('<p>원래 공개 코드에서 사용한 Q20 신호와 AIC로 시차를 고르는 ADF 검정은 유지했습니다. 방향은 양방향 회귀 중 더 음의 ADF 통계량을 택하는 공통 규칙으로 고정했습니다. 따라서 논문의 모든 명세나 표를 정확히 재현한 실험은 아닙니다.</p><p>신호와 회귀계수 β로 주식 수량 비율을 만든 뒤, 관측한 가격으로 달러 비중을 계산하는 공개 코드의 배분 방식을 사용했습니다. 주 분석은 이 전체 비중 지시를 다음 종가로 늦춰 집행하므로, 실제 체결 시 수량 비율은 달라질 수 있습니다. 월말 선정도 같은 지연 규칙을 적용하며 거래일은 한 번씩만 계산합니다.</p></section>')
    add('<section><h2>주 분석: 수익, 위험, 비용을 함께 보기</h2>')
    add(table(original,[('cost_bp','비용 bp','integer'),('style','전략','text'),('cumulative','누적수익','percent'),('cagr','CAGR','percent'),('volatility','연 변동성','percent'),('sharpe','샤프','number'),('max_drawdown','최대낙폭','percent')]))
    add('<p class="caption">샤프는 무위험수익률 0, 연 252일, 표본 표준편차를 사용합니다. 최대낙폭에는 최초 자산 1을 포함합니다. bp는 거래대금의 0.01%이며, 5bp는 0.05%입니다. 이는 논문의 연 1%÷252 비용 산식과 다른 비용 민감도입니다.</p>')
    add('<details><summary>모든 비용과 두 종목 제외 경로의 순자산 추이</summary><figure><img src="../results/figures/primary_nav.svg" alt="주 분석의 비용별 순자산 추이"><figcaption>실제 현금 장부에서 계산한 순자산입니다. 서로 다른 비용 경로의 보유 수량과 재투자 금액도 달라집니다.</figcaption></figure></details>')
    add('<h3>무엇을 관찰했는가</h3><p><b>주된 판단: '+interval_sentence(boot,5,'M-B')+'</b></p>')
    add('<ul>'+''.join('<li>비용 민감도: '+interval_sentence(boot,fee,'M-B')+'</li>' for fee in (0,10))+'</ul>')
    add('<p>낮은 변동성, 높은 샤프, 작은 최대낙폭은 서로 다른 결과입니다. 한 지표의 개선이 나머지 지표의 개선이나 실전 수익성을 자동으로 뜻하지 않습니다.</p></section>')
    add('<section><h2>성과 차이의 불확실성</h2>')
    add(table(pd.DataFrame(ci_rows),[('cost_bp','비용 bp','integer'),('comparison','비교','text'),('sharpe_gap','샤프 차이 [95% 구간]','text'),('mean_annual_gap','연 산술평균 차이 %p','text'),('volatility_ratio','변동성 비율 M/비교','text')]))
    add('<figure><img src="../results/figures/primary_intervals.svg" alt="샤프 차이와 변동성 비율 신뢰구간"><figcaption>샤프 차이의 기준은 0, 변동성 비율의 기준은 1입니다. 비율이 1보다 작으면 M의 변동성이 더 낮습니다.</figcaption></figure>')
    add('<p>전체 전략·비용에 같은 날짜 블록을 적용한 순환 이동블록 부트스트랩입니다. 주 설정은 21거래일·4,000회이며, 원 유니버스에서 5일·63일·각 2,000회도 확인합니다. 각 구간은 <b>다중 비교 조정을 하지 않은 명목 95% 구간</b>이며, 보조 비교의 양성 결과만 골라 일반화하지 않습니다. 이 구간은 이미 계산된 시계열과 고정 설정에 조건부입니다. 같은 역사 자료를 다시 사용하므로 <b>새로운 독립 검증 표본이나 미래 성과의 보장</b>이 아닙니다. 주 구간 전체가 양수여도 순수 인과효과가 입증되는 것은 아닙니다. 저장된 기준값 초과 재표집 비율은 사후확률도, 빈도주의 p값도 아닙니다.</p>')
    sensitivity_ci=boot[(boot.dataset=='original')&(boot.metric=='sharpe_gap')].copy()
    add('<details><summary>블록 길이별 샤프 차이 구간 전체</summary>'+table(sensitivity_ci,[('cost_bp','비용 bp','integer'),('comparison','비교','text'),('block_length','블록 일수','integer'),('estimate','차이','signed'),('ci_low','하한','signed'),('ci_high','상한','signed')])+'</details></section>')
    add('<section><h2>기간과 알려진 이상일의 영향</h2>')
    add(table(period,[('scope_name','범위','text'),('cost_bp','비용 bp','integer'),('sharpe_gap','M−B 샤프','signed'),('volatility_ratio','변동성 비율','number'),('cumulative_gap','누적수익 차이','pp')]))
    add('<p>기간별 수익은 전체 경로에서 잘라 보았습니다. 2020년에 포지션과 현금을 새로 시작한 실험이 아니므로 이전 보유분과 미청산 잔여분의 영향이 이어집니다. 부분 기간의 표시는 시작 자산을 1로 환산한 수익률입니다.</p>')
    add('<div class="callout warning"><b>2018-01-02 제외는 수익 기여도 진단입니다.</b> 해당 하루 수익 관측만 빼고 남은 수익을 연결했으며, 가격·이후 회귀창·선정·신호·보유 경로는 다시 계산하지 않았습니다. 따라서 가격 오류의 전체 영향을 제거한 재실행이나 실제 계좌 수익곡선으로 읽으면 안 됩니다.</div>')
    add('<h3>연도별 M−B: 주 분석 경로</h3>')
    add(table(yearly,[('year_label','연도·관측범위','text'),('cost_bp','비용 bp','integer'),('M_return','M 수익','percent'),('B_return','B 수익','percent'),('sharpe_gap','샤프 차이','signed'),('volatility_ratio','변동성 비율','number')]))
    add('<p class="caption">연도별 표는 사후 안정성 진단이며 별도 유의성 검정이 아닙니다. 마지막 연도는 표에 표시한 관측 기간만 포함합니다.</p></section>')
    if 'clean' in set(primary.dataset):
        clean=primary[primary.dataset=='clean'].sort_values(['cost_bp','style'])
        add('<section><h2>SRE·CPRT 제외 민감도</h2><p>알려진 가격 불연속이 있는 두 종목을 전체 기간 후보에서 제외하고, M·B·U를 다시 선정한 뒤 주 분석의 세 비용 경로를 다시 실행했습니다. 이는 <b>사후에 알려진 두 오류 종목을 격리한 민감도</b>이며 당시 이용 가능했던 전략이나 원가격을 완전히 정정한 결과가 아닙니다.</p>')
        add(table(clean,[('cost_bp','비용 bp','integer'),('style','전략','text'),('cumulative','누적수익','percent'),('volatility','연 변동성','percent'),('sharpe','샤프','number'),('max_drawdown','최대낙폭','percent')]))
        cb=boot[(boot.dataset=='clean')&(boot.metric=='sharpe_gap')]
        add(table(cb,[('cost_bp','비용 bp','integer'),('comparison','비교','text'),('estimate','샤프 차이','signed'),('ci_low','95% 하한','signed'),('ci_high','95% 상한','signed')]))
        add('</section>')
    add('<section><h2>실행·배분 가정을 바꾼 전체 민감도</h2><p><code>stockgross</code>는 같은 종목의 주문을 합친 뒤의 총노출, <code>pairgross</code>는 합치기 전 페어 다리의 절댓값 예산을 맞춥니다. 두 조건은 같지 않습니다. <code>delay0</code>은 관측한 종가에 즉시 체결한다고 보는 이상화, <code>delay1</code>은 전체 주문을 한 거래일 더 늦춘 주 분석입니다.</p>')
    add(table(sensitivities,[('budget','배분','text'),('delay','추가 지연','integer'),('cost_bp','비용 bp','integer'),('comparison','비교','text'),('sharpe_gap','샤프 차이','signed'),('volatility_ratio','변동성 비율','number'),('cumulative_gap','누적수익 차이','pp')]))
    add('<h3>주 분석의 실제 노출과 거래량</h3>')
    add(table(original,[('cost_bp','비용 bp','integer'),('style','전략','text'),('gross_exposure_mean','평균 총노출','number'),('net_exposure_mean','평균 순노출','number'),('turnover_annual','연 거래대금/NAV','number'),('hhi_mean','평균 집중도 HHI','number'),('frozen_gross_max','잔여 총노출 최대','number')]))
    add('<p class="caption">HHI는 종목별 노출 집중도 지표로, 큰 값은 노출이 더 집중되어 있음을 뜻합니다. 미관측 종목의 수량은 동결하고 마지막 관측가격으로 평가하므로, 수익이 기록되지 않는 동안에도 잔여 노출이 존재할 수 있습니다.</p></section>')
    add('<section><h2>검증된 범위와 남는 한계</h2><ul><li>월별 유니버스는 그때까지의 504개 완전한 양수 가격으로만 구성해, 공개 코드의 전체 기간 결측률 필터에 의한 미래 선별을 제거했습니다. 다만 원파일 505종목 자체의 생존편향과 당시 지수 구성종목 문제까지 해결된 것은 아닙니다.</li><li>모든 정책은 공통 후보·방향·점수·Q 신호·실행·비용 규칙을 사용합니다. 그래도 M−B는 페어 품질, 중복, 신호, 노출과 거래량이 함께 달라지는 선정 정책 전체의 차이입니다.</li><li>U는 하나의 사전 고정된 무가중 조합입니다. 무작위 비교군의 분포나 모든 비중복 선정법을 대표하지 않습니다.</li><li>일별 잔차/BETA를 504거래일 회귀의 마지막 값으로 만드는 방식은 공개되지 않은 생성기의 보완 가정입니다. 양방향 중 좋은 통계량 선택과 AIC 검정도 명시적 실험 규칙이며, p값을 선별 후의 엄밀한 유의확률로 읽지 않습니다.</li><li>거래비용은 거래대금 비례 민감도입니다. 대차 가능성·대차료·시장 충격·거래 제약을 모두 검증한 실전 수익성 결과가 아닙니다.</li></ul>')
    add('<p>분석 코드는 입력 수익에서 순자산을 재구성하고, 저장된 순자산의 변화율과 대조했습니다. 성과 지표는 별도의 한 번 순회 계산으로도 재검산했습니다. 자세한 수치와 입력 해시는 아래 파일에 있습니다.</p>')
    add('<nav>'+''.join(f'<a href="../results/{name}">{label}</a>' for name,label in [('metrics.csv','전체 성과 CSV'),('comparisons.csv','정책 차이 CSV'),('bootstrap.csv','공통 블록 구간 CSV'),('annual_primary.csv','연도별 CSV'),('analysis_checks.json','검산·입력 식별 JSON')])+'</nav></section>')
    timestamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    add(f'<footer>생성 시각 {timestamp} · 모든 표와 그림은 이번 일별 결과 파일에서 생성했습니다.<p>이 결과는 지정한 역사 자료와 규칙에서의 선정 정책 비교이며, 투자 전략의 실전 유효성이 확정되었다는 뜻은 아닙니다.</p></footer>')
    css='''*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#172630;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;font-size:16px;line-height:1.78}main{max-width:1120px;margin:40px auto;padding:44px 52px;background:white;border-radius:12px;box-shadow:0 5px 30px #1830400c}header{padding:8px 0 24px;border-bottom:3px solid #087f8c}h1{font-size:38px;line-height:1.3;letter-spacing:-1px;margin:10px 0 16px}h2{font-size:25px;line-height:1.4;margin:0 0 18px}h3{font-size:19px;margin:28px 0 12px}.eyebrow{font-size:13px;letter-spacing:1.4px;color:#087f8c;font-weight:700}.lead{color:#526574}section{padding:34px 0;border-bottom:1px solid #dce4e9}p{margin:12px 0}.callout{padding:18px 22px;border-left:4px solid #087f8c;background:#edf7f7;margin:20px 0}.warning{background:#fff6eb;border-color:#ce8a2a}.table-wrap{overflow-x:auto;margin:14px 0}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;line-height:1.5}th{text-align:right;background:#e9f0f3;padding:11px 10px;white-space:nowrap}td{padding:10px;border-bottom:1px solid #e2e8ed;text-align:right}th:first-child,td:first-child{text-align:left}tr:nth-child(even) td{background:#f8fafb}.caption,figcaption{font-size:14px;color:#566874}figure{margin:26px 0}figure img{width:100%;height:auto;display:block}figcaption{margin-top:9px}li{margin-bottom:9px}details{margin-top:20px;padding:14px;background:#f6f8fa}summary{cursor:pointer;font-weight:600}nav{display:flex;flex-wrap:wrap;gap:12px}a{color:#087f8c;text-decoration-thickness:1px;text-underline-offset:3px}code{background:#edf2f5;padding:2px 5px;border-radius:3px}footer{font-size:12px;color:#687a87;padding-top:26px}@media(max-width:760px){main{margin:0;padding:26px 20px;border-radius:0}h1{font-size:29px}h2{font-size:22px}body{font-size:16px}table{min-width:650px}}'''
    document='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>매칭 전략 기준 모델 검증</title><style>'+css+'</style></head><body><main>'+''.join(pieces)+'</main></body></html>'
    report_dir=out/'reports'
    report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir/'validation.html').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daily-dir',type=Path,default=ROOT/'results/daily')
    parser.add_argument('--require-clean',action='store_true')
    args=parser.parse_args()
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True)
    data,meta,dates,sources=load_inputs(args.daily_dir,args.require_clean)
    perf=all_metrics(data,meta,dates);gaps=compare_metrics(perf)
    boot=bootstrap_primary(data,meta);annual=annual_comparisons(data,meta,dates)
    for name,frame in [('metrics',perf),('comparisons',gaps),('bootstrap',boot),('annual_primary',annual)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    plot_results(data,meta,boot,out/'figures')
    protocol=ROOT/'config/model_spec.json'
    audit=dict(passed=True,checks=len(CHECKS),max_absolute_error=max(c['max_absolute_error'] for c in CHECKS),
        details=CHECKS,sources=sources,dates=[str(dates.min().date()),str(dates.max().date())],days=len(dates),
        input_paths=len(data),primary='M-B Sharpe gap at stockgross, delay1, 5bp; 21-session paired block 95% interval',
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        bootstrap=dict(method='joint-date circular moving blocks, explicit ordered samples',seed_base=SEED,
                       main_length=21,main_reps=4000,sensitivity_lengths=[5,63],sensitivity_reps=2000,
                       clean_scope='21-day primary intervals only if clean inputs exist',
                       fraction_note='Empirical resample fraction above null, not a posterior probability or frequentist p-value',
                       multiplicity_note='Nominal 95% intervals without multiple-comparison adjustment; only original-universe 5bp M-B Sharpe gap is the predeclared primary contrast.'),
        limitations=['Conditional on fixed settings and historical return paths; not a fresh holdout.',
                     'Return-row exclusion does not rerun or repair price, selection, signal, or portfolio paths.',
                     'Subperiods inherit holdings/capital from the continuous run, not a fresh-start simulation.',
                     'Original 505-column data may retain survivorship/index-membership selection.',
                     'Cost sensitivity differs from the paper cost formula.',
                     'Clean exclusion is retrospective and does not fully repair source prices.'],
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    (out/'analysis_checks.json').write_text(json.dumps(clean_json(audit),ensure_ascii=False,indent=2))
    summary=dict(primary_contrast=audit['primary'],
        primary_metrics=perf[(perf.scope=='full')&(perf.budget=='stockgross')&(perf.delay==1)].to_dict('records'),
        primary_intervals=boot[boot.block_length==21].to_dict('records'),
        primary_period_comparisons=gaps[(gaps.budget=='stockgross')&(gaps.delay==1)].to_dict('records'),
        interpretation_notes=audit['limitations'],multiple_comparisons=audit['bootstrap']['multiplicity_note'])
    (out/'statistical_summary.json').write_text(json.dumps(clean_json(summary),ensure_ascii=False,indent=2))
    render_report(perf,gaps,boot,annual,meta,dates,ROOT)
    print(json.dumps({'passed':True,'paths':len(data),'checks':len(CHECKS),'max_error':audit['max_absolute_error'],
                      'report':str(ROOT/'reports/validation.html')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
