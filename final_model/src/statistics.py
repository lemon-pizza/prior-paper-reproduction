"""Independently recalculate corrected daily outputs and primary paired intervals."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
def moments(r):
    r = np.asarray(r, float)
    n = len(r)
    mean = sum(r) / n
    variance = sum((r-mean)**2) / (n-1)
    sd = variance**.5
    nav, peak, worst = 1., 1., 0.
    for ret in r:
        nav *= 1+ret
        peak = max(peak, nav)
        worst = min(worst, nav/peak-1)
    return dict(days=n, mean_daily=mean, mean_annual=mean*252,
                volatility=sd*252**.5, sharpe=mean/sd*252**.5,
                cumulative=nav-1, cagr=nav**(252/n)-1, max_drawdown=worst)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT
COSTS=['gross','net_fixed','net_turnover','net_5bp','net_10bp']
STYLES=['MQ','MZ','BQ','BZ']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--daily-dir',type=Path,default=ROOT/'results/daily')
    args=ap.parse_args()
    rows=[];frames={}
    for path in sorted(args.daily_dir.glob('*.csv')):
        scenario,style=path.stem.rsplit('_',1)
        d=pd.read_csv(path,index_col='date',parse_dates=True)
        frames[path.stem]=d
        for cost in COSTS:
            met=moments(d[cost]);rows.append(dict(scenario=scenario,style=style,cost=cost,**met))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'results/metrics.csv',index=False)
    keys=[(s,c) for s in STYLES for c in COSTS]
    values=np.column_stack([frames['paper_daily_'+s][c] for s,c in keys])
    n=len(values);length=21;reps=4000;seed=240307998+21
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(reps,(n+length-1)//length))
    mean=[];vol=[];sharpe=[]
    for a in range(0,reps,100):
        ix=((starts[a:a+100,:,None]+np.arange(length))%n).reshape(-1,((n+length-1)//length)*length)[:,:n]
        sample=values[ix];m=sample.mean(axis=1)
        sd=np.sqrt(((sample-m[:,None,:])**2).sum(axis=1)/(n-1))
        mean.append(m*252);vol.append(sd*np.sqrt(252));sharpe.append(m/sd*np.sqrt(252))
    boot=dict(mean_annual=np.concatenate(mean),volatility=np.concatenate(vol),sharpe=np.concatenate(sharpe))
    intervals=[]
    for j,(style,cost) in enumerate(keys):
        mask=(result.scenario=='paper_daily')&(result['style']==style)&(result.cost==cost)
        for metric,values_b in boot.items():
            ci=np.quantile(values_b[:,j],[.025,.975])
            result.loc[mask,metric+'_ci_low']=ci[0];result.loc[mask,metric+'_ci_high']=ci[1]
    result.to_csv(OUT/'results/metrics.csv',index=False)
    for mode in ['Q','Z']:
        for cost in COSTS:
            a=keys.index(('M'+mode,cost));b=keys.index(('B'+mode,cost))
            dist=dict(sharpe_gap=boot['sharpe'][:,a]-boot['sharpe'][:,b],
                      mean_annual_gap=boot['mean_annual'][:,a]-boot['mean_annual'][:,b],
                      volatility_gap=boot['volatility'][:,a]-boot['volatility'][:,b],
                      volatility_ratio=boot['volatility'][:,a]/boot['volatility'][:,b],
                      variance_ratio=(boot['volatility'][:,a]/boot['volatility'][:,b])**2)
            ra=moments(frames['paper_daily_M'+mode][cost]);rb=moments(frames['paper_daily_B'+mode][cost])
            estimates=dict(sharpe_gap=ra['sharpe']-rb['sharpe'],mean_annual_gap=ra['mean_annual']-rb['mean_annual'],
                           volatility_gap=ra['volatility']-rb['volatility'],volatility_ratio=ra['volatility']/rb['volatility'],
                           variance_ratio=(ra['volatility']/rb['volatility'])**2)
            for metric,sample in dist.items():
                ci=np.quantile(sample,[.025,.975])
                intervals.append(dict(mode=mode,cost=cost,metric=metric,estimate=estimates[metric],
                                      ci_low=ci[0],ci_high=ci[1],block_length=length,reps=reps,seed=seed))
    pd.DataFrame(intervals).to_csv(OUT/'results/primary_bootstrap.csv',index=False)
    manifest=dict(daily_directory=str(args.daily_dir.resolve()),daily_files=len(frames),metrics_rows=len(rows),
                  primary_joint_bootstrap=dict(length=length,reps=reps,seed=seed,method='materialized ordered resampled arrays'),
                  cost_paths=COSTS)
    (OUT/'audit/statistics_run.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(result[(result.scenario=='paper_daily')][['style','cost','volatility','sharpe','cumulative','max_drawdown']].to_string(index=False))


if __name__=='__main__':main()
