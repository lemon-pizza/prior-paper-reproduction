"""Signed risk-change accounting and conditional month-block covariance intervals.

The month-block intervals resample estimated within-month covariance matrices,
not raw daily returns. They describe stability of signed monthly components and
are deliberately separate from the daily Sharpe bootstrap in statistics.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
COMPONENTS=['own','overlap','disjoint','frozen_variance','frozen_cross','between_month']


def main():
    out=ROOT/'results'
    d=pd.read_csv(out/'risk_decomposition.csv')
    d=d[d.scope=='within_month_plus_between'].set_index('scenario')
    contrasts=[]
    for mode in ['Q','Z']:
        for a,b in [(f'paper_daily_M{mode}',f'paper_daily_B{mode}'),
                    (f'equal_pair_M{mode}',f'equal_pair_B{mode}'),
                    *[(f'control_C1_{mode}',f'control_{c}_{mode}') for c in ['C2','C4','Cinf']]]:
            if a not in d.index or b not in d.index:continue
            row=dict(treatment=a,reference=b,mode=mode,actual=d.loc[a,'actual']-d.loc[b,'actual'])
            for c in COMPONENTS:row[c]=d.loc[a,c]-d.loc[b,c]
            row['sum']=sum(row[c] for c in COMPONENTS)
            row['error']=row['sum']-row['actual']
            for c in COMPONENTS+['actual']:row[c+'_annual']=row[c]*252
            contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(out/'risk_gap_accounting.csv',index=False)
    m=pd.read_csv(out/'risk_monthly.csv'); scenarios=sorted(m.scenario.unique())
    months=sorted(m.month.unique());n=len(months);length=3;reps=4000;seed=240308001
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(reps,(n+length-1)//length))
    ix=((starts[:,:,None]+np.arange(length))%n).reshape(reps,-1)[:,:n]
    counts=np.zeros((reps,n))
    for j,v in enumerate(ix):counts[j]=np.bincount(v,minlength=n)
    rows=[];boot={}
    for scenario in scenarios:
        s=m[m.scenario==scenario].set_index('month').loc[months]
        days=s.days.to_numpy();denominator=counts@days-1
        for c in COMPONENTS[:-1]:
            v=(days-1)*s[c].to_numpy()
            point=v.sum()/(days.sum()-1)
            assert np.isclose(point,d.loc[scenario,c],rtol=1e-10,atol=1e-18), (scenario,c)
            samples=counts@v/denominator
            boot[(scenario,c)]=samples
            lo,hi=np.quantile(samples,[.025,.975])
            rows.append(dict(scenario=scenario,component=c,estimate=point,ci_low=lo,ci_high=hi,
                             estimate_annual=point*252,ci_low_annual=lo*252,ci_high_annual=hi*252,
                             positive_months=int((s[c]>1e-18).sum()),negative_months=int((s[c]<-1e-18).sum()),
                             block_months=length,reps=reps))
    pd.DataFrame(rows).to_csv(out/'risk_month_block_intervals.csv',index=False)
    deltas=[]
    for contrast in contrasts:
        for c in COMPONENTS[:-1]:
            a=contrast['treatment'];b=contrast['reference'];samples=boot[a,c]-boot[b,c]
            lo,hi=np.quantile(samples,[.025,.975])
            deltas.append(dict(treatment=a,reference=b,mode=contrast['mode'],component=c,
                               estimate=contrast[c],ci_low=lo,ci_high=hi,
                               estimate_annual=contrast[c]*252,ci_low_annual=lo*252,ci_high_annual=hi*252))
    pd.DataFrame(deltas).to_csv(out/'risk_gap_month_intervals.csv',index=False)
    meta=dict(seed=seed,reps=reps,block_length_months=length,months=n,
              method='Common circular blocks of 3 months; component = sum((days-1)*monthly_covariance_component)/(sum(days)-1).',
              caveat='Conditional on estimated monthly covariances; descriptive covariance stability, not the daily return bootstrap. Between-month means term omitted from these intervals, retained in exact accounting.',
              point_matches_exact_within_components=True)
    (out/'risk_statistics_manifest.json').write_text(json.dumps(meta,indent=2))
    print(pd.DataFrame(contrasts).to_string(index=False))

if __name__=='__main__':main()
