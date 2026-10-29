"""Verified 4 primary + 16 fixed auxiliary paths; netted cash and risk ledger."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from engine import load_data,make_targets,backtest_weights
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'

def var(x): return np.var(x,axis=0,ddof=1)

def variance_parts(c,incidence):
    """2*sum unordered covariance, no independence restriction. Works on sparse C."""
    n=c.shape[0]
    means=np.asarray(c.mean(axis=0)).ravel()
    diagonal=(float(c.multiply(c).sum())-n*float(means@means))/(n-1)
    incident=(c@incidence).toarray()
    overlap=float(var(incident).sum()-2*diagonal)
    pair_total=np.asarray(c.sum(axis=1)).ravel()
    disjoint=float(var(pair_total)-diagonal-overlap)
    return diagonal,overlap,disjoint,pair_total

def decompose(name,dates,c,incidence,frozen,actual):
    total_n=len(dates); mean=float(actual.mean()); monthly=[]
    allparts=variance_parts(c,incidence)
    frozen_var=float(var(frozen)); frozen_cross=2*float(np.cov(allparts[3],frozen,ddof=1)[0,1])
    full=dict(scenario=name,scope='full_pair_identity',own=allparts[0],overlap=allparts[1],
              disjoint=allparts[2],frozen_variance=frozen_var,frozen_cross=frozen_cross,between_month=0.,actual=float(var(actual)))
    totals=np.zeros(6)
    for month in dates.to_period('M').unique():
        ids=np.where(dates.to_period('M')==month)[0];cm=c[ids]
        d,o,u,pt=variance_parts(cm,incidence)
        fv=float(var(frozen[ids]));fc=2*float(np.cov(pt,frozen[ids],ddof=1)[0,1])
        between=len(ids)*(actual[ids].mean()-mean)**2/(total_n-1)
        scale=(len(ids)-1)/(total_n-1)
        parts=np.array([d,o,u,fv,fc])*scale
        totals[:5]+=parts;totals[5]+=between
        # Positive/negative covariance separately, classified by endpoint intersection.
        used=np.flatnonzero(np.asarray(cm.multiply(cm).sum(axis=0)).ravel()>0)
        cc=cm[:,used].toarray();v=np.atleast_2d(np.cov(cc,rowvar=False,ddof=1))
        common=(incidence[used]@incidence[used].T).toarray()>0
        upper=np.triu(np.ones(v.shape,dtype=bool),1)
        ov=2*v[upper&common];di=2*v[upper&~common]
        row=dict(scenario=name,month=str(month),days=len(ids),own=d,overlap=o,disjoint=u,
                 frozen_variance=fv,frozen_cross=fc,actual=float(var(actual[ids])),weight=scale,between_contribution=between,
                 overlap_positive=float(ov[ov>0].sum()),overlap_negative=float(ov[ov<0].sum()),
                 disjoint_positive=float(di[di>0].sum()),disjoint_negative=float(di[di<0].sum()))
        row['error']=row['actual']-sum(row[k] for k in ['own','overlap','disjoint','frozen_variance','frozen_cross'])
        monthly.append(row)
    within=dict(scenario=name,scope='within_month_plus_between',**dict(zip(['own','overlap','disjoint','frozen_variance','frozen_cross','between_month'],totals)),actual=float(var(actual)))
    for row in [full,within]:
        row['sum']=sum(row[k] for k in ['own','overlap','disjoint','frozen_variance','frozen_cross','between_month'])
        row['error']=row['sum']-row['actual']
        assert abs(row['error'])<1e-14,(name,row)
        for k in ['own','overlap','disjoint','frozen_variance','frozen_cross','between_month','actual']:
            row[k+'_annual']=row[k]*252
    assert max(abs(r['error']) for r in monthly)<1e-14
    return [full,within],monthly

def execute(name,features,prices,flag,mode,equal=False,budget=False,lag2=False,verify_reference=False):
    target,detail=make_targets(features,prices,flag,mode,equal)
    rows=detail.sort_values('date',kind='stable').reset_index(drop=True)
    dateidx=np.where((prices.index>='2017-01-01')&(prices.index<='2023-05-31'))[0]
    first,last=dateidx[0],dateidx[-1]+1;dates=prices.index[first:last]
    if lag2:
        # Delay the entire target-dollar instruction by one session, including
        # monthly pair replacement. Recompute quantities at execution prices.
        oldloc=dates.get_indexer(rows.date)
        keep=oldloc<len(dates)-1
        rows=rows.loc[keep].copy()
        rows['date']=dates[oldloc[keep]+1].to_numpy()
        rows=rows.reset_index(drop=True)
    raw=prices.iloc[first-1:last].to_numpy();marks=pd.DataFrame(raw).ffill().to_numpy()
    ret=np.nan_to_num(marks[1:]/marks[:-1]-1,nan=0.,posinf=0.,neginf=0.)
    ti=dates.get_indexer(rows.date);xi=rows.xi.to_numpy();yi=rows.yi.to_numpy()
    wx0=(rows.holding_x*rows.price_x_prev).to_numpy();wy0=(rows.holding_y*rows.price_y_prev).to_numpy()
    pairs=rows[['x','y']].drop_duplicates().reset_index(drop=True)
    lookup={(x,y):i for i,(x,y) in enumerate(pairs.itertuples(index=False,name=None))}
    pi=np.fromiter((lookup[(x,y)] for x,y in rows[['x','y']].itertuples(index=False,name=None)),int)
    pxi=np.array([prices.columns.get_loc(x) for x in pairs.x]);pyi=np.array([prices.columns.get_loc(y) for y in pairs.y])
    incidence=sparse.csr_matrix((np.ones(2*len(pairs)),(np.r_[np.arange(len(pairs)),np.arange(len(pairs))],np.r_[pxi,pyi])),shape=(len(pairs),prices.shape[1]))
    bounds=np.searchsorted(ti,np.arange(len(dates)+1));n=len(dates)
    weights=np.zeros((n,prices.shape[1]));fcontrib=np.zeros_like(weights)
    pc=np.zeros(len(rows));wx=np.zeros(len(rows));wy=np.zeros(len(rows));daily=[]
    prev=np.zeros(prices.shape[1]);prevret=np.zeros_like(prev);prevr=0.;nav=1.;shares=np.zeros_like(prev);cash=1.;audits=[]
    frozen_records=[]
    for t in range(n):
        a,b=bounds[t:t+2];sel=slice(a,b)
        drift=prev*(1+prevret)/(1+prevr)
        trade=np.isfinite(raw[t])&(raw[t]>0)
        fweight=np.where(trade,0.,drift)
        xx=wx0[sel]*trade[xi[sel]];yy=wy0[sel]*trade[yi[sel]]
        available=float(np.abs(xx).sum()+np.abs(yy).sum());fgross=float(np.abs(fweight).sum())
        scale=max(0.,1-fgross)/available if budget and available>0 else 1.
        if budget: assert fgross<=1+1e-10,'Frozen holdings exhaust budget'
        xx*=scale;yy*=scale;wx[sel]=xx;wy[sel]=yy
        now=fweight.copy();np.add.at(now,xi[sel],xx);np.add.at(now,yi[sel],yy)
        pc[sel]=xx*ret[t,xi[sel]]+yy*ret[t,yi[sel]]
        fc=fweight*ret[t];fcontrib[t]=fc
        r=float(now@ret[t]);turn=float(np.abs(now-drift).sum())
        if t==n-1:
            # Do not charge a fictitious terminal sale for missing-quote holdings.
            terminal_tradable=np.isfinite(raw[t+1])&(raw[t+1]>0)
            turn+=float(np.abs(now*(1+ret[t]))[terminal_tradable].sum())
        weights[t]=now;pre=float(np.abs(xx).sum()+np.abs(yy).sum()+fgross);post=float(np.abs(now).sum())
        net_before=float(xx.sum()+yy.sum()+fweight.sum())
        assert abs(net_before-now.sum())<1e-12
        stock_pre=np.abs(fweight);np.add.at(stock_pre,xi[sel],np.abs(xx));np.add.at(stock_pre,yi[sel],np.abs(yy))
        daily.append(dict(gross=r,net_fixed=r-.01/252*(post>0),net_turnover=r-.01/252*turn,net_5bp=r-.0005*turn,net_10bp=r-.001*turn,
            turnover=turn,gross_pre_net=pre,gross_post_net=post,net_exposure=float(now.sum()),net_before_offset=net_before,net_after_offset=float(now.sum()),cash_weight=float(1-now.sum()),
            offset_fraction=1-post/pre if pre else 0.,hhi_pre=float(((stock_pre/pre)**2).sum()) if pre else 0.,
            hhi_post=float(((now/post)**2).sum()) if post else 0.,max_weight=float(np.abs(now).max()),
            frozen_gross=fgross,frozen_assets=int(np.count_nonzero(fweight)),frozen_return=float(fc.sum()),
            pair_return=float(pc[sel].sum()),scale=scale,active_pairs=int(np.count_nonzero((np.abs(xx)+np.abs(yy))>0))))
        # Independent actual quantity / cash ledger, including netted transactions.
        newshares=np.divide(now*nav,marks[t],out=np.zeros_like(now),where=np.isfinite(marks[t])&(marks[t]>0))
        delta=newshares-shares;cash-=float(np.nansum(delta*marks[t]));endnav=cash+float(np.nansum(newshares*marks[t+1]))
        audits.append(abs(endnav-nav*(1+r)))
        if np.any(fweight):
            ids=np.flatnonzero(fweight)
            frozen_records.extend((dates[t],prices.columns[j],fweight[j],fc[j],bool(trade[j])) for j in ids)
        shares=newshares;nav=endnav;prev,prevret,prevr=now,ret[t],r
    d=pd.DataFrame(daily,index=dates);d.index.name='date'
    c=sparse.csr_matrix((pc,(ti,pi)),shape=(n,len(pairs)));fr=fcontrib.sum(axis=1)
    attribution=np.asarray(c.sum(axis=1)).ravel()+fr
    maxerr=float(np.max(np.abs(attribution-d.gross.to_numpy())))
    assert maxerr<1e-13 and max(audits)<1e-12
    if budget:
        # Same 1-NAV investment budget, but an all-zero signal legitimately
        # remains in cash. Do not invent a trade just to consume the budget.
        assert d.gross_pre_net.max()<=1+1e-10
        active=d.active_pairs>0
        assert np.max(np.abs(d.loc[active,'gross_pre_net']-1))<1e-10
    d['unused_gross_budget']=np.maximum(0.,1-d.gross_pre_net) if budget else np.nan
    replay_error=None
    if verify_reference:
        reference=backtest_weights(target[first:last],raw)
        replay_error=float(np.max(np.abs(reference['gross']-d.gross.to_numpy())))
        assert replay_error<1e-12
        assert np.max(np.abs(reference['weights']-weights))<1e-12
        assert np.max(np.abs(reference['turnover']-d.turnover.to_numpy()))<1e-12
    # Market proxy = equal-weight available stock returns; explicitly not official SP500.
    valid=np.isfinite(raw[:-1])&np.isfinite(raw[1:])
    proxy=np.nanmean(np.where(valid,ret,np.nan),axis=1)
    beta=float(np.cov(d.gross,proxy,ddof=1)[0,1]/var(proxy))
    residual=d.gross.to_numpy()-beta*proxy
    exposure=dict(scenario=name,market_proxy_beta=beta,market_proxy_corr=float(np.corrcoef(d.gross,proxy)[0,1]),
        market_variance_share=float(beta*beta*var(proxy)/var(d.gross)),market_residual_vol=float(np.std(residual,ddof=1)*np.sqrt(252)))
    for k in ['gross_pre_net','gross_post_net','net_exposure','cash_weight','offset_fraction','hhi_pre','hhi_post','max_weight','frozen_gross','turnover','active_pairs']:
        exposure[k+'_mean']=float(d[k].mean());exposure[k+'_max']=float(d[k].max())
    exposure['gross_pre_net_min']=float(d.gross_pre_net.min());exposure['final_frozen_gross']=float(d.frozen_gross.iloc[-1])
    exposure['inactive_signal_days']=int((d.active_pairs==0).sum())
    exposure['unused_budget_mean']=float(d.unused_gross_budget.mean()) if budget else np.nan
    for sub in ['daily','pair_ledger','frozen_ledger','asset_weights']:(RESULTS/sub).mkdir(exist_ok=True)
    d.to_csv(ROOT/'results/daily'/f'{name}.csv')
    pd.DataFrame(weights,index=dates,columns=prices.columns).to_parquet(ROOT/'results/asset_weights'/f'{name}.parquet')
    ledger=rows[['date','signal_date','x','y','signal','beta_used','score']].copy()
    ledger['weight_x']=wx;ledger['weight_y']=wy;ledger['contribution']=pc
    ledger['target_weight_x']=wx0;ledger['target_weight_y']=wy0
    ledger.to_parquet(ROOT/'results/pair_ledger'/f'{name}.parquet',index=False)
    pd.DataFrame(frozen_records,columns=['date','asset','weight','contribution','tradable']).to_parquet(ROOT/'results/frozen_ledger'/f'{name}.parquet',index=False)
    dec,monthly=decompose(name,dates,c,incidence,fr,d.gross.to_numpy())
    check=dict(scenario=name,days=n,pair_attribution_error=maxerr,cash_ledger_error=max(audits),reference_gross_error=replay_error,
        variance_error=max(abs(x['error']) for x in dec),monthly_variance_error=max(abs(x['error']) for x in monthly),
        budget_equal=bool(budget),gross_budget_error=float(np.maximum(d.gross_pre_net-1,0).max()) if budget else None,
        inactive_signal_days=int((d.active_pairs==0).sum()))
    print(name,'SR',round(d.gross.mean()/d.gross.std()*np.sqrt(252),5),'var',dec[1]['actual'],'attribution',maxerr,flush=True)
    return dec,monthly,exposure,check

def run(scope='all'):
    """Run four primary paths or all 20 previously verified paths."""
    RESULTS.mkdir(parents=True,exist_ok=True)
    files=sorted((ROOT/'data/features').glob('????-??.parquet'))
    assert len(files)==77, f'Expected 77 monthly feature files, found {len(files)}'
    features=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    prices=load_data();outputs=[];monthly=[];exposures=[];checks=[]
    specs=[]
    for scenario,equal in [('paper_daily',False),('equal_pair',True)]:
        if scope=='primary' and scenario!='paper_daily':continue
        for strategy in ['M','B']:
            for mode in ['Q','Z']:
                specs.append((f'{scenario}_{strategy}{mode}', 'is_'+strategy, mode,
                              dict(equal=equal,verify_reference=scenario=='paper_daily')))
    if scope=='all':
        for strategy in ['M','B']:
            for mode in ['Q','Z']:
                specs.append((f'paper_lag2_{strategy}{mode}','is_'+strategy,mode,dict(lag2=True)))
        for cap in ['C1','C2','C4','Cinf']:
            for mode in ['Q','Z']:
                specs.append((f'control_{cap}_{mode}','is_'+cap,mode,dict(equal=True,budget=True)))
    for name,flag,mode,kw in specs:
        a,b,c,d=execute(name,features,prices,flag,mode,**kw)
        outputs.extend(a);monthly.extend(b);exposures.append(c);checks.append(d)
    for filename,rows in [('risk_decomposition',outputs),('risk_monthly',monthly),
                          ('exposure_summary',exposures),('accounting_checks',checks)]:
        dest=RESULTS/f'{filename}.csv';new=pd.DataFrame(rows)
        if dest.exists() and scope=='primary':
            prev=pd.read_csv(dest);new=pd.concat([prev[~prev.scenario.isin(new.scenario)],new],ignore_index=True)
        new.to_csv(dest,index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['primary','all'],default='all')
    run(parser.parse_args().scope)
