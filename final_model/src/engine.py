"""Fixed verified model: raw prices, rolling OLS and self-financing execution."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data'

def load_data(path=None):
    p = pd.read_parquet(path or SOURCE / 'prices.parquet.gz').sort_index()
    if p.index.has_duplicates:
        raise ValueError('Duplicate price dates')
    if not p.index.is_monotonic_increasing:
        raise ValueError('Nonmonotonic dates')
    return p.astype(float).where(p > 0)

def rolling_beta(x, y, window=504):
    """OLS with intercept on observations through each row, inclusive. No backfill."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    one = x.ndim == 1
    if one: x, y = x[:, None], y[:, None]
    ok = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(ok,x,0), np.where(ok,y,0)
    def roll(a):
        c = np.vstack([np.zeros((1,a.shape[1])), np.cumsum(a, axis=0)])
        z = np.full_like(a, np.nan, dtype=float)
        z[window-1:] = c[window:] - c[:-window]
        return z
    count = roll(ok.astype(float)); sx, sy = roll(xx), roll(yy)
    vx = roll(xx*xx)-sx*sx/window
    cov = roll(xx*yy)-sx*sy/window
    with np.errstate(divide='ignore', invalid='ignore'):
        b = cov/vx; a = (sy-b*sx)/window
        sse = roll(yy*yy)-sy*sy/window-b*cov
        sigma = np.sqrt(np.maximum(sse,0)/(window-2))
    b[(count < window)|(vx < 1e-12)] = np.nan
    a[~np.isfinite(b)] = np.nan; sigma[~np.isfinite(b)] = np.nan
    res = y-a-b*x
    return tuple(z[:,0] if one else z for z in (a,b,res,sigma))

def backtest_weights(target_weights, raw_prices):
    """Targets at yesterday's close; missing quotes freeze held quantities.

    Weights and P&L are per unit start-of-day gross-account equity. Net costs
    are reported afterwards from this same gross target path, preventing
    portfolio-specific cost rules. Terminal turnover includes observed final quotes only; stale holdings remain marked.
    """
    raw = np.asarray(raw_prices,float)
    p = pd.DataFrame(raw).ffill().to_numpy()
    with np.errstate(invalid='ignore',divide='ignore'):
        r = np.nan_to_num(p[1:]/p[:-1]-1, nan=0.,posinf=0.,neginf=0.)
    target = np.nan_to_num(np.asarray(target_weights,float),nan=0.)
    if target.shape != r.shape: raise ValueError('target/return shape mismatch')
    w = np.zeros_like(target); contrib = np.zeros_like(target)
    gross=np.zeros(len(target)); turn=np.zeros(len(target)); frozen=np.zeros(len(target),int)
    prev=np.zeros(raw.shape[1]); prev_r=np.zeros(raw.shape[1]); prev_pnl=0.
    for t in range(len(target)):
        drift = prev*(1+prev_r)/(1+prev_pnl)
        tradable = np.isfinite(raw[t]) & (raw[t]>0)
        now=np.where(tradable,target[t],drift)
        frozen[t]=np.count_nonzero((~tradable)&(np.abs(drift)>1e-14))
        turn[t]=np.abs(now-drift).sum()
        contrib[t]=now*r[t]; gross[t]=contrib[t].sum(); w[t]=now
        prev,prev_r,prev_pnl=now,r[t],gross[t]
    if len(turn):
        # Final orders can be executed only for assets with an observed final quote.
        terminal_tradable = np.isfinite(raw[-1]) & (raw[-1] > 0)
        turn[-1] += np.abs(w[-1]*(1+r[-1]))[terminal_tradable].sum()
    return dict(gross=gross,turnover=turn,weights=w,contrib=contrib,
                frozen_assets=frozen,asset_returns=r)

def make_targets(features, prices, flag, mode, equal=False):
    """Locked Q/Z signals and beta-dollar legs, normalized by active gross.

    Keep arithmetic order from the independently reviewed implementation.
    All experimental normalization, units, sign and threshold grids are removed.
    """
    d=features[features[flag]].copy()
    score=d['q_fit' if mode=='Q' else 'z_fit'].to_numpy()
    if mode=='Q':
        signal=-np.sign(score)*np.round(np.abs(score))
    elif mode=='Z':
        score=np.clip(score,-3,3)
        signal=np.where(score>=2.,-1.,np.where(score<=-2.,1.,0.))
    else:
        raise ValueError(mode)
    signal=np.nan_to_num(signal,nan=0.,posinf=0.,neginf=0.)
    beta=d.beta.to_numpy();signal[~np.isfinite(beta)]=0;beta=np.nan_to_num(beta)
    px=prices.ffill().to_numpy();ix=prices.index.get_indexer(d.date)
    xi=d.xi.to_numpy();yi=d.yi.to_numpy()
    h1=-signal*beta/px[ix-1,xi];h2=signal.copy()/px[ix-1,yi]
    cap=np.abs(h1*px[ix-1,xi])+np.abs(h2*px[ix-1,yi])
    if equal:
        unit=np.where(abs(signal)>0,cap/np.maximum(abs(signal),1e-30),np.nan)
        h1=np.nan_to_num(h1/unit);h2=np.nan_to_num(h2/unit)
        cap=np.abs(signal)
    den=pd.Series(cap,index=d.index).groupby(d.date).transform('sum').to_numpy()
    h1=np.divide(h1,den,out=np.zeros_like(h1),where=den>0)
    h2=np.divide(h2,den,out=np.zeros_like(h2),where=den>0)
    d['signal']=signal;d['beta_used']=beta;d['score']=score
    d['holding_x']=h1;d['holding_y']=h2
    d['price_x_prev']=px[ix-1,xi];d['price_y_prev']=px[ix-1,yi]
    target=np.zeros(prices.shape)
    np.add.at(target,(ix,xi),h1*px[ix-1,xi]);np.add.at(target,(ix,yi),h2*px[ix-1,yi])
    return target,d
