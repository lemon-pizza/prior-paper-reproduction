"""Shared ADF and rolling-OLS numerics for the fixed research baseline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import ndtr
from statsmodels.tsa import adfvalues
from statsmodels.tsa.stattools import adfuller

WINDOW = 504


def adf_pvalues(t):
    t = np.asarray(t)
    small = np.polyval(adfvalues._tau_smallps['c'][0][::-1], t)
    large = np.polyval(adfvalues._tau_largeps['c'][0][::-1], t)
    out = ndtr(np.where(t <= adfvalues._tau_stars['c'][0], small, large))
    return np.where(t > adfvalues._tau_maxs['c'][0], 1.,
                    np.where(t < adfvalues._tau_mins['c'][0], 0., out))


def _design(e, lag):
    diff = np.diff(e, axis=1)
    y = diff[:, lag:].copy()
    cols = [e[:, lag:-1]]
    cols.extend(diff[:, lag-j:-j] for j in range(1, lag+1))
    x = np.stack(cols, axis=2)
    # Frisch-Waugh elimination of the intercept retains the exact regressions.
    y -= y.mean(axis=1, keepdims=True)
    x -= x.mean(axis=1, keepdims=True)
    return x, y


def adf_aic_batch(residuals):
    """Default adfuller equivalent: common-sample AIC then selected-lag refit.

    Cholesky of the full centered Gram matrix yields RSS for every nested
    model. Ill-conditioned/near-tied/threshold-adjacent cases use statsmodels.
    Returns t, p, usedlag, icbest, fallback count. No fixed lag substitution.
    """
    e = np.asarray(residuals, float)
    if e.ndim != 2 or not np.isfinite(e).all():
        raise ValueError('Expected complete residual rows')
    nseries, n = e.shape
    maxlag = min(int(np.ceil(12.0*(n/100.0)**.25)), n//2-2)
    x, y = _design(e, maxlag)
    gram = x.transpose(0, 2, 1) @ x
    xy = np.einsum('bni,bn->bi', x, y)
    yy = np.einsum('bn,bn->b', y, y)
    fallback = np.zeros(nseries, dtype=bool)
    # Batched Cholesky may fail if one series is singular. Isolate those rows.
    chol = np.empty_like(gram)
    try:
        chol[:] = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError:
        for i in range(nseries):
            try:
                chol[i] = np.linalg.cholesky(gram[i])
            except np.linalg.LinAlgError:
                fallback[i] = True
                chol[i] = np.eye(maxlag+1)
    whitened = np.linalg.solve(chol, xy[..., None])[..., 0]
    rss = yy[:, None] - np.cumsum(whitened**2, axis=1)
    ncommon = y.shape[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        aic = ncommon*(np.log(2*np.pi)+1+np.log(rss/ncommon)) + 2*np.arange(2, maxlag+3)
    fallback |= (~np.isfinite(aic)).any(axis=1)
    order = np.argsort(aic, axis=1, kind='stable')
    usedlag = order[:, 0].copy()
    icbest = aic[np.arange(nseries), usedlag]
    with np.errstate(invalid='ignore'):
        gap = aic[np.arange(nseries), order[:, 1]] - icbest
    fallback |= gap < 1e-8
    # Diagonal-based warning is conservative; exact library fallback is cheap.
    diagonals = np.diagonal(chol, axis1=1, axis2=2)
    fallback |= diagonals.min(axis=1) < 1e-8*diagonals.max(axis=1)
    t = np.full(nseries, np.nan)
    for lag in np.unique(usedlag[~fallback]):
        ii = np.flatnonzero((usedlag == lag) & ~fallback)
        xx, dep = _design(e[ii], int(lag))
        gg = xx.transpose(0, 2, 1) @ xx
        gdep = np.einsum('bni,bn->bi', xx, dep)
        rhs = np.zeros((len(ii), lag+1, 2))
        rhs[:, :, 0] = gdep
        rhs[:, 0, 1] = 1.
        try:
            solved = np.linalg.solve(gg, rhs)
        except np.linalg.LinAlgError:
            fallback[ii] = True
            continue
        beta = solved[:, :, 0]
        errors = dep - np.einsum('bni,bi->bn', xx, beta)
        ssr = np.einsum('bn,bn->b', errors, errors)
        variance = ssr/(dep.shape[1]-(lag+2))
        with np.errstate(divide='ignore', invalid='ignore'):
            t[ii] = beta[:, 0]/np.sqrt(variance*solved[:, 0, 1])
    p = adf_pvalues(t)
    fallback |= ~np.isfinite(t) | (np.abs(p-.05) < 1e-8)
    for i in np.flatnonzero(fallback):
        try:
            ans = adfuller(e[i])
            t[i], p[i], usedlag[i], icbest[i] = ans[0], ans[1], ans[2], ans[5]
        except (ValueError, np.linalg.LinAlgError):
            # Source cache_spread catches failed/constant pair regressions.
            t[i], p[i], usedlag[i], icbest[i] = np.nan, np.nan, -1, np.nan
    return t, p, usedlag, icbest, int(fallback.sum())


def calculate_edges(log_prices, directed=False, batch_size=256):
    a = log_prices.to_numpy(float)
    centered = a-a.mean(axis=0)
    cov = centered.T@centered
    if directed:
        i, j = np.where(~np.eye(a.shape[1], dtype=bool))
    else:
        i, j = np.triu_indices(a.shape[1], k=1)
    beta = np.divide(cov[i, j], cov[i, i], out=np.zeros(len(i)), where=cov[i, i] != 0)
    alpha = a.mean(axis=0)[j]-beta*a.mean(axis=0)[i]
    t, p, usedlag, aic = (np.empty(len(i)) for _ in range(4))
    fallback_count = 0
    for start in range(0, len(i), batch_size):
        stop = start+batch_size
        e = (a[:, j[start:stop]] - beta[start:stop]*a[:, i[start:stop]] - alpha[start:stop]).T
        tt, pp, ll, aa, ff = adf_aic_batch(e)
        t[start:stop], p[start:stop], usedlag[start:stop], aic[start:stop] = tt, pp, ll, aa
        fallback_count += ff
    names = log_prices.columns.to_numpy()
    frame = pd.DataFrame({'ASSET_1': names[i], 'ASSET_2': names[j],
                          'BETA': beta, 'ALPHA': alpha, 'T_STAT': t,
                          'P_VALUE': p, 'USEDLAG': usedlag.astype(int), 'AIC': aic})
    # Resolve any floating-point-near sort tie with the public sklearn+ADF path.
    # Graph insertion order can affect an unweighted matching, so ordering matters.
    ordered = np.argsort(t, kind='stable')
    close = np.flatnonzero(np.diff(t[ordered]) < 1e-9)
    ambiguous = np.unique(np.r_[ordered[close], ordered[close+1]])
    if len(ambiguous):
        from sklearn.linear_model import LinearRegression
        for row in ambiguous:
            x, y = a[:, i[row]], a[:, j[row]]
            fit = LinearRegression().fit(np.column_stack([x, np.ones(len(x))]), y)
            residual = y-fit.coef_[0]*x-fit.intercept_
            exact = adfuller(residual)
            frame.loc[row, ['BETA', 'ALPHA', 'T_STAT', 'P_VALUE', 'USEDLAG', 'AIC']] = [
                fit.coef_[0], fit.intercept_, exact[0], exact[1], exact[2], exact[5]]
    failed = ~np.isfinite(frame.T_STAT) | ~np.isfinite(frame.P_VALUE)
    skipped = int(failed.sum())
    frame = frame.loc[~failed].copy()
    frame.attrs['exact_sort_tie_refits'] = int(len(ambiguous))
    frame.attrs['adf_failed_pairs_skipped'] = skipped
    # Same sort as pairs.py / betas.py cache_spread, including default sort kind.
    return frame.sort_values('P_VALUE').reset_index(drop=True), fallback_count


def rolling_endpoint(x,y,window=WINDOW):
    """Complete-window intercept OLS; no price imputation or future backfill."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim==1:x=x[:,None];y=y[:,None]
    ok=np.isfinite(x)&np.isfinite(y)
    xx=np.where(ok,x,0.);yy=np.where(ok,y,0.)
    def roll(a):
        cs=np.vstack([np.zeros((1,a.shape[1])),np.cumsum(a,axis=0)])
        z=np.full_like(a,np.nan,dtype=float)
        if len(a)>=window:z[window-1:]=cs[window:]-cs[:-window]
        return z
    n=roll(ok.astype(float));sx=roll(xx);sy=roll(yy)
    vx=roll(xx*xx)-sx*sx/window
    cov=roll(xx*yy)-sx*sy/window
    with np.errstate(divide='ignore',invalid='ignore'):
        beta=cov/vx;alpha=(sy-beta*sx)/window
    beta[(n<window)|(vx<1e-12)]=np.nan
    alpha[~np.isfinite(beta)]=np.nan
    spread=y-alpha-beta*x
    return beta,spread

