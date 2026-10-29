"""Prespecified comparison of weighted matching and equally sized portfolios.

Selection is handled by selection.py. Rolling OLS is supplied by numerics.py;
trading and fee accounting follow the fixed causal execution rules.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from numerics import rolling_endpoint

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / 'data/prices.parquet.gz'
STYLES = ('M', 'B', 'U')
EXPECTED_SHA = '7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'


def load_rolling():
    """Expose the local numerical routine; replaceable for isolated timing tests."""
    return rolling_endpoint


def month_targets(prices, selected, cutoff, end, styles=STYLES):
    """Return instructions for close-to-close returns strictly after cutoff.

Each instruction uses the previous close's signal, beta and observed prices.
Selected schema is x,y,is_M,is_B,is_U; no forward filling enters signals.
"""
    rolling = load_rolling()
    left = prices.index.searchsorted(pd.Timestamp(cutoff), side='right')
    right = prices.index.searchsorted(pd.Timestamp(end), side='right')
    warm = max(0, left - 504 - 22)
    data = prices.iloc[warm:right].to_numpy(float)
    xi = prices.columns.get_indexer(selected.x)
    yi = prices.columns.get_indexer(selected.y)
    assert (xi >= 0).all() and (yi >= 0).all()
    with np.errstate(divide='ignore', invalid='ignore'):
        lx, ly = np.log(data[:, xi]), np.log(data[:, yi])
    beta, residual = rolling(lx, ly)
    # Removing the all-NaN OLS prefix does not alter the 20-observation Q window.
    trim = 503
    rr = pd.DataFrame(residual[trim:])
    with np.errstate(divide='ignore', invalid='ignore'):
        q = (rr - rr.rolling(20).median()) / (rr.rolling(20).quantile(.25) - rr.rolling(20).quantile(.75))
    signal = (np.sign(q) * np.round(q.abs())).to_numpy()
    info = np.arange(left, right) - warm - 1
    signal = signal[info - trim]
    beta = beta[info]
    px, py = data[info[:, None], xi], data[info[:, None], yi]
    good = np.isfinite(signal) & np.isfinite(beta) & np.isfinite(px) & np.isfinite(py) & (px > 0) & (py > 0)
    sx = np.where(good, -signal * beta * px, 0.)
    sy = np.where(good, signal * py, 0.)
    dates = prices.index[left:right]
    target, pair_abs, meta = {}, {}, []
    for style in styles:
        pick = selected['is_' + style].to_numpy(bool)
        a, b = sx[:, pick], sy[:, pick]
        denominator = (np.abs(a) + np.abs(b)).sum(axis=1)
        a = np.divide(a, denominator[:, None], out=np.zeros_like(a), where=denominator[:, None] > 0)
        b = np.divide(b, denominator[:, None], out=np.zeros_like(b), where=denominator[:, None] > 0)
        w = np.zeros((len(dates), prices.shape[1]))
        absolute = np.zeros_like(w)
        rows = np.arange(len(dates))[:, None]
        np.add.at(w, (rows, xi[pick][None, :]), a)
        np.add.at(w, (rows, yi[pick][None, :]), b)
        np.add.at(absolute, (rows, xi[pick][None, :]), np.abs(a))
        np.add.at(absolute, (rows, yi[pick][None, :]), np.abs(b))
        target[style], pair_abs[style] = w, absolute
        meta.append(dict(cutoff=str(pd.Timestamp(cutoff).date()), style=style, pairs=int(pick.sum()),
                         return_start=str(dates[0].date()), return_end=str(dates[-1].date()),
                         signal_start=str(prices.index[left-1].date()), signal_end=str(prices.index[right-2].date()),
                         mean_active_pairs=float(np.count_nonzero(np.abs(a)+np.abs(b), axis=1).mean()),
                         mean_net_gross_instruction=float(np.abs(w).sum(axis=1).mean())))
    return dates, target, pair_abs, meta


def execute(raw, targets, pair_abs, fee=0., budget_mode='stockgross'):
    """Actual quantity/cash ledger, one more raw-price row than targets.

An instruction for row t executes at raw[t] and earns through raw[t+1].
Missing execution quotes freeze quantities, with last-observation marks.
Fee cases have independent NAV paths; fees are debited from cash. Terminal
liquidation occurs only for assets having an observed final quote.
"""
    raw, targets, pair_abs = map(lambda a: np.asarray(a, float), (raw, targets, pair_abs))
    assert targets.shape == pair_abs.shape == (len(raw)-1, raw.shape[1])
    assert np.isfinite(targets).all() and np.isfinite(pair_abs).all()
    assert budget_mode in ('stockgross', 'pairgross') and fee >= 0
    marks = pd.DataFrame(raw).ffill().to_numpy()
    shares = np.zeros(raw.shape[1]); cash = 1.
    rows, weights, held, closed = [], [], [], []
    max_frozen_change = 0.; max_identity_error = 0.
    for t in range(len(targets)):
        previous = shares.copy()
        p0, p1 = marks[t], marks[t+1]
        nav0 = cash + np.nansum(shares * p0)
        if not np.isfinite(nav0) or nav0 <= 0:
            raise ValueError('Nonpositive or nonfinite starting NAV')
        tradable = np.isfinite(raw[t]) & (raw[t] > 0)
        frozen_value = np.where(tradable, 0., np.nan_to_num(shares * p0))
        frozen_gross = float(np.abs(frozen_value).sum() / nav0)
        available = max(0., 1-frozen_gross)
        denominator = float((np.abs(targets[t]) if budget_mode == 'stockgross' else pair_abs[t])[tradable].sum())
        scale = available / denominator if denominator > 1e-15 else 0.
        desired = np.divide(targets[t] * scale * nav0, p0, out=np.zeros_like(shares), where=np.isfinite(p0) & (p0 > 0))
        shares = np.where(tradable, desired, shares)
        delta = shares - previous
        traded = float(np.nansum(np.abs(delta) * p0))
        paid = fee * traded
        cash -= float(np.nansum(delta * p0)) + paid
        w = np.nan_to_num(shares * p0) / nav0
        holding = shares.copy()
        mark_pnl = float(np.nansum(shares * (p1-p0)))
        terminal_notional = 0.
        if t == len(targets)-1:
            terminal_observed = np.isfinite(raw[t+1]) & (raw[t+1] > 0)
            terminal_notional = float(np.sum(np.abs(shares[terminal_observed]) * p1[terminal_observed]))
            cash += float(np.sum(shares[terminal_observed] * p1[terminal_observed])) - fee*terminal_notional
            shares[terminal_observed] = 0.
            paid += fee*terminal_notional
        nav1 = cash + float(np.nansum(shares * p1))
        identity_error = abs(nav1 - (nav0 + mark_pnl - paid))
        max_identity_error = max(max_identity_error, identity_error)
        if (~tradable).any():
            max_frozen_change = max(max_frozen_change, float(np.max(np.abs(holding[~tradable]-previous[~tradable]))))
        gross = float(np.abs(w).sum())
        rows.append(dict(returns=nav1/nav0-1, nav=nav1, nav_start=nav0, cash_end=cash,
                         turnover=(traded+terminal_notional)/nav0, transaction_fee=paid,
                         mark_pnl=mark_pnl, terminal_notional=terminal_notional,
                         gross_exposure=gross, net_exposure=float(w.sum()),
                         pair_gross_instruction=float(pair_abs[t, tradable].sum()*scale+frozen_gross),
                         hhi=float(((w/gross)**2).sum()) if gross else 0.,
                         frozen_gross=frozen_gross, frozen_assets=int(np.count_nonzero(frozen_value)),
                         frozen_budget_overrun=max(0.,frozen_gross-1), active_stock_count=int(np.count_nonzero(w)),
                         ledger_identity_error=identity_error))
        weights.append(w); held.append(holding); closed.append(shares.copy())
    assert max_identity_error < 1e-10 and max_frozen_change == 0.
    daily = pd.DataFrame(rows)
    assert np.isfinite(daily.to_numpy()).all()
    return dict(daily=daily, weights=np.asarray(weights), quantities=np.asarray(held),
                end_quantities=np.asarray(closed), marks=marks,
                audit=dict(max_cash_identity_error=max_identity_error, max_frozen_quantity_change=max_frozen_change,
                           max_gross=float(daily.gross_exposure.max()),
                           frozen_budget_overrun_days=int((daily.frozen_budget_overrun>1e-10).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection-dir', type=Path, default=ROOT/'data/selection')
    ap.add_argument('--wait-for-selection', action='store_true')
    args = ap.parse_args()
    if hashlib.sha256(PRICE.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise ValueError('Unexpected original price file')
    prices = pd.read_parquet(PRICE).sort_index()
    prices = prices.astype(float).where(prices > 0)
    assert not prices.index.has_duplicates
    dates = prices.loc['2017-01-03':'2023-05-31'].index
    assert len(dates) == 1613
    cutoffs = prices.loc['2016-12-01':'2023-05-31'].groupby(lambda x: (x.year,x.month)).tail(1).index
    styles = STYLES + tuple(s+'_clean' for s in STYLES)
    targets = {s: np.zeros((len(dates),prices.shape[1])) for s in styles}
    absolute = {s: np.zeros_like(targets[s]) for s in styles}
    covered = np.zeros(len(dates),int); records=[]
    for cutoff, end in zip(cutoffs[:-1],cutoffs[1:]):
        path = args.selection_dir/(cutoff.strftime('%Y%m%d')+'.parquet')
        while not path.exists():
            if not args.wait_for_selection: raise FileNotFoundError(path)
            time.sleep(5)
        selected = pd.read_parquet(path)
        selected = selected.rename(columns={'ASSET_1':'x','ASSET_2':'y'})
        dd, tt, aa, mm = month_targets(prices,selected,cutoff,end,styles=styles)
        loc=dates.get_indexer(dd); assert (loc>=0).all()
        for s in styles: targets[s][loc]=tt[s]; absolute[s][loc]=aa[s]
        covered[loc]+=1;records.extend(mm)
        print('targets',str(cutoff.date()),len(selected),flush=True)
    assert (covered==1).all(), 'Every return date must be assigned exactly once'
    results=ROOT/'results'; (results/'daily').mkdir(parents=True,exist_ok=True)
    (results/'weights').mkdir(exist_ok=True);(ROOT/'checks').mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(results/'signal_summary.csv',index=False)
    start=prices.index.get_loc(dates[0]);end=prices.index.get_loc(dates[-1])
    raw=prices.iloc[start-1:end+1].to_numpy()
    checks=[]
    for delay in (0,1):
        for style in styles:
            clean = style.endswith('_clean')
            if clean and delay != 1: continue
            tt,aa=targets[style],absolute[style]
            if delay:
                tt=np.vstack([np.zeros((1,tt.shape[1])),tt[:-1]])
                aa=np.vstack([np.zeros((1,aa.shape[1])),aa[:-1]])
            for budget in ('stockgross','pairgross'):
                if clean and budget != 'stockgross': continue
                for bp in (0,5,10):
                    prefix='clean_' if clean else ''
                    name=f'{prefix}{budget}_d{delay}_{style.split("_")[0]}_c{bp}'
                    result=execute(raw,tt,aa,fee=bp/10000,budget_mode=budget)
                    daily=result['daily'];daily.index=dates;daily.index.name='date'
                    daily.to_csv(results/'daily'/(name+'.csv'))
                    if delay==1 and budget=='stockgross' and bp==5 and not clean:
                        pd.DataFrame(result['weights'],index=dates,columns=prices.columns).to_parquet(results/'weights'/(name+'.parquet'))
                    checks.append(dict(path=name,rows=len(daily),unique_dates=bool(daily.index.is_unique),**result['audit']))
                    sr=daily.returns.mean()/daily.returns.std()*np.sqrt(252)
                    print(name,'cumulative',daily.nav.iloc[-1]-1,'Sharpe',sr,flush=True)
    (ROOT/'checks/execution.json').write_text(json.dumps(dict(price_sha256=EXPECTED_SHA,paths=checks,all_dates_once=True,
         primary='stockgross_d1; one full trading session between information and execution; fees reinvested',
         delayed_instruction='Dollar weights computed at signal close are executed at next close; share ratios can therefore differ at execution.',
         selection_count=len(cutoffs)-1),indent=2))
    print('COMPLETE 45 paths',flush=True)


if __name__=='__main__': main()
