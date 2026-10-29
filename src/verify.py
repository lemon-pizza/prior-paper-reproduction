"""Independent small-number checks for the additional validation execution.

The expected cash flows below are hand-derived. They do not reimplement the
engine's normalization, drift, or cash loop as an alternative oracle. The
target timing test injects known regression outputs to isolate indexing from
the rolling-OLS mathematics covered by the independent regression check.
After these tests, saved results are independently reconciled to raw prices
and stock weights. Run src/backtest.py first; src/verify.py requires all 45 daily paths
and the three primary stock-weight files and never regenerates a strategy.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline_execution_under_test', ROOT / 'src/backtest.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
ATOL = 1e-12


def same(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0, equal_nan=True)


def test_beta_share_single_period():
    # beta=1.5, signal=-2, prices X=100,Y=50 gives raw shares +3,-2.
    # Raw gross $400 -> weights +.75,-.25, unlike beta-dollar +.6,-.4.
    # With NAV=1: shares .0075,-.005 and cash=.5 before fees.
    # Next prices 102,49 give +.015+.005=.02 mark PNL.
    # Entry $1 plus observed terminal gross $1.01 => 2.01 turnover.
    raw = [[100., 50.], [102., 49.]]
    weights = [[.75, -.25]]
    out = engine.execute(raw, weights, [[.75, .25]], fee=.0005)
    d = out['daily'].iloc[0]
    same(out['quantities'], [[.0075, -.005]])
    same(out['weights'], weights)
    same(d.mark_pnl, .02)
    same(d.terminal_notional, 1.01)
    same(d.turnover, 2.01)
    same(d.transaction_fee, .001005)
    same(d.nav, 1.018995)
    same(d.returns, .018995)
    same(d.cash_end, d.nav)
    same(out['end_quantities'], [[0., 0.]])
    return dict(expected_nav=1.018995, expected_total_turnover=2.01)


def test_stock_netting_and_two_gross_definitions():
    # Two pair legs (-.25 A,+.25 B),(-.25 B,+.25 C) cancel B entirely.
    # pair-gross=1 leaves stock gross .5; stock-gross=1 doubles A and C.
    raw = [[100., 100., 100.], [100., 100., 110.]]
    target = [[-.25, 0., .25]]
    pair_abs = [[.25, .5, .25]]
    expected = {'pairgross': (.5, .025, 1.025, 1.0244875),
                'stockgross': (1., .05, 2.05, 1.048975)}
    for mode, (gross, pnl, turn, nav) in expected.items():
        out = engine.execute(raw, target, pair_abs, fee=.0005, budget_mode=mode)
        d = out['daily'].iloc[0]
        same(d.gross_exposure, gross)
        same(d.mark_pnl, pnl)
        same(d.turnover, turn)
        same(d.nav, nav)
        same(out['quantities'][0, 1], 0.)
        same(d.transaction_fee, turn * .0005)
    return dict(pairgross_final_nav=1.0244875, stockgross_final_nav=1.048975,
                canceled_middle_stock_quantity=0.)


def test_frozen_quantity_reserve_and_reappearance():
    # Day 1 holds 0.005 A and -0.005 B. A's next quote is absent: mark 100.
    # At day 2 start A is frozen at $0.5, leaving only $0.5 active gross.
    # New B/C pair splits that reserve into -.25,+.25. A's return at 120
    # realizes the full stale-to-observed +$0.10, with unchanged 0.005 shares.
    raw = [[100., 100., 100.], [np.nan, 100., 100.], [120., 100., 100.]]
    target = [[.5, -.5, 0.], [0., -.5, .5]]
    out = engine.execute(raw, target, np.abs(target))
    same(out['marks'][:, 0], [100., 100., 120.])
    same(out['quantities'], [[.005, -.005, 0.], [.005, -.0025, .0025]])
    same(out['weights'][1], [.5, -.25, .25])
    same(out['daily'].frozen_gross, [0., .5])
    same(out['daily'].mark_pnl, [0., .1])
    same(out['daily'].nav, [1., 1.1])
    same(out['daily'].turnover, [1., 1.6])
    same(out['daily'].gross_exposure, [1., 1.])
    same(out['end_quantities'][-1], [0., 0., 0.])
    return dict(frozen_quantity=.005, active_gross_after_reserve=.5,
                reappearance_pnl=.1, final_nav=1.1)


def test_terminal_missing_quote_has_no_fictitious_trade():
    # A remains unobservable at the final date; B's short is closed at 110.
    # Entry $1 and terminal short cover $.55, not liquidation of stale A $.5.
    out = engine.execute([[100., 100.], [np.nan, 110.]],
                         [[.5, -.5]], [[.5, .5]], fee=.0005)
    d = out['daily'].iloc[0]
    same(d.mark_pnl, -.05)
    same(d.terminal_notional, .55)
    same(d.turnover, 1.55)
    same(d.transaction_fee, .000775)
    same(d.cash_end, .449225)
    same(d.nav, .949225)
    same(out['end_quantities'][-1], [.005, 0.])
    same(d.nav, d.cash_end + .005 * 100.)
    return dict(stale_position_final_quantity=.005, terminal_notional=.55,
                final_cash=.449225, final_nav=.949225)


def test_frozen_exposure_can_exceed_remaining_equity():
    # The short B loses $.75 while long A $.5 cannot be sold. Equity is $.25.
    # Frozen A is 200% of NAV, so active budget is zero; the engine may not
    # fabricate an A sale to force gross back to 100%.
    raw = [[100., 100.], [np.nan, 250.], [np.nan, 250.]]
    target = [[.5, -.5], [0., -1.]]
    out = engine.execute(raw, target, np.abs(target))
    same(out['daily'].nav, [.25, .25])
    same(out['quantities'][1], [.005, 0.])
    same(out['weights'][1], [2., 0.])
    same(out['daily'].iloc[1].frozen_gross, 2.)
    same(out['daily'].iloc[1].frozen_budget_overrun, 1.)
    same(out['daily'].iloc[1].cash_end, -.25)
    assert out['audit']['frozen_budget_overrun_days'] == 1
    return dict(unavoidable_frozen_gross=2., active_quantity=0., final_nav=.25)


def test_cost_paths_reinvest_their_own_nav():
    # One long stock, 100 -> 110 -> 121. Initial entry fee f reduces NAV to
    # 1.1-f. Rebalancing then sells exactly f dollars, costing f^2. The final
    # stock value is 1.21-1.1f, so terminal NAV = 1.21-2.31f+0.1f^2.
    navs = {}
    for f in [0., .0005, .001]:
        out = engine.execute([[100.], [110.], [121.]], [[1.], [1.]], [[1.], [1.]], fee=f)
        same(out['daily'].iloc[0].nav, 1.1-f)
        same(out['quantities'][1, 0], (1.1-f)/110.)
        same(out['daily'].iloc[1].nav_start, 1.1-f)
        same(out['daily'].iloc[1].transaction_fee, f*f+f*(1.21-1.1*f))
        same(out['daily'].iloc[1].nav, 1.21-2.31*f+.1*f*f)
        same(np.prod(1+out['daily'].returns), out['daily'].iloc[-1].nav)
        navs[str(f)] = float(out['daily'].iloc[-1].nav)
    assert navs['0.0'] > navs['0.0005'] > navs['0.001']
    return dict(final_nav_by_fee=navs)


def test_no_signal_stays_cash_and_first_missing_quote_is_not_backfilled():
    out = engine.execute([[np.nan, 100.], [50., 100.], [55., 100.]],
                         [[1., 0.], [0., 0.]], [[1., 0.], [0., 0.]], fee=.001)
    same(out['daily'].nav, [1., 1.])
    same(out['daily'].transaction_fee, [0., 0.])
    same(out['quantities'], np.zeros((2, 2)))
    assert np.isnan(out['marks'][0, 0])
    return dict(final_nav=1., fees=0., never_observed_price_not_backfilled=True)


def test_close_to_close_execution_and_full_session_delay():
    # Known instructions .5A/-.5B at first available close, then zero.
    # A jumps 100 -> 200 -> 300 -> 300. Same-close execution earns 50% on
    # the first jump. Delaying that dollar instruction one row earns 25%
    # only on the second jump; no first-period position may appear.
    raw = [[100., 100.], [200., 100.], [300., 100.], [300., 100.]]
    instruction = np.array([[.5, -.5], [0., 0.], [0., 0.]])
    lagged = np.vstack([np.zeros((1, 2)), instruction[:-1]])
    immediate = engine.execute(raw, instruction, np.abs(instruction))
    delayed = engine.execute(raw, lagged, np.abs(lagged))
    same(immediate['daily'].returns, [.5, 0., 0.])
    same(delayed['daily'].returns, [0., .25, 0.])
    same(delayed['quantities'][0], [0., 0.])
    same(delayed['quantities'][1], [.0025, -.005])
    return dict(delay0_returns=[.5, 0., 0.], delay1_returns=[0., .25, 0.],
                delayed_order_is_dollar_weight_not_fixed_share_ratio=True)


def test_month_target_signal_beta_price_lag_and_boundary():
    # Inject deterministic OLS outputs so this test isolates the month runner's
    # signal/beta/price indices. The mathematical OLS has separate sklearn tests.
    dates = pd.bdate_range('2018-01-01', periods=560)
    t = np.arange(560)
    prices = pd.DataFrame({'A': 100.+t, 'B': 50.+.2*t,
                           'C': 80.+.5*t, 'D': 60.+.1*t}, index=dates)
    selected = pd.DataFrame({'x': ['A', 'C'], 'y': ['B', 'D'],
                             'is_M': [True, True], 'is_B': [True, True],
                             'is_U': [True, True]})
    beta = np.column_stack([.5+.001*t, -1.5+.0001*t])
    residual = np.column_stack([((t*7) % 23).astype(float), ((t*11) % 29).astype(float)])
    residual[540] = [80., -60.]
    residual[541] = [-90., 100.]
    original = engine.load_rolling

    def invoke(cutoff_idx, end_idx, input_prices):
        left = cutoff_idx+1
        warm = max(0, left-504-22)
        def known_rolling(x, y):
            assert x.shape == y.shape
            return beta[warm:warm+len(x)].copy(), residual[warm:warm+len(x)].copy()
        engine.load_rolling = lambda: known_rolling
        return engine.month_targets(input_prices, selected, dates[cutoff_idx], dates[end_idx])

    try:
        dd, target, absolute, meta = invoke(540, 544, prices)
        # Hand oracle: Q uses residual observations 521..540; beta/prices 540.
        # np.quantile/median here is independent of the pandas rolling code.
        window = residual[521:541]
        q = (residual[540]-np.median(window, axis=0)) / (np.quantile(window,.25,axis=0)-np.quantile(window,.75,axis=0))
        signal = np.sign(q)*np.round(np.abs(q))
        assert np.all(signal != 0)
        # Two disjoint beta-share pairs; exact dollar legs before normalization.
        unscaled = np.array([-signal[0]*beta[540,0]*prices.A.iloc[540],
                             signal[0]*prices.B.iloc[540],
                             -signal[1]*beta[540,1]*prices.C.iloc[540],
                             signal[1]*prices.D.iloc[540]])
        expected = unscaled / np.abs(unscaled).sum()
        for style in engine.STYLES:
            same(target[style][0], expected)
            same(absolute[style][0], np.abs(expected))
        assert dd[0] == dates[541] and dd[-1] == dates[544]
        assert all(m['signal_start'] == str(dates[540].date()) for m in meta)
        changed = prices.copy()
        changed.iloc[541:] *= 7.
        _, changed_target, _, _ = invoke(540, 544, changed)
        same(changed_target['M'][0], expected)
        dd2, _, _, _ = invoke(544, 548, prices)
        combined = dd.append(dd2)
        assert combined.is_unique and combined.equals(dates[541:549])
    finally:
        engine.load_rolling = original
    return dict(first_signal_date=str(dates[540].date()),
                first_return_date=str(dates[541].date()),
                q=q.tolist(), signal=signal.tolist(), expected_weights=expected.tolist(),
                future_price_perturbation_does_not_change_first_instruction=True,
                adjacent_month_returns_cover_once=True)


def test_actual_selected_pairs_future_price_causality():
    # This follows the real raw-price -> rolling504 -> Q20 -> target route.
    # It complements the injected regression outputs above; no mock is used.
    selection_path = ROOT/'data/selection/20161230.parquet'
    selected = pd.read_parquet(selection_path).rename(columns={'ASSET_1':'x', 'ASSET_2':'y'})
    sample = pd.concat([selected[selected['is_'+s]].head(3) for s in engine.STYLES])
    sample = sample.drop_duplicates(['x', 'y']).reset_index(drop=True)
    assert 5 <= len(sample) <= 10
    prices = pd.read_parquet(engine.PRICE).sort_index().astype(float)
    prices = prices.where(prices > 0)
    dates, before, before_abs, _ = engine.month_targets(prices, sample, '2016-12-30', '2017-01-31')
    evaluation_date = dates[9]
    changed = prices.copy()
    changed.loc[evaluation_date:] *= 1.7
    dd, after, after_abs, _ = engine.month_targets(changed, sample, '2016-12-30', '2017-01-31')
    assert dates.equals(dd)
    unchanged_rows = dates <= evaluation_date
    changed_rows = dates > evaluation_date
    max_past_error = 0.
    max_later_change = 0.
    active = {}
    for style in engine.STYLES:
        same(before[style][unchanged_rows], after[style][unchanged_rows])
        same(before_abs[style][unchanged_rows], after_abs[style][unchanged_rows])
        max_past_error = max(max_past_error, float(np.max(np.abs(before[style][unchanged_rows]-after[style][unchanged_rows]))))
        max_later_change = max(max_later_change, float(np.max(np.abs(before[style][changed_rows]-after[style][changed_rows]))))
        active[style] = float(np.abs(before[style][9]).sum())
    assert any(value > 0 for value in active.values()), 'Causality check must include a live target'
    assert max_later_change > 1e-8, 'Perturbation must affect a later instruction'
    return dict(selection_cache=str(selection_path),
                selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                actual_selected_pairs=sample[['x', 'y']].to_dict('records'),
                return_date=str(evaluation_date.date()),
                input_price_multiplier=1.7,
                changed_prices_start=str(evaluation_date.date()),
                tested_unchanged_target_rows=int(unchanged_rows.sum()),
                active_stock_gross_by_style=active,
                max_past_or_current_target_error=max_past_error,
                max_later_target_change=max_later_change,
                actual_rolling_implementation_used=True)


TESTS = [test_beta_share_single_period, test_stock_netting_and_two_gross_definitions,
         test_frozen_quantity_reserve_and_reappearance,
         test_terminal_missing_quote_has_no_fictitious_trade,
         test_frozen_exposure_can_exceed_remaining_equity,
         test_cost_paths_reinvest_their_own_nav,
         test_no_signal_stays_cash_and_first_missing_quote_is_not_backfilled,
         test_close_to_close_execution_and_full_session_delay,
         test_month_target_signal_beta_price_lag_and_boundary,
         test_actual_selected_pairs_future_price_causality]


def require_actual_audit_inputs():
    """Fail explicitly instead of silently skipping the stored-path audit."""
    expected_daily = {
        f'{budget}_d{delay}_{style}_c{cost}.csv'
        for budget in ('stockgross', 'pairgross') for delay in (0, 1)
        for style in ('M', 'B', 'U') for cost in (0, 5, 10)
    } | {f'clean_stockgross_d1_{style}_c{cost}.csv'
         for style in ('M', 'B', 'U') for cost in (0, 5, 10)}
    expected_weights = {f'stockgross_d1_{style}_c5.parquet' for style in ('M', 'B', 'U')}
    required = [engine.PRICE, ROOT/'data/selection/20161230.parquet']
    required += [ROOT/'results/daily'/name for name in sorted(expected_daily)]
    required += [ROOT/'results/weights'/name for name in sorted(expected_weights)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Required audit inputs are missing. Complete selection generation, '
            'then run src/backtest.py before src/verify.py. '
            'Missing files: ' + ', '.join(missing))
    found_daily = {p.name for p in (ROOT/'results/daily').glob('*.csv')}
    found_weights = {p.name for p in (ROOT/'results/weights').glob('*.parquet')}
    if found_daily != expected_daily or found_weights != expected_weights:
        raise ValueError('Expected exactly 45 daily CSV files and 3 primary weight Parquet files. '
                         f'Unexpected daily files: {sorted(found_daily-expected_daily)}; '
                         f'unexpected weight files: {sorted(found_weights-expected_weights)}')


def audit_actual_paths(output=None):
    """Audit persisted positions, prices and cash without calling the engine.

Recorded start-of-day NAV converts stored dollar weights back into quantities.
That NAV is then checked against an independently accumulated cash/share
ledger. Fees come from changes in reconstructed quantities, not from reported
turnover. Terminal transactions use only actual observed final prices.
"""
    require_actual_audit_inputs()
    output = Path(output) if output else ROOT/'checks/actual_path_audit.json'
    price_path = engine.PRICE
    prices = pd.read_parquet(price_path).sort_index().astype(float)
    prices = prices.where(prices > 0)
    prices.index = prices.index.as_unit('ns')
    dates = prices.loc['2017-01-03':'2023-05-31'].index
    assert len(dates) == 1613 and dates.is_unique
    first, last = prices.index.get_loc(dates[0]), prices.index.get_loc(dates[-1])
    raw = prices.iloc[first-1:last+1].to_numpy()
    marks = pd.DataFrame(raw).ffill().to_numpy()
    names = prices.columns
    file_checks, reports = [], []
    for path in sorted((ROOT/'results/daily').glob('*.csv')):
        d = pd.read_csv(path, index_col=0, parse_dates=True)
        ok = len(d) == 1613 and d.index.is_unique and d.index.equals(dates) and np.isfinite(d.to_numpy()).all()
        file_checks.append(dict(file=path.name, rows=len(d), unique_dates=bool(d.index.is_unique),
                                exact_expected_dates=bool(d.index.equals(dates)),
                                all_numeric_values_finite=bool(np.isfinite(d.to_numpy()).all()), passed=bool(ok)))
    assert len(file_checks) == 45 and all(c['passed'] for c in file_checks), file_checks
    tolerance = 1e-10
    for style in ('M', 'B', 'U'):
        name = f'stockgross_d1_{style}_c5'
        daily_path = ROOT/'results/daily'/f'{name}.csv'
        weight_path = ROOT/'results/weights'/f'{name}.parquet'
        d = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        weights = pd.read_parquet(weight_path)
        weights.index = weights.index.as_unit('ns')
        assert weights.index.equals(dates) and weights.columns.equals(names), name
        w = weights.to_numpy()
        assert np.isfinite(w).all(), name
        cash, total_fees = 1., 0.
        previous = np.zeros(len(names))
        errors = {k: 0. for k in ('start_nav', 'end_nav', 'return', 'cash', 'mark_pnl', 'fee',
                                  'turnover', 'terminal_notional', 'frozen_gross', 'frozen_quantity',
                                  'gross', 'net', 'compound_nav')}
        independent_rows = []
        for t in range(len(d)):
            r = d.iloc[t]
            p0, p1 = marks[t], marks[t+1]
            quoted = np.isfinite(raw[t]) & (raw[t] > 0)
            independent_start = cash + np.nansum(previous*p0)
            quantity = np.divide(w[t]*r.nav_start, p0, out=np.zeros(len(names)),
                                 where=np.isfinite(p0) & (p0 > 0))
            delta = quantity-previous
            traded = np.nansum(np.abs(delta)*p0)
            fee = .0005*traded
            cash -= np.nansum(delta*p0)+fee
            pnl = np.nansum(quantity*(p1-p0))
            terminal = 0.
            end_quantity = quantity.copy()
            if t == len(d)-1:
                observed = np.isfinite(raw[t+1]) & (raw[t+1] > 0)
                terminal = np.sum(np.abs(quantity[observed])*p1[observed])
                terminal_fee = .0005*terminal
                cash += np.sum(quantity[observed]*p1[observed])-terminal_fee
                fee += terminal_fee
                end_quantity[observed] = 0.
            nav = cash + np.nansum(end_quantity*p1)
            frozen = np.nansum(np.abs(previous[~quoted]*p0[~quoted]))/r.nav_start
            actual = dict(start_nav=independent_start, end_nav=nav, return_=nav/independent_start-1,
                          cash=cash, mark_pnl=pnl, fee=fee, turnover=(traded+terminal)/r.nav_start,
                          terminal_notional=terminal, frozen_gross=frozen,
                          gross=np.abs(w[t]).sum(), net=w[t].sum())
            actual['return'] = actual.pop('return_')
            expected = dict(start_nav=r.nav_start, end_nav=r.nav, return_=r.returns, cash=r.cash_end,
                            mark_pnl=r.mark_pnl, fee=r.transaction_fee, turnover=r.turnover,
                            terminal_notional=r.terminal_notional, frozen_gross=r.frozen_gross,
                            gross=r.gross_exposure, net=r.net_exposure)
            expected['return'] = expected.pop('return_')
            for k in actual:
                errors[k] = max(errors[k], abs(float(actual[k]-expected[k])))
            if (~quoted).any():
                errors['frozen_quantity'] = max(errors['frozen_quantity'], float(np.max(np.abs(delta[~quoted]))))
            total_fees += fee
            previous = end_quantity
            independent_rows.append(dict(date=str(dates[t].date()), nav=float(nav), pnl=float(pnl),
                                         fee=float(fee), turnover=float(actual['turnover'])))
        errors['compound_nav'] = float(np.max(np.abs(np.cumprod(1+d.returns.to_numpy())-d.nav.to_numpy())))
        stale = ~(np.isfinite(raw[-1]) & (raw[-1] > 0))
        held = np.flatnonzero(stale & (np.abs(previous) > 1e-14))
        positions = []
        for j in held:
            history = prices.iloc[:last+1, j].dropna()
            positions.append(dict(asset=str(names[j]), quantity_per_initial_NAV=float(previous[j]),
                                  last_observed_price=float(marks[-1, j]),
                                  last_observed_date=str(history.index[-1].date()),
                                  marked_value_per_initial_NAV=float(previous[j]*marks[-1, j]),
                                  absolute_value_fraction_of_final_NAV=float(abs(previous[j]*marks[-1, j])/nav)))
        gross = np.abs(w).sum(axis=1)
        exposed, frozen_days = gross > 1e-12, d.frozen_gross > 1e-12
        report = dict(scenario=name, passed=bool(max(errors.values()) < tolerance), max_absolute_errors=errors,
                      rows=len(d), weight_shape=list(weights.shape),
                      gross_1_days=int(np.sum(np.abs(gross-1) < 1e-10)),
                      gross_0_days=int(np.sum(gross < 1e-10)),
                      other_gross_days=int(np.sum((gross >= 1e-10) & (np.abs(gross-1) >= 1e-10))),
                      gross_min=float(gross.min()), gross_max=float(gross.max()),
                      active_day_gross_mean=float(gross[exposed].mean()),
                      net_exposure_mean=float(w.sum(axis=1).mean()),
                      net_exposure_min=float(w.sum(axis=1).min()), net_exposure_max=float(w.sum(axis=1).max()),
                      max_single_stock_absolute_weight=float(np.abs(w).max()),
                      mean_daily_turnover=float(d.turnover.mean()), fees_paid_per_initial_NAV=float(total_fees),
                      final_nav=float(nav), cumulative_return=float(nav-1), final_cash=float(cash),
                      frozen_days=int(frozen_days.sum()),
                      first_frozen_return_date=str(d.index[frozen_days][0].date()) if frozen_days.any() else None,
                      final_unquoted_holdings_count=len(held),
                      final_unquoted_gross_value_per_initial_NAV=float(np.nansum(np.abs(previous[stale])*marks[-1, stale])),
                      final_unquoted_gross_fraction_of_final_NAV=float(np.nansum(np.abs(previous[stale])*marks[-1, stale])/nav),
                      final_unquoted_net_value_per_initial_NAV=float(np.nansum(previous[stale]*marks[-1, stale])),
                      final_observed_quantity_max=float(np.abs(previous[~stale]).max()),
                      final_terminal_notional=float(d.terminal_notional.iloc[-1]), final_unquoted_positions=positions,
                      illustrative_rows=[independent_rows[0], independent_rows[1], independent_rows[-1]],
                      daily_sha256=hashlib.sha256(daily_path.read_bytes()).hexdigest(),
                      weights_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest())
        reports.append(report)
    result = dict(passed=all(r['passed'] for r in reports),
                  price_sha256=hashlib.sha256(price_path.read_bytes()).hexdigest(),
                  scope='45 stored daily files date/finite inventory; independent raw-price plus saved stock-weight quantity/cash replay for three primary stockgross_d1 cost5 paths. No engine functions called; no model rerun or modification.',
                  absolute_tolerance=tolerance,
                  timestamp_unit_normalization='Parquet microseconds and CSV nanoseconds converted to nanoseconds before exact date comparison. Actual dates are unchanged.',
                  expected_return_dates=dict(start=str(dates[0].date()), end=str(dates[-1].date()), count=len(dates)),
                  daily_file_count=len(file_checks),
                  primary_weight_file_count=len(list((ROOT/'results/weights').glob('*.parquet'))),
                  daily_file_checks=file_checks, primary_path_checks=reports,
                  remaining_accounting_scope_limits=[
                      '1-NAV gross is measured before execution fees; immediate post-fee gross can exceed 1 by the fee fraction.',
                      'Unquoted residual holdings remain at the last observed price, without an assured realizable exit price or settlement.',
                      'One-session delay applies to dollar weights computed at the signal close; subsequent execution does not preserve the exact signal-close beta-share quantity ratio.',
                      'Modeled stock-net turnover fees exclude market impact, bid-ask execution uncertainty, financing and short-borrow costs.',
                      'Recorded accounting validity does not establish statistical superiority or raw-price adjustment correctness.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    if not result['passed']:
        raise AssertionError(f'Actual stored-path accounting audit failed: {output}')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'checks/execution_independent.json')
    args = ap.parse_args()
    require_actual_audit_inputs()
    records = []
    for test in TESTS:
        try:
            detail = test()
            records.append(dict(test=test.__name__, passed=True, detail=detail))
            print('PASS', test.__name__, flush=True)
        except Exception as exc:
            records.append(dict(test=test.__name__, passed=False, error=repr(exc)))
            print('FAIL', test.__name__, repr(exc), flush=True)
    result = dict(passed=all(r['passed'] for r in records),
                  engine_sha256=hashlib.sha256((ROOT/'src/backtest.py').read_bytes()).hexdigest(),
                  check_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tolerance=ATOL, independent_checks=len(records),
                  independent_synthetic_checks=len(records)-1, real_data_causality_checks=1, tests=records,
                  scope='Small hand-derived ledgers, injected-output target timing, and one real-data full-path causality perturbation; no performance sweep or selection regeneration.',
                  limitations=['Signal outputs are injected only for indexing isolation; independent actual-data sklearn regression checks are recorded separately.',
                               'Delay shifts precomputed dollar weights. It does not preserve the signal-close share ratio at the later execution price.',
                               'A full-session-delayed order is executed at the next close; same-close execution remains an explicitly idealized diagnostic.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
    actual = audit_actual_paths()
    print(f"PASS actual_path_audit: {actual['daily_file_count']} daily files, "
          f"{actual['primary_weight_file_count']} independently reconciled primary paths", flush=True)


if __name__ == '__main__':
    main()
