"""Audit saved fixed-model results without rerunning the historical strategy.

Historical checks use independent quantity/cash, least-squares and statistics
calculations. Only the small terminal-quote regression case imports the engine.
Writes audit/verification.json, including a failed report if a check fails.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COSTS = ("gross", "net_fixed", "net_turnover", "net_5bp", "net_10bp")
STYLES = ("MQ", "MZ", "BQ", "BZ")
EXPECTED = {
    *(f"{scenario}_{style}" for scenario in
      ("paper_daily", "equal_pair", "paper_lag2") for style in STYLES),
    *(f"control_{cap}_{mode}" for cap in ("C1", "C2", "C4", "Cinf")
      for mode in ("Q", "Z")),
}
SAMPLE_MONTHS = ("2017-01", "2020-03", "2023-05")


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def error(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape, f"Shape mismatch: {a.shape} != {b.shape}")
    require(np.isfinite(a).all() and np.isfinite(b).all(),
            "Non-finite values in a numerical comparison")
    return float(np.max(np.abs(a-b))) if a.size else 0.0


def check_errors(values, tolerance, label):
    for key, value in values.items():
        require(np.isfinite(value) and value <= tolerance,
                f"{label}: {key} error {value:.12g} exceeds {tolerance:g}")


def moments(returns):
    """Direct scalar accumulation, including initial NAV in the drawdown peak."""
    r = np.asarray(returns, dtype=float)
    require(len(r) > 1 and np.isfinite(r).all(), "Invalid return series")
    n = len(r)
    mean = math.fsum(float(x) for x in r)/n
    sd = math.sqrt(math.fsum((float(x)-mean)**2 for x in r)/(n-1))
    nav = peak = 1.0
    worst = 0.0
    for ret in r:
        nav *= 1.0+float(ret)
        peak = max(peak, nav)
        worst = min(worst, nav/peak-1.0)
    require(sd > 0 and nav > 0, "Statistics require positive variance and NAV")
    return dict(days=n, mean_daily=mean, mean_annual=mean*252,
                volatility=sd*math.sqrt(252), sharpe=mean/sd*math.sqrt(252),
                cumulative=nav-1, cagr=nav**(252/n)-1, max_drawdown=worst)


def inspect_scope(prices):
    scope = {}
    for folder, suffix in (("daily", ".csv"), ("asset_weights", ".parquet"),
                           ("pair_ledger", ".parquet"), ("frozen_ledger", ".parquet")):
        names = {p.stem for p in (ROOT/"results"/folder).glob("*"+suffix)}
        require(names == EXPECTED,
                f"{folder}: missing={sorted(EXPECTED-names)}, extra={sorted(names-EXPECTED)}")
        scope[folder] = len(names)
    months = {str(p) for p in pd.period_range("2017-01", "2023-05", freq="M")}
    for folder in ("features", "selection"):
        names = {p.stem for p in (ROOT/"data"/folder).glob("*.parquet")}
        require(names == months, f"{folder}: expected exactly 77 monthly Parquet files")
        scope[folder+"_months"] = len(names)
    require(prices.index.is_unique and prices.index.is_monotonic_increasing,
            "Price dates must be unique and chronological")
    require(prices.columns.is_unique, "Price asset columns must be unique")
    dates = prices.loc["2017-01-01":"2023-05-31"].index
    require(len(dates) == 1613, "Expected 1,613 evaluation trading dates")
    scope.update(primary=4, auxiliary=16, phase_base=8, phase_new=12,
                 phase_all=20, days=len(dates))
    return scope, dates


def verify_ledger(name, prices, expected_dates):
    daily = pd.read_csv(ROOT/"results/daily"/(name+".csv"),
                        index_col="date", parse_dates=True)
    weights = pd.read_parquet(ROOT/"results/asset_weights"/(name+".parquet"))
    # CSV and Parquet may deserialize the same dates at ns/us precision.
    expected_values = expected_dates.to_numpy(dtype="datetime64[ns]")
    require(np.array_equal(daily.index.to_numpy(dtype="datetime64[ns]"), expected_values),
            f"{name}: unexpected daily dates")
    require(np.array_equal(weights.index.to_numpy(dtype="datetime64[ns]"), expected_values)
            and weights.columns.equals(prices.columns),
            f"{name}: weight axes do not match prices/daily results")
    loc = prices.index.get_indexer(expected_dates)
    require(loc[0] > 0 and np.all(np.diff(loc) == 1), f"{name}: invalid evaluation window")
    raw = prices.iloc[loc[0]-1:loc[-1]+1].to_numpy(dtype=float)
    marks = pd.DataFrame(raw).ffill().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.nan_to_num(marks[1:]/marks[:-1]-1, nan=0., posinf=0., neginf=0.)
    w = weights.to_numpy(dtype=float)
    require(np.isfinite(w).all(), f"{name}: invalid saved weight")
    n, assets = w.shape
    nav = cash = 1.0
    old_quantity = np.zeros(assets)
    reconstructed_return = np.empty(n)
    reconstructed_turnover = np.empty(n)
    reconstructed_cash_weight = np.empty(n)
    cash_identity_error = frozen_quantity_error = 0.0
    samples = []
    for t, date in enumerate(expected_dates):
        p0, p1 = marks[t], marks[t+1]
        valid_mark = np.isfinite(p0) & (p0 > 0)
        require(np.all(np.abs(w[t, ~valid_mark]) < 1e-14),
                f"{name}: position without a valuation price on {date}")
        quantity = np.divide(w[t]*nav, p0, out=np.zeros(assets), where=valid_mark)
        delta = quantity-old_quantity
        quoted = np.isfinite(raw[t]) & (raw[t] > 0)
        if np.any(~quoted):
            frozen_quantity_error = max(frozen_quantity_error,
                                        float(np.max(np.abs(delta[~quoted]))))
        order_dollars = float(np.nansum(np.abs(delta*p0)))
        terminal_dollars = 0.0
        if t == n-1:
            terminal_quoted = np.isfinite(raw[t+1]) & (raw[t+1] > 0)
            terminal_dollars = float(np.nansum(np.abs(quantity[terminal_quoted]*p1[terminal_quoted])))
        reconstructed_turnover[t] = (order_dollars+terminal_dollars)/nav
        cash -= float(np.nansum(delta*p0))
        reconstructed_cash_weight[t] = cash/nav
        end_nav = cash+float(np.nansum(quantity*p1))
        pnl = float(np.nansum(quantity*(p1-p0)))
        cash_identity_error = max(cash_identity_error, abs(end_nav-nav-pnl))
        reconstructed_return[t] = pnl/nav
        require(end_nav > 0 and np.isfinite(end_nav), f"{name}: invalid NAV on {date}")
        if t in (0, n-1):
            samples.append(dict(date=str(date.date()), start_nav=nav,
                                cash_after_orders=cash, pnl=pnl, end_nav=end_nav,
                                order_dollars=order_dollars,
                                terminal_observed_dollars=terminal_dollars))
        old_quantity, nav = quantity, end_nav

    pairs = pd.read_parquet(ROOT/"results/pair_ledger"/(name+".parquet"))
    frozen = pd.read_parquet(ROOT/"results/frozen_ledger"/(name+".parquet"))
    require(not pairs.duplicated(["date", "x", "y"]).any(), f"{name}: duplicate pair rows")
    require(not frozen.duplicated(["date", "asset"]).any(), f"{name}: duplicate frozen rows")
    ti = expected_dates.get_indexer(pairs.date)
    xi = prices.columns.get_indexer(pairs.x)
    yi = prices.columns.get_indexer(pairs.y)
    require(np.all(ti >= 0) and np.all(xi >= 0) and np.all(yi >= 0),
            f"{name}: unknown pair date or asset")
    require((pd.to_datetime(pairs.signal_date) < pd.to_datetime(pairs.date)).all(),
            f"{name}: signal is not prior to return date")
    combined_weights = np.zeros_like(w)
    np.add.at(combined_weights, (ti, xi), pairs.weight_x.to_numpy())
    np.add.at(combined_weights, (ti, yi), pairs.weight_y.to_numpy())
    pair_return = np.zeros(n)
    np.add.at(pair_return, ti, pairs.contribution.to_numpy())
    pair_expected = (pairs.weight_x.to_numpy()*returns[ti, xi]
                     + pairs.weight_y.to_numpy()*returns[ti, yi])
    frozen_return = np.zeros(n)
    frozen_gross = np.zeros(n)
    frozen_contribution_error = 0.0
    if len(frozen):
        fi = expected_dates.get_indexer(frozen.date)
        ai = prices.columns.get_indexer(frozen.asset)
        require(np.all(fi >= 0) and np.all(ai >= 0), f"{name}: unknown frozen date/asset")
        require(not frozen.tradable.astype(bool).any(), f"{name}: tradable frozen row")
        require((~np.isfinite(raw[fi, ai]) | (raw[fi, ai] <= 0)).all(),
                f"{name}: frozen row has a current execution quote")
        np.add.at(combined_weights, (fi, ai), frozen.weight.to_numpy())
        np.add.at(frozen_return, fi, frozen.contribution.to_numpy())
        np.add.at(frozen_gross, fi, np.abs(frozen.weight.to_numpy()))
        frozen_contribution_error = error(frozen.contribution.to_numpy(),
                                          frozen.weight.to_numpy()*returns[fi, ai])
    errors = dict(
        quantity_return=error(reconstructed_return, daily.gross.to_numpy()),
        quantity_turnover=error(reconstructed_turnover, daily.turnover.to_numpy()),
        cash_weight=error(reconstructed_cash_weight, daily.cash_weight.to_numpy()),
        cash_identity=cash_identity_error, frozen_quantity_change=frozen_quantity_error,
        pair_plus_frozen_weights=error(combined_weights, w),
        pair_contribution=error(pairs.contribution.to_numpy(), pair_expected),
        frozen_contribution=frozen_contribution_error,
        pair_daily=error(pair_return, daily.pair_return.to_numpy()),
        frozen_daily=error(frozen_return, daily.frozen_return.to_numpy()),
        total_contribution=error(pair_return+frozen_return, daily.gross.to_numpy()),
        frozen_gross=error(frozen_gross, daily.frozen_gross.to_numpy()),
        stock_return=error((w*returns).sum(axis=1), daily.gross.to_numpy()),
        stock_gross=error(np.abs(w).sum(axis=1), daily.gross_post_net.to_numpy()),
        net_exposure=error(w.sum(axis=1), daily.net_exposure.to_numpy()),
    )
    for column, rate, basis in (
        ("net_fixed", .01/252, (daily.gross_post_net > 0).astype(float)),
        ("net_turnover", .01/252, daily.turnover),
        ("net_5bp", .0005, daily.turnover), ("net_10bp", .001, daily.turnover),
    ):
        errors[column] = error(daily[column].to_numpy(), (daily.gross-rate*basis).to_numpy())
    check_errors(errors, 2e-11, name)
    return dict(scenario=name, days=n, pair_rows=len(pairs), frozen_rows=len(frozen),
                errors=errors, tolerance=2e-11, boundary_cash_examples=samples), daily


def direct_fit(history):
    values = history.to_numpy(dtype=float)
    require(values.shape == (504, 2) and np.isfinite(values).all() and (values > 0).all(),
            "Sample regression requires 504 complete positive price pairs")
    x, y = np.log(values).T
    design = np.column_stack((np.ones(504), x))
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y-design@np.array([alpha, beta])
    change = np.diff(residual)
    adf_x = np.column_stack((np.ones(502), residual[1:-1], change[:-1]))
    adf_y = change[1:]
    coefficients = np.linalg.lstsq(adf_x, adf_y, rcond=None)[0]
    unexplained = adf_y-adf_x@coefficients
    variance = float(unexplained@unexplained)/(len(adf_y)-3)
    inverse = np.linalg.pinv(adf_x)
    standard_error = math.sqrt(variance*float(inverse[1]@inverse[1]))
    quantiles = np.quantile(residual, [.25, .5, .75])
    sigma = math.sqrt(float(residual@residual)/502)
    return dict(alpha=float(alpha), beta=float(beta),
                t_stat=float(coefficients[1]/standard_error),
                q_fit=float((residual[-1]-quantiles[1])/(quantiles[2]-quantiles[0])),
                z_fit=float(residual[-1]/sigma))


def verify_raw_samples(prices):
    records = []
    for month in SAMPLE_MONTHS:
        selected = pd.read_parquet(ROOT/"data/selection"/(month+".parquet"))
        features = pd.read_parquet(ROOT/"data/features"/(month+".parquet"))
        require(set(selected.cap) == {"C1", "C2", "C4", "Cinf", "M", "B"},
                f"{month}: missing selection cap")
        require(not features.duplicated(["date", "x", "y"]).any(),
                f"{month}: duplicate feature rows")
        first = features.loc[features.date == features.date.min()].sort_values(["x", "y"])
        indices = np.linspace(0, len(first)-1, min(6, len(first)), dtype=int)
        for item in first.iloc[indices].itertuples():
            signal = pd.Timestamp(item.signal_date)
            date = pd.Timestamp(item.date)
            require(prices.index[prices.index.get_loc(date)-1] == signal,
                    f"{month} {item.x}/{item.y}: signal-date alignment")
            require(int(item.xi) == prices.columns.get_loc(item.x)
                    and int(item.yi) == prices.columns.get_loc(item.y),
                    f"{month} {item.x}/{item.y}: feature asset indices")
            history = prices.loc[:signal, [item.x, item.y]].tail(504)
            calc = direct_fit(history)
            source = selected.loc[(selected.x == item.x) & (selected.y == item.y)]
            require(len(source) > 0 and (pd.to_datetime(source.formation_cutoff) == signal).all(),
                    f"{month} {item.x}/{item.y}: selection cutoff")
            errors = {"selection_"+key: float(np.max(np.abs(source[key]-calc[key])))
                      for key in ("alpha", "beta", "t_stat")}
            errors.update({"feature_"+key: abs(float(getattr(item, key))-calc[key])
                           for key in ("beta", "q_fit", "z_fit")})
            check_errors(errors, 1e-6, f"{month} {item.x}/{item.y}")
            records.append(dict(month=month, x=item.x, y=item.y,
                                signal_date=str(signal.date()), observations=len(history),
                                independent_values=calc, errors=errors, tolerance=1e-6))
    require(len(records) == 18, "Expected six sampled pairs in each of three months")
    return records


def verify_terminal_quote():
    # This two-day example is the only production execution invoked here.
    from engine import backtest_weights
    targets = np.array([[.6, -.4], [.6, -.4]])
    missing = np.array([[100., 50.], [102., 49.], [104., np.nan]])
    observed = missing.copy()
    observed[-1, 1] = 49.
    actual = backtest_weights(targets, missing)
    full = backtest_weights(targets, observed)
    nav = cash = 1.
    quantity = np.zeros(2)
    gross, turnover = [], []
    marks = pd.DataFrame(missing).ffill().to_numpy()
    for t, target in enumerate(targets):
        new_quantity = target*nav/marks[t]
        orders = new_quantity-quantity
        cash -= float(orders@marks[t])
        end = cash+float(new_quantity@marks[t+1])
        traded = float(np.abs(orders*marks[t]).sum())
        if t == len(targets)-1:
            # Only A has an observed final quote; B stays marked and unliquidated.
            traded += abs(float(new_quantity[0]*marks[t+1, 0]))
        gross.append((end-nav)/nav)
        turnover.append(traded/nav)
        quantity, nav = new_quantity, end
    errors = dict(gross=error(actual["gross"], gross),
                  turnover=error(actual["turnover"], turnover),
                  unchanged_gross=error(actual["gross"], full["gross"]),
                  omitted_unquoted_sale=abs(float(full["turnover"][-1]-actual["turnover"][-1])-.4))
    check_errors(errors, 1e-13, "terminal observed-price regression")
    return dict(days=2, assets=2, errors=errors, tolerance=1e-13,
                missing_quote_turnover=np.asarray(actual["turnover"]).tolist(),
                observed_quote_turnover=np.asarray(full["turnover"]).tolist(),
                omitted_terminal_turnover=.4, avoided_5bp_cost=.4*.0005)


def verify_statistics(frames):
    metrics = pd.read_csv(ROOT/"results/metrics.csv")
    keys = ["scenario", "style", "cost"]
    require(not metrics.duplicated(keys).any(), "Duplicate metric keys")
    expected_keys = {(name.rsplit("_", 1)[0], name.rsplit("_", 1)[1], cost)
                     for name in EXPECTED for cost in COSTS}
    require(set(metrics[keys].itertuples(index=False, name=None)) == expected_keys,
            "Expected exactly 20 scenarios x 5 cost metrics")
    errors = {}
    for row in metrics.itertuples():
        name = row.scenario+"_"+row.style
        calculated = moments(frames[name][row.cost].to_numpy())
        for key, value in calculated.items():
            delta = abs(float(getattr(row, key))-value)
            require(np.isfinite(delta), f"{name}/{row.cost}: non-finite {key}")
            errors[key] = max(errors.get(key, 0.), delta)
    check_errors(errors, 2e-11, "saved performance metrics")
    toy = moments([-.10, .01])
    require(abs(toy["max_drawdown"]+.10) < 1e-14,
            "Initial NAV must participate in drawdown")
    require(abs(toy["cumulative"]-(-.091)) < 1e-14,
            "Cumulative return must compound sequential returns")
    return dict(rows=len(metrics), errors=errors, tolerance=2e-11,
                drawdown_initial_nav_example=toy)


def main():
    report = dict(passed=False, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  historical_strategy_rerun=False,
                  method="Independent saved-weight quantity/cash ledger, raw-price least squares, "
                         "and scalar performance statistics; engine used only for a two-day toy.",
                  limitations="This verifies stored arithmetic and 18 raw-price samples. It does not "
                              "validate economic assumptions, source price accuracy or a full selection replay.",
                  ledger_checks=[])
    try:
        prices = pd.read_parquet(ROOT/"data/prices.parquet.gz").astype(float)
        prices = prices.where(prices > 0)
        report["scope"], dates = inspect_scope(prices)
        frames = {}
        for name in sorted(EXPECTED):
            record, frame = verify_ledger(name, prices, dates)
            report["ledger_checks"].append(record)
            frames[name] = frame
        report["raw_price_samples"] = verify_raw_samples(prices)
        report["terminal_quote_test"] = verify_terminal_quote()
        report["statistics"] = verify_statistics(frames)
        report["passed"] = True
    except Exception as exc:
        report["failure"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        destination = ROOT/"audit/verification.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        temporary.replace(destination)
    print(json.dumps(dict(passed=True, scenarios=len(report["ledger_checks"]),
                          raw_price_samples=len(report["raw_price_samples"]),
                          metrics_rows=report["statistics"]["rows"],
                          output="audit/verification.json"), ensure_ascii=False))


if __name__ == "__main__":
    main()
