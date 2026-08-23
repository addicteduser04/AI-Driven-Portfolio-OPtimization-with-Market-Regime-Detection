"""Portfolio construction and out-of-sample backtesting."""

import cvxpy as cp
import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, ProjectConfig

TRADING_DAYS_PER_YEAR = 252


def _validate_backtest_input(dataset: pd.DataFrame, config: ProjectConfig) -> None:
    required_columns = [
        "SPY_Log_Return",
        "Regime",
        *config.factor_tickers,
    ]
    missing = [column for column in required_columns if column not in dataset]
    if missing:
        raise ValueError(f"Missing backtest columns: {missing}")
    if not dataset.index.is_monotonic_increasing or not dataset.index.is_unique:
        raise ValueError("The dataset index must contain sorted, unique dates.")
    numerical_columns = ["SPY_Log_Return", *config.factor_tickers]
    if not np.isfinite(dataset[numerical_columns].to_numpy()).all():
        raise ValueError("Backtest returns must contain only finite values.")


def _estimate_statistics(returns: pd.DataFrame, regularization: float) -> tuple[pd.Series, pd.DataFrame]:
    mean_returns = returns.mean() * TRADING_DAYS_PER_YEAR
    covariance = returns.cov() * TRADING_DAYS_PER_YEAR
    covariance = covariance + np.eye(len(covariance)) * regularization
    return mean_returns, covariance


def _equal_weights(asset_names: list[str]) -> pd.Series:
    return pd.Series(1 / len(asset_names), index=asset_names, dtype=float)


def _solve_markowitz_weights(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
    maximum_weight: float,
) -> pd.Series:
    """Solve a long-only Markowitz problem, or use equal weights if it fails."""
    asset_names = list(mean_returns.index)
    weights = cp.Variable(len(asset_names))
    covariance_array = covariance.to_numpy()
    covariance_array = (covariance_array + covariance_array.T) / 2

    objective = cp.Maximize(
        mean_returns.to_numpy() @ weights
        - 0.5 * risk_aversion * cp.quad_form(weights, cp.psd_wrap(covariance_array))
    )
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= maximum_weight,
    ]
    problem = cp.Problem(objective, constraints)

    for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except cp.SolverError:
            continue
        if problem.status in {"optimal", "optimal_inaccurate"} and weights.value is not None:
            solution = np.clip(np.asarray(weights.value).ravel(), 0, maximum_weight)
            if solution.sum() > 0:
                solution = solution / solution.sum()
                return pd.Series(solution, index=asset_names)

    return _equal_weights(asset_names)


def _compute_kelly_exposure(
    weights: pd.Series,
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    annual_risk_free_rate: float,
    fraction: float,
    maximum_exposure: float,
) -> float:
    """Use fractional Kelly: fraction * expected excess return / variance."""
    expected_return = float(weights @ mean_returns)
    variance = float(weights @ covariance @ weights)
    expected_excess_return = expected_return - annual_risk_free_rate

    if variance <= 0 or expected_excess_return <= 0:
        return 0.0

    raw_exposure = fraction * expected_excess_return / variance
    return float(np.clip(raw_exposure, 0, maximum_exposure))


def _risk_aversion_for_regime(regime: str, config: ProjectConfig) -> float:
    if regime == "Risk-Off":
        return config.risk_off_aversion
    if regime == "Neutral":
        return config.neutral_aversion
    return config.risk_on_aversion


def _maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def _strategy_metrics(
    returns: pd.Series,
    annual_risk_free_rate: float,
    turnover: pd.Series | None = None,
    transaction_costs: pd.Series | None = None,
) -> dict[str, object]:
    clean_returns = returns.dropna()
    if clean_returns.empty:
        raise ValueError("Cannot calculate metrics from an empty return series.")

    observations = len(clean_returns)
    final_wealth = float((1 + clean_returns).prod())
    years = observations / TRADING_DAYS_PER_YEAR
    cagr = final_wealth ** (1 / years) - 1
    annual_mean = float(clean_returns.mean() * TRADING_DAYS_PER_YEAR)
    annual_volatility = float(clean_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    excess_return = annual_mean - annual_risk_free_rate
    sharpe = np.nan if annual_volatility == 0 else excess_return / annual_volatility

    downside_returns = clean_returns[clean_returns < 0]
    downside_deviation = float(downside_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino = np.nan if downside_deviation == 0 or np.isnan(downside_deviation) else excess_return / downside_deviation
    max_drawdown = _maximum_drawdown(clean_returns)
    calmar = np.nan if max_drawdown == 0 else cagr / abs(max_drawdown)

    annual_turnover = 0.0
    if turnover is not None:
        annual_turnover = float(turnover.loc[clean_returns.index].sum() / years)
    costs_paid = 0.0
    if transaction_costs is not None:
        costs_paid = float(transaction_costs.loc[clean_returns.index].sum())

    return {
        "Cumulative Return": final_wealth - 1,
        "CAGR": cagr,
        "Annual Mean Return": annual_mean,
        "Volatility": annual_volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar": calmar,
        "Final Wealth": final_wealth,
        "Turnover": annual_turnover,
        "Transaction Costs Paid": costs_paid,
        "Observations": observations,
        "Backtest Start": clean_returns.index[0].date().isoformat(),
        "Backtest End": clean_returns.index[-1].date().isoformat(),
    }


def run_dynamic_allocation(
    dataset: pd.DataFrame,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every strategy over the same leakage-safe test dates."""
    config.validate()
    _validate_backtest_input(dataset, config)

    result = dataset.copy()
    factor_returns = np.exp(result[config.factor_tickers]) - 1
    spy_returns = np.exp(result["SPY_Log_Return"]) - 1
    daily_risk_free_rate = (1 + config.annual_risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    transaction_cost_rate = config.transaction_cost_bps / 10_000

    # A regime observed at today's close can first be traded on the next date.
    result["Trade_Regime"] = result["Regime"].shift(1)
    valid_dates = result.index[result["Trade_Regime"].notna()]
    if len(valid_dates) == 0:
        raise ValueError("No tradable dates remain after lagging the HMM signal.")

    dynamic_weights = pd.DataFrame(np.nan, index=result.index, columns=config.factor_tickers)
    rolling_weights = pd.DataFrame(np.nan, index=result.index, columns=config.factor_tickers)
    dynamic_exposure = pd.Series(np.nan, index=result.index, dtype=float)
    dynamic_turnover = pd.Series(np.nan, index=result.index, dtype=float)
    rolling_turnover = pd.Series(np.nan, index=result.index, dtype=float)

    previous_dynamic_position = pd.Series(0.0, index=config.factor_tickers)
    previous_rolling_position = pd.Series(0.0, index=config.factor_tickers)

    for date in valid_dates:
        position = result.index.get_loc(date)
        history = factor_returns.iloc[:position].tail(config.allocation_lookback)
        if len(history) < config.minimum_allocation_history:
            raise ValueError(f"Insufficient allocation history on first test date {date.date()}.")

        mean_returns, covariance = _estimate_statistics(
            history, config.covariance_regularization
        )
        regime = str(result.at[date, "Trade_Regime"])
        dynamic_composition = _solve_markowitz_weights(
            mean_returns,
            covariance,
            _risk_aversion_for_regime(regime, config),
            config.maximum_factor_weight,
        )
        exposure = _compute_kelly_exposure(
            dynamic_composition,
            mean_returns,
            covariance,
            config.annual_risk_free_rate,
            config.kelly_fraction,
            config.maximum_risky_exposure,
        )
        dynamic_position = dynamic_composition * exposure

        rolling_position = _solve_markowitz_weights(
            mean_returns,
            covariance,
            config.neutral_aversion,
            config.maximum_factor_weight,
        )

        dynamic_weights.loc[date] = dynamic_position
        rolling_weights.loc[date] = rolling_position
        dynamic_exposure.loc[date] = exposure
        dynamic_turnover.loc[date] = float((dynamic_position - previous_dynamic_position).abs().sum())
        rolling_turnover.loc[date] = float((rolling_position - previous_rolling_position).abs().sum())
        previous_dynamic_position = dynamic_position
        previous_rolling_position = rolling_position

    result[[f"Weight_{name}" for name in config.factor_tickers]] = dynamic_weights.rename(
        columns={name: f"Weight_{name}" for name in config.factor_tickers}
    )
    result["Risky_Exposure"] = dynamic_exposure
    result["Cash_Weight"] = 1 - dynamic_exposure
    result["Turnover"] = dynamic_turnover
    result["Transaction_Cost"] = dynamic_turnover * transaction_cost_rate

    dynamic_risky_return = (dynamic_weights * factor_returns).sum(axis=1, min_count=1)
    dynamic_cash_return = (1 - dynamic_exposure) * daily_risk_free_rate
    result["Dynamic_Gross_Return"] = dynamic_risky_return + dynamic_cash_return
    result["Dynamic_Net_Return"] = result["Dynamic_Gross_Return"] - result["Transaction_Cost"]

    result["Rolling_Turnover"] = rolling_turnover
    result["Rolling_Transaction_Cost"] = rolling_turnover * transaction_cost_rate
    rolling_gross = (rolling_weights * factor_returns).sum(axis=1, min_count=1)
    result["Rolling_Markowitz_Return"] = rolling_gross - result["Rolling_Transaction_Cost"]
    result["Equal_Weight_Return"] = factor_returns.mean(axis=1).where(result.index.isin(valid_dates))
    result["SPY_Buy_Hold_Return"] = spy_returns.where(result.index.isin(valid_dates))

    return_columns = {
        "Dynamic HMM + Markowitz + Kelly, gross": "Dynamic_Gross_Return",
        "Dynamic HMM + Markowitz + Kelly, net": "Dynamic_Net_Return",
        "Rolling Markowitz": "Rolling_Markowitz_Return",
        "Equal Weight": "Equal_Weight_Return",
        "SPY Buy & Hold": "SPY_Buy_Hold_Return",
    }
    metrics = {}
    for strategy, column in return_columns.items():
        turnover = None
        costs = None
        if strategy.startswith("Dynamic"):
            turnover = result["Turnover"]
            costs = result["Transaction_Cost"] if strategy.endswith("net") else None
        elif strategy == "Rolling Markowitz":
            turnover = result["Rolling_Turnover"]
            costs = result["Rolling_Transaction_Cost"]
        metrics[strategy] = _strategy_metrics(
            result[column], config.annual_risk_free_rate, turnover, costs
        )

    performance_summary = pd.DataFrame(metrics).T
    test_slice = result.loc[valid_dates]
    result.loc[valid_dates, "Dynamic_Gross_Cumulative"] = (
        1 + test_slice["Dynamic_Gross_Return"]
    ).cumprod()
    result.loc[valid_dates, "Dynamic_Net_Cumulative"] = (
        1 + test_slice["Dynamic_Net_Return"]
    ).cumprod()
    result.loc[valid_dates, "Rolling_Markowitz_Cumulative"] = (
        1 + test_slice["Rolling_Markowitz_Return"]
    ).cumprod()
    result.loc[valid_dates, "Equal_Weight_Cumulative"] = (
        1 + test_slice["Equal_Weight_Return"]
    ).cumprod()
    result.loc[valid_dates, "SPY_Buy_Hold_Cumulative"] = (
        1 + test_slice["SPY_Buy_Hold_Return"]
    ).cumprod()

    print(f"Backtest: {valid_dates[0].date()} to {valid_dates[-1].date()} ({len(valid_dates)} rows)")
    return result, performance_summary
