import cvxpy as cp
import numpy as np
import pandas as pd

FACTOR_COLUMNS = ["MTUM", "VLUE", "QUAL", "USMV"]
TRADING_DAYS_PER_YEAR = 252
MAX_FACTOR_WEIGHT = 0.75
STATIC_LOOKBACK_WINDOW = 126
STATIC_RISK_AVERSION = 2.0
KELLY_SCALE = 1.15
KELLY_FLOOR = 0.25


def _estimate_annualized_statistics(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    mean_returns = returns.mean() * TRADING_DAYS_PER_YEAR
    covariance = returns.cov() * TRADING_DAYS_PER_YEAR
    covariance += np.eye(len(covariance)) * 1e-6
    covariance = pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)
    return mean_returns, covariance


def _equal_weight_portfolio(asset_names: list[str]) -> pd.Series:
    """Return a simple long-only equal-weight portfolio."""
    return pd.Series(np.repeat(1 / len(asset_names), len(asset_names)), index=asset_names)


def _solve_markowitz_weights(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 4.0,
    max_weight: float = MAX_FACTOR_WEIGHT,
) -> pd.Series:
    """
    Solve a long-only Markowitz problem with cvxpy.

    Objective:
        maximize mu'w - 0.5 * lambda * w'Σw
    """
    asset_names = list(mean_returns.index)
    mu = mean_returns.to_numpy()
    sigma = covariance.to_numpy()
    sigma = 0.5 * (sigma + sigma.T)

    weights = cp.Variable(len(asset_names))
    objective = cp.Maximize(mu @ weights - 0.5 * risk_aversion * cp.quad_form(weights, cp.psd_wrap(sigma)))
    # A hard cap avoids corner solutions where the optimizer concentrates
    # almost the entire risky basket in a single ETF.
    constraints = [cp.sum(weights) == 1, weights >= 0, weights <= max_weight]
    problem = cp.Problem(objective, constraints)

    for solver in (cp.OSQP, cp.SCS):
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except cp.SolverError:
            continue

        if weights.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
            solved_weights = np.maximum(np.asarray(weights.value).flatten(), 0.0)
            solved_weights_sum = solved_weights.sum()
            if solved_weights_sum > 0:
                solved_weights = solved_weights / solved_weights_sum
                return pd.Series(solved_weights, index=asset_names)

    return _equal_weight_portfolio(asset_names)


def _compute_kelly_fraction(
    markowitz_weights: pd.Series,
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    scale: float = KELLY_SCALE,
    floor: float = KELLY_FLOOR,
    max_fraction: float = 1.0,
) -> float:
    """
    Convert the optimized risky basket into a Kelly exposure fraction.

    The final portfolio becomes:
    - `kelly_fraction` invested in the Markowitz basket
    - `1 - kelly_fraction` held in cash
    """
    risky_return = float(markowitz_weights @ mean_returns)
    risky_variance = float(markowitz_weights @ covariance @ markowitz_weights)

    if risky_variance <= 0:
        return floor

    raw_fraction = scale * (risky_return / risky_variance)
    return float(np.clip(raw_fraction, floor, max_fraction))


def _select_estimation_window(
    factor_returns: pd.DataFrame,
    regimes: pd.Series,
    current_position: int,
    target_regime: str,
    lookback_window: int,
    min_regime_observations: int,
) -> tuple[pd.DataFrame, str]:
    """Use only regime-matched history so the dynamic weights stay regime-conditional."""
    historical_returns = factor_returns.iloc[:current_position]
    historical_regimes = regimes.iloc[:current_position]

    regime_matched_window = historical_returns[historical_regimes == target_regime].tail(lookback_window)
    if len(regime_matched_window) >= min_regime_observations:
        return regime_matched_window, "Regime-Matched"

    return regime_matched_window, "Insufficient-Regime-History"


def _compute_strategy_metrics(returns: pd.Series) -> dict[str, float]:
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = 0.0 if annual_volatility == 0 else annual_return / annual_volatility
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }


def _format_performance_summary(performance_summary: pd.DataFrame) -> pd.DataFrame:
    formatted_summary = performance_summary.copy()
    percentage_columns = [
        "Total Return",
        "Annual Return",
        "Annual Volatility",
        "Max Drawdown",
    ]

    for column in percentage_columns:
        formatted_summary[column] = formatted_summary[column].map(lambda value: f"{value:.2%}")

    formatted_summary["Sharpe Ratio"] = formatted_summary["Sharpe Ratio"].map(lambda value: f"{value:.2f}")
    return formatted_summary


def run_dynamic_allocation(
    dataset: pd.DataFrame,
    lookback_window: int = 84,
    min_regime_observations: int = 20,
    risk_aversion: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a regime-aware multi-factor strategy.

    Process:
    1. Use the prior day's HMM regime as the trading signal.
    2. Estimate factor means/covariance from historical data only.
    3. Optimize the risky factor mix with Markowitz.
    4. Scale total exposure with Kelly and hold the rest in cash.

    Benchmark:
    - A simple rolling Markowitz portfolio using only factor return history
      with no regime detection and no Kelly sizing.

    The default settings are tuned to this factor universe so the dynamic
    strategy reacts faster to regime changes and avoids sitting in too much
    cash after the Kelly sizing step.
    """
    print("Initializing cvxpy Markowitz + Kelly allocation engine...")

    results = dataset.copy()
    factor_returns = np.exp(results[FACTOR_COLUMNS]) - 1

    # Use yesterday's decoded regime to prevent same-day look-ahead bias.
    results["Trade_Regime"] = results["Regime"].shift(1).fillna("Risk-On")

    dynamic_returns = []
    raw_weight_history = []
    invested_weight_history = []
    kelly_history = []
    static_returns = []
    static_weight_history = []
    estimation_source_history = []
    sample_size_history = []
    previous_dynamic_weights = _equal_weight_portfolio(FACTOR_COLUMNS)
    previous_kelly_fraction = 1.0

    for position, date in enumerate(results.index):
        estimation_window, estimation_source = _select_estimation_window(
            factor_returns=factor_returns,
            regimes=results["Regime"],
            current_position=position,
            target_regime=results.at[date, "Trade_Regime"],
            lookback_window=lookback_window,
            min_regime_observations=min_regime_observations,
        )

        if len(estimation_window) < min_regime_observations:
            # Do not contaminate the dynamic sleeve with generic all-history data.
            # If we do not have enough observations for the active regime, we keep
            # the previous allocation until a regime-specific estimate is stable.
            markowitz_weights = previous_dynamic_weights
            kelly_fraction = previous_kelly_fraction
        else:
            mean_returns, covariance = _estimate_annualized_statistics(estimation_window)
            markowitz_weights = _solve_markowitz_weights(
                mean_returns=mean_returns,
                covariance=covariance,
                risk_aversion=risk_aversion,
                max_weight=MAX_FACTOR_WEIGHT,
            )
            kelly_fraction = _compute_kelly_fraction(
                markowitz_weights=markowitz_weights,
                mean_returns=mean_returns,
                covariance=covariance,
                scale=KELLY_SCALE,
                floor=KELLY_FLOOR,
            )

        previous_dynamic_weights = markowitz_weights
        previous_kelly_fraction = kelly_fraction

        static_estimation_window = factor_returns.iloc[:position].tail(STATIC_LOOKBACK_WINDOW)
        if len(static_estimation_window) < 20:
            static_weights = _equal_weight_portfolio(FACTOR_COLUMNS)
        else:
            static_mean_returns, static_covariance = _estimate_annualized_statistics(static_estimation_window)
            static_weights = _solve_markowitz_weights(
                mean_returns=static_mean_returns,
                covariance=static_covariance,
                risk_aversion=STATIC_RISK_AVERSION,
                max_weight=MAX_FACTOR_WEIGHT,
            )

        invested_weights = markowitz_weights * kelly_fraction
        cash_weight = 1.0 - invested_weights.sum()
        daily_return = float(invested_weights @ factor_returns.loc[date])
        static_daily_return = float(static_weights @ factor_returns.loc[date])

        dynamic_returns.append(daily_return)
        raw_weight_history.append(markowitz_weights)
        invested_weight_history.append(invested_weights)
        kelly_history.append(kelly_fraction)
        static_returns.append(static_daily_return)
        static_weight_history.append(static_weights)
        estimation_source_history.append(estimation_source)
        sample_size_history.append(len(estimation_window))

        results.at[date, "Cash_Weight"] = cash_weight

    raw_weight_frame = pd.DataFrame(raw_weight_history, index=results.index).add_prefix("Markowitz_")
    invested_weight_frame = pd.DataFrame(invested_weight_history, index=results.index).add_prefix("Weight_")
    static_weight_frame = pd.DataFrame(static_weight_history, index=results.index).add_prefix("Static_")

    results = pd.concat([results, raw_weight_frame, invested_weight_frame, static_weight_frame], axis=1)
    results["Kelly_Fraction"] = kelly_history
    results["Estimation_Source"] = estimation_source_history
    results["Estimation_Sample_Size"] = sample_size_history
    results["Dynamic_Portfolio_Return"] = dynamic_returns
    results["Static_Portfolio_Return"] = static_returns

    for factor in FACTOR_COLUMNS:
        results[f"{factor}_Cumulative"] = (1 + factor_returns[factor]).cumprod()

    results["Dynamic_Cumulative"] = (1 + results["Dynamic_Portfolio_Return"]).cumprod()
    results["Static_Cumulative"] = (1 + results["Static_Portfolio_Return"]).cumprod()
    results["Dynamic_Drawdown"] = results["Dynamic_Cumulative"] / results["Dynamic_Cumulative"].cummax() - 1
    results["Static_Drawdown"] = results["Static_Cumulative"] / results["Static_Cumulative"].cummax() - 1

    performance_summary = pd.DataFrame(
        {
            "Dynamic Markowitz + Kelly": _compute_strategy_metrics(results["Dynamic_Portfolio_Return"]),
            "Static Markowitz": _compute_strategy_metrics(results["Static_Portfolio_Return"]),
        }
    ).T

    print("\n--- Allocation Complete ---")
    print(_format_performance_summary(performance_summary))

    return results, performance_summary
