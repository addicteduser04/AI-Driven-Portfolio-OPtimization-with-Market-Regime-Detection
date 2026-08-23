"""Charts and tables for the repaired out-of-sample backtest."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, ProjectConfig


def _save_figure(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _test_data(dataset: pd.DataFrame) -> pd.DataFrame:
    test = dataset.loc[dataset["Dynamic_Net_Return"].notna()].copy()
    if test.empty:
        raise ValueError("There are no valid test-period returns to plot.")
    return test


def _plot_cumulative_wealth(test: pd.DataFrame, output_dir: Path) -> str:
    columns = {
        "Dynamic_Gross_Cumulative": "Dynamic gross",
        "Dynamic_Net_Cumulative": "Dynamic net",
        "Rolling_Markowitz_Cumulative": "Rolling Markowitz",
        "Equal_Weight_Cumulative": "Equal weight",
        "SPY_Buy_Hold_Cumulative": "SPY",
    }
    fig, ax = plt.subplots(figsize=(13, 7))
    for column, label in columns.items():
        ax.plot(test.index, test[column], label=label)
    ax.set(title="Out-of-Sample Cumulative Wealth", ylabel="Growth of $1", xlabel="Date")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir / "cumulative_wealth.png")


def _plot_drawdowns(test: pd.DataFrame, output_dir: Path) -> str:
    return_series = {
        "Dynamic net": test["Dynamic_Net_Return"],
        "Rolling Markowitz": test["Rolling_Markowitz_Return"],
        "SPY": test["SPY_Buy_Hold_Return"],
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    for label, returns in return_series.items():
        wealth = (1 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        ax.plot(test.index, drawdown, label=label)
    ax.set(title="Out-of-Sample Drawdowns", ylabel="Drawdown", xlabel="Date")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir / "drawdowns.png")


def _plot_regimes(test: pd.DataFrame, output_dir: Path) -> str:
    regime_number = test["Trade_Regime"].map({"Risk-On": 0, "Neutral": 1, "Risk-Off": 2})
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(test.index, regime_number, where="post")
    ax.set_yticks([0, 1, 2], ["Risk-On", "Neutral", "Risk-Off"])
    ax.set(title="Lagged Regime Used for Trading", xlabel="Date")
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir / "regime_timeline.png")


def _plot_state_probabilities(test: pd.DataFrame, output_dir: Path) -> str:
    probability_columns = [
        column
        for column in test.columns
        if column.startswith("State_") and column.endswith("_Probability")
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    for column in probability_columns:
        ax.plot(test.index, test[column], label=column.replace("_Probability", ""))
    ax.set(title="Filtered HMM State Probabilities", ylabel="Probability", xlabel="Date", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir / "state_probabilities.png")


def _plot_exposure(test: pd.DataFrame, output_dir: Path, config: ProjectConfig) -> str:
    weight_columns = [f"Weight_{ticker}" for ticker in config.factor_tickers]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(
        test.index,
        *[test[column] for column in weight_columns],
        test["Cash_Weight"],
        labels=[*config.factor_tickers, "Cash"],
        alpha=0.85,
    )
    ax.set(title="Portfolio Exposure", ylabel="Portfolio weight", xlabel="Date", ylim=(0, 1.05))
    ax.legend(loc="upper left", ncol=3)
    return _save_figure(fig, output_dir / "portfolio_exposure.png")


def _plot_turnover(test: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test.index, test["Turnover"], linewidth=1)
    ax.set(title="Dynamic Portfolio Turnover", ylabel="One-way turnover", xlabel="Date")
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir / "turnover.png")


def _regime_summary(test: pd.DataFrame) -> pd.DataFrame:
    grouped = test.groupby("Trade_Regime")["Dynamic_Net_Return"]
    summary = grouped.agg(["count", "mean", "std"])
    summary["Annual Mean Return"] = summary["mean"] * 252
    summary["Annual Volatility"] = summary["std"] * np.sqrt(252)
    summary["Time Percentage"] = summary["count"] / len(test)
    return summary.drop(columns=["mean", "std"])


def _write_research_report(
    full_dataset: pd.DataFrame,
    test: pd.DataFrame,
    performance_summary: pd.DataFrame,
    config: ProjectConfig,
    output_dir: Path,
) -> str:
    table_columns = [
        "CAGR", "Volatility", "Sharpe", "Sortino", "Max Drawdown",
        "Calmar", "Final Wealth", "Turnover",
    ]
    report_table = performance_summary[table_columns].copy()
    for column in ["CAGR", "Volatility", "Max Drawdown", "Turnover"]:
        report_table[column] = report_table[column].map(lambda value: f"{float(value):.2%}")
    for column in ["Sharpe", "Sortino", "Calmar", "Final Wealth"]:
        report_table[column] = report_table[column].map(lambda value: f"{float(value):.3f}")

    report = f"""# Dynamic Asset Allocation Research Report

## Methodology

- Data: Yahoo Finance adjusted close prices for {config.market_ticker} and {', '.join(config.factor_tickers)}.
- Full data request: {config.start_date} to {config.end_date} (end date is exclusive).
- HMM: {config.hmm_states} Gaussian states, {config.hmm_warmup}-row warmup, refit every {config.hmm_refit_every} rows.
- Leakage controls: historical-only scaling and fitting, forward-only filtering, and a one-day trading-signal lag.
- State mapping: historical return, volatility, and drawdown rank states as Risk-On, Neutral, and Risk-Off.
- Allocation: long-only rolling Markowitz over {config.allocation_lookback} rows with a {config.maximum_factor_weight:.0%} weight cap.
- Exposure: {config.kelly_fraction:.0%} fractional Kelly, bounded between 0% and {config.maximum_risky_exposure:.0%}.
- Transaction costs: {config.transaction_cost_bps:.1f} basis points per unit of one-way turnover.
- Cash return: constant {config.annual_risk_free_rate:.2%} annual rate.

## Test period

- Cleaned data start: {full_dataset.index[0].date()}
- Warmup observations: {config.hmm_warmup}
- First tradable date: {test.index[0].date()}
- Final test date: {test.index[-1].date()}
- Test observations: {len(test)}
- Average risky exposure: {test['Risky_Exposure'].mean():.2%}

## Results

{report_table.to_markdown()}

## Limitations

This is an educational research backtest, not investment advice. It uses adjusted Yahoo Finance data,
a constant risk-free rate, simplified transaction costs, close-to-close returns, and no market-impact,
tax, borrowing, or execution model. HMM and expected-return estimates remain uncertain, and results do
not guarantee future performance.
"""
    path = output_dir / "research_report.md"
    path.write_text(report, encoding="utf-8")
    return str(path)


def plot_strategy_performance(
    dataset: pd.DataFrame,
    performance_summary: pd.DataFrame,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> list[str]:
    """Create report files from one common out-of-sample test slice."""
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = _test_data(dataset)

    files = [
        _plot_cumulative_wealth(test, output_dir),
        _plot_drawdowns(test, output_dir),
        _plot_regimes(test, output_dir),
        _plot_state_probabilities(test, output_dir),
        _plot_exposure(test, output_dir, config),
        _plot_turnover(test, output_dir),
    ]
    performance_path = output_dir / "performance_summary.csv"
    performance_summary.to_csv(performance_path)
    files.append(str(performance_path))

    regime_path = output_dir / "regime_summary.csv"
    _regime_summary(test).to_csv(regime_path)
    files.append(str(regime_path))
    files.append(_write_research_report(dataset, test, performance_summary, config, output_dir))
    return files
