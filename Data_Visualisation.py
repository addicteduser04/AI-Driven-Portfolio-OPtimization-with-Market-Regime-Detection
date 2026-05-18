import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd

FACTOR_COLUMNS = ["MTUM", "VLUE", "QUAL", "USMV"]
WEIGHT_COLUMNS = [f"Weight_{factor}" for factor in FACTOR_COLUMNS]
ROLLING_WINDOW = 63


def _shade_risk_off_regimes(ax: plt.Axes, dataset: pd.DataFrame) -> None:
    """Highlight Risk-Off periods on a chart."""
    in_risk_off = False
    start_date = None

    for date, regime in dataset["Regime"].items():
        if regime == "Risk-Off" and not in_risk_off:
            start_date = date
            in_risk_off = True
        elif regime == "Risk-On" and in_risk_off:
            ax.axvspan(start_date, date, color="#f4cccc", alpha=0.35, linewidth=0)
            in_risk_off = False

    if in_risk_off:
        ax.axvspan(start_date, dataset.index[-1], color="#f4cccc", alpha=0.35, linewidth=0)


def _save_figure(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_cumulative_performance(dataset: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dataset.index, dataset["Dynamic_Cumulative"], label="Dynamic Markowitz + Kelly", linewidth=2.5)
    ax.plot(dataset.index, dataset["Static_Cumulative"], label="Static Markowitz", linewidth=1.8, alpha=0.9)
    _shade_risk_off_regimes(ax, dataset)
    ax.set_title("Portfolio Growth with HMM Regime Shading")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir / "strategy_performance.png")


def _plot_drawdowns(dataset: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dataset.index, dataset["Dynamic_Drawdown"], label="Dynamic Drawdown", linewidth=2.0)
    ax.plot(dataset.index, dataset["Static_Drawdown"], label="Static Markowitz Drawdown", linewidth=1.6, alpha=0.85)
    ax.fill_between(dataset.index, dataset["Dynamic_Drawdown"], 0, alpha=0.2)
    ax.set_title("Max Drawdown Profile")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir / "drawdown_profile.png")


def _plot_return_distribution(dataset: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(dataset["Dynamic_Portfolio_Return"], bins=50, alpha=0.6, label="Dynamic", density=True)
    ax.hist(dataset["Static_Portfolio_Return"], bins=50, alpha=0.5, label="Static Markowitz", density=True)
    ax.axvline(dataset["Dynamic_Portfolio_Return"].mean(), color="C0", linestyle="--", linewidth=1.5)
    ax.axvline(dataset["Static_Portfolio_Return"].mean(), color="C1", linestyle="--", linewidth=1.5)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir / "returns_distribution.png")


def _plot_factor_returns(dataset: pd.DataFrame, output_dir: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for axis, factor in zip(axes, FACTOR_COLUMNS):
        axis.plot(dataset.index, dataset[f"{factor}_Cumulative"], linewidth=2.0)
        axis.set_title(f"{factor} Cumulative Return")
        axis.set_ylabel("Growth of $1")
        axis.grid(alpha=0.3)

    return _save_figure(fig, output_dir / "factor_returns.png")


def _plot_rolling_risk_metrics(dataset: pd.DataFrame, output_dir: Path) -> str:
    rolling_dynamic_vol = dataset["Dynamic_Portfolio_Return"].rolling(ROLLING_WINDOW).std() * (252 ** 0.5)
    rolling_static_vol = dataset["Static_Portfolio_Return"].rolling(ROLLING_WINDOW).std() * (252 ** 0.5)

    dynamic_sharpe = (
        dataset["Dynamic_Portfolio_Return"].rolling(ROLLING_WINDOW).mean()
        / dataset["Dynamic_Portfolio_Return"].rolling(ROLLING_WINDOW).std()
    ) * (252 ** 0.5)
    static_sharpe = (
        dataset["Static_Portfolio_Return"].rolling(ROLLING_WINDOW).mean()
        / dataset["Static_Portfolio_Return"].rolling(ROLLING_WINDOW).std()
    ) * (252 ** 0.5)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(dataset.index, rolling_dynamic_vol, label="Dynamic Volatility", linewidth=2.0)
    axes[0].plot(dataset.index, rolling_static_vol, label="Static Markowitz Volatility", linewidth=1.7)
    axes[0].set_title(f"{ROLLING_WINDOW}-Day Rolling Volatility")
    axes[0].set_ylabel("Annualized Volatility")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(dataset.index, dynamic_sharpe, label="Dynamic Sharpe", linewidth=2.0)
    axes[1].plot(dataset.index, static_sharpe, label="Static Markowitz Sharpe", linewidth=1.7)
    axes[1].set_title(f"{ROLLING_WINDOW}-Day Rolling Sharpe Ratio")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    return _save_figure(fig, output_dir / "rolling_risk_metrics.png")


def _plot_allocation_history(dataset: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(14, 7))
    stacked_weights = [dataset[column] for column in WEIGHT_COLUMNS] + [dataset["Cash_Weight"]]
    labels = FACTOR_COLUMNS + ["Cash"]
    ax.stackplot(dataset.index, stacked_weights, labels=labels, alpha=0.85)
    ax.set_title("Dynamic Allocation History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Weight")
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    return _save_figure(fig, output_dir / "allocation_history.png")


def plot_strategy_performance(
    dataset: pd.DataFrame,
    performance_summary: pd.DataFrame,
    output_dir: str = "outputs",
) -> list[str]:
    """Create a multi-chart visual report and save the summary table."""
    print("Generating strategy report...")

    plt.style.use("default")
    output_path = Path(output_dir)

    saved_files = [
        _plot_cumulative_performance(dataset, output_path),
        _plot_drawdowns(dataset, output_path),
        _plot_return_distribution(dataset, output_path),
        _plot_factor_returns(dataset, output_path),
        _plot_rolling_risk_metrics(dataset, output_path),
        _plot_allocation_history(dataset, output_path),
    ]

    summary_path = output_path / "performance_summary.csv"
    performance_summary.to_csv(summary_path)
    saved_files.append(str(summary_path))

    return saved_files
