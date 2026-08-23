"""Command-line entry point for the complete research pipeline."""

import argparse
import logging
from dataclasses import replace

from config import DEFAULT_CONFIG
from Data_Extraction import build_data_pipeline
from Data_Train_HMM import save_hmm_matrices_as_images, train_and_decode_hmm
from Data_Visualisation import plot_strategy_performance
from Dynamic_asset_allocation import run_dynamic_allocation
from sensitivity_analysis import run_sensitivity_analysis


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dynamic asset allocation backtest.")
    parser.add_argument("--start-date", default=DEFAULT_CONFIG.start_date)
    parser.add_argument("--end-date", default=DEFAULT_CONFIG.end_date)
    parser.add_argument("--hmm-states", type=int, choices=(2, 3), default=DEFAULT_CONFIG.hmm_states)
    parser.add_argument("--hmm-refit-every", type=int, default=DEFAULT_CONFIG.hmm_refit_every)
    parser.add_argument("--allocation-lookback", type=int, default=DEFAULT_CONFIG.allocation_lookback)
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_CONFIG.transaction_cost_bps)
    parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_CONFIG.annual_risk_free_rate)
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="Run predefined robustness checks after the main backtest.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    config = replace(
        DEFAULT_CONFIG,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        hmm_states=arguments.hmm_states,
        hmm_refit_every=arguments.hmm_refit_every,
        allocation_lookback=arguments.allocation_lookback,
        transaction_cost_bps=arguments.transaction_cost_bps,
        annual_risk_free_rate=arguments.risk_free_rate,
    )
    config.validate()

    market_data = build_data_pipeline(config.start_date, config.end_date)
    data_with_regimes, model = train_and_decode_hmm(market_data, config)
    portfolio_data, performance = run_dynamic_allocation(data_with_regimes, config)
    hmm_files = save_hmm_matrices_as_images(
        model, output_dir=f"{config.output_directory}/hmm"
    )
    report_files = plot_strategy_performance(portfolio_data, performance, config)
    if arguments.run_sensitivity:
        sensitivity = run_sensitivity_analysis(market_data, data_with_regimes, config)
        print("\nSensitivity analysis:")
        print(sensitivity)

    print("\nPerformance summary:")
    columns = ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Final Wealth"]
    print(performance[columns])
    print("\nSaved files:")
    for path in [*hmm_files, *report_files]:
        print(f"- {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
