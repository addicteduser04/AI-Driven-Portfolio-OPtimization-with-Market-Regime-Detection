"""Small, predefined robustness checks. These are not used to tune the strategy."""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import ProjectConfig
from Data_Train_HMM import train_and_decode_hmm
from Dynamic_asset_allocation import run_dynamic_allocation


def run_sensitivity_analysis(
    market_data: pd.DataFrame,
    default_regime_data: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Run a few reasonable alternatives and save their net-strategy results."""
    scenarios = [
        ("Default", config, False),
        ("Two HMM states", replace(config, hmm_states=2), True),
        ("Alternative HMM seed", replace(config, hmm_random_seeds=(7,)), True),
        ("Refit every 42 days", replace(config, hmm_refit_every=42), True),
        ("84-day allocation lookback", replace(config, allocation_lookback=84), False),
        ("10 bps transaction cost", replace(config, transaction_cost_bps=10.0), False),
        ("Quarter Kelly", replace(config, kelly_fraction=0.25), False),
    ]

    rows = []
    for name, scenario_config, refit_hmm in scenarios:
        print(f"Sensitivity scenario: {name}")
        regime_data = default_regime_data
        if refit_hmm:
            regime_data, _ = train_and_decode_hmm(market_data, scenario_config)
        result, performance = run_dynamic_allocation(regime_data, scenario_config)
        net = performance.loc["Dynamic HMM + Markowitz + Kelly, net"]
        rows.append(
            {
                "Scenario": name,
                "CAGR": net["CAGR"],
                "Volatility": net["Volatility"],
                "Sharpe": net["Sharpe"],
                "Max Drawdown": net["Max Drawdown"],
                "Final Wealth": net["Final Wealth"],
                "Turnover": net["Turnover"],
                "Average Exposure": result["Risky_Exposure"].mean(),
            }
        )

    summary = pd.DataFrame(rows).set_index("Scenario")
    output_path = Path(config.output_directory) / "sensitivity_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    return summary
