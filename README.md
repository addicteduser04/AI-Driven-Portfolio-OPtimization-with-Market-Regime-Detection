# Dynamic Asset Allocation with Market Regime Detection

This student research project tests whether market regimes can improve a multi-factor portfolio. It is
an educational backtest, not an investment product and not investment advice.

## Pipeline

```text
Yahoo Finance adjusted prices
        -> SPY market features
        -> walk-forward Gaussian HMM
        -> Markowitz factor weights
        -> fractional Kelly exposure
        -> costs, benchmarks, metrics, and charts
```

The risky assets are MTUM, VLUE, QUAL, and USMV. SPY is used for the market-regime features and as a
buy-and-hold benchmark.

## Leakage controls

- The HMM has an initial 252-row warmup.
- At every refit, the feature scaler and HMM see only dates before the prediction date.
- State probabilities use a forward filter. No future observations are used for smoothing.
- The regime observed after one close is shifted by one day before it becomes a trading signal.
- Portfolio means and covariances use only returns before the day being traded.
- Warmup rows are excluded from every strategy, benchmark, metric, and chart.

## Default methodology

All defaults are defined in `config.py`.

- Data request: 2015-01-01 through 2026-01-01 (Yahoo's end date is exclusive)
- HMM: three Gaussian states, expanding training history, refitted every 21 trading days
- Regimes: Risk-On, Neutral, and Risk-Off, remapped after each fit from historical market statistics
- Allocation lookback: 126 trading days
- Portfolio constraints: long-only, sums to one, maximum 75% in one factor
- Risk aversion: 2.0 in Risk-On, 3.0 in Neutral, and 4.0 in Risk-Off
- Exposure: half-Kelly, from 0% to 100%; negative estimated excess return produces zero exposure
- Cash: constant 2% annual return
- Transaction costs: 5 basis points per unit of one-way turnover

The comparison strategies are SPY buy-and-hold, equal-weight factors, and rolling Markowitz without
the HMM/Kelly overlay. All use the same official test dates.

## Installation

Python 3.12 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run the full analysis

```bash
python main.py
```

Run the predefined robustness checks separately from the default specification:

```bash
python main.py --run-sensitivity
```

Important settings can be changed from the command line:

```bash
python main.py \
  --start-date 2015-01-01 \
  --end-date 2026-01-01 \
  --hmm-states 3 \
  --hmm-refit-every 21 \
  --allocation-lookback 126 \
  --transaction-cost-bps 5 \
  --risk-free-rate 0.02
```

## Tests and quality checks

```bash
pytest
ruff check .
mypy config.py Data_Extraction.py Data_Train_HMM.py Dynamic_asset_allocation.py Data_Visualisation.py main.py
python -m pip check
```

Tests cover sequential filtering, transition effects, future-data independence, state mapping, signal
lagging, Kelly limits, portfolio constraints, warmup exclusion, transaction costs, benchmarks, and
performance arithmetic. GitHub Actions runs the same checks on pushes and pull requests.

## Outputs

Running the project creates an ignored `outputs/` directory containing:

- `research_report.md`
- `performance_summary.csv`
- `regime_summary.csv`
- cumulative wealth, drawdown, regime, state probability, exposure, and turnover charts
- final HMM transition and standardized emission-mean charts

Generated files are intentionally excluded from Git because they can be reproduced from the code.

## Main limitations

- Yahoo Finance data can change and is not an institutional point-in-time database.
- The risk-free rate is constant rather than a historical Treasury series.
- Costs are simplified and omit spreads, market impact, taxes, and fund fees.
- Trades use close-to-close returns without a detailed execution model.
- Expected returns and HMM regimes are estimates and can be unstable.
- The default specification is a research choice, not a parameter set optimized for future performance.

## Project structure

```text
config.py                       shared research settings
Data_Extraction.py              downloads prices and builds SPY features
Data_Train_HMM.py               fits and filters the walk-forward HMM
Dynamic_asset_allocation.py     constructs portfolios and calculates metrics
Data_Visualisation.py           creates research tables, charts, and report
main.py                         command-line pipeline
tests/                          automated research-integrity tests
```

## License and authorship

Academic PFA project at ENSIAS. No open-source license has been selected; all rights are reserved by
the project author(s).
