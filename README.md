# Dynamic Asset Allocation with Regime Detection

A quantitative finance application that implements a sophisticated portfolio management system combining Hidden Markov Models (HMM) for market regime detection with Markowitz mean-variance optimization and Kelly Criterion scaling.

## 📋 Project Overview

This project develops an intelligent dynamic asset allocation strategy that:

1. **Detects Market Regimes** - Uses a Gaussian Hidden Markov Model to identify "risk-on" and "risk-off" market conditions based on SPY market data
2. **Optimizes Portfolio Weights** - Applies Markowitz mean-variance optimization with regime-aware risk aversion adjustments
3. **Scales with Kelly Criterion** - Adjusts position sizing using the Kelly Criterion for optimal growth
4. **Backtests Strategy** - Evaluates performance against market data from 2015 to 2026
5. **Visualizes Results** - Generates comprehensive performance reports with key metrics

## 🎯 Key Features

- **Regime Detection**: 2-state Hidden Markov Model identifies market volatility regimes
- **Dynamic Allocation**: Portfolio weights adjust based on detected market regime
- **Factor-Based Exposure**: Allocates across momentum (MTUM), value (VLUE), quality (QUAL), and low volatility (USMV) factors
- **Risk Management**: 
  - Regime-aware risk aversion coefficients
  - Maximum factor weight constraints (75%)
  - Kelly Criterion scaling with floor constraints (25% minimum)
- **Comprehensive Analytics**: Performance summaries, portfolio statistics, and regime analysis

## 📊 Data Sources

The strategy uses financial data from Yahoo Finance:

- **SPY**: S&P 500 ETF (primary market indicator and HMM emissions)
- **MTUM**: US Momentum factor
- **VLUE**: US Value factor
- **QUAL**: US Quality factor
- **USMV**: US Minimum Volatility factor

**Historical Period**: January 1, 2015 - January 1, 2026 (11 years)

## 🏗️ Project Structure

```
├── main.py                          # Entry point - orchestrates the full pipeline
├── Data_Extraction.py               # Fetches market data from yfinance
├── Data_Train_HMM.py                # Trains HMM for regime detection
├── Dynamic_asset_allocation.py      # Executes portfolio optimization & backtesting
├── Data_Visualisation.py            # Generates performance visualizations
├── requirements.txt                 # Python dependencies
├── outputs/
│   └── performance_summary.csv      # Strategy performance metrics
└── README.md                        # This file
```

## 📦 Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **yfinance** - Financial data fetching
- **scikit-learn** - Machine learning utilities
- **hmmlearn** - Hidden Markov Model implementation
- **matplotlib** - Data visualization
- **cvxpy** - Convex optimization for portfolio weights

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
cd /path/to/project
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Running the Project

Execute the main pipeline:

```bash
python main.py
```

### Pipeline Execution Flow

1. **Data Extraction** (`Data_Extraction.py`)
   - Downloads historical price data for all tickers
   - Computes HMM features: log returns, realized volatility, Z-score, drawdown metrics
   - Outputs structured dataset for HMM training

2. **HMM Training** (`Data_Train_HMM.py`)
   - Trains 2-state Gaussian HMM on the first 90% of data
   - Identifies risk-off regime based on volatility
   - Decodes hidden states for the entire dataset

3. **Dynamic Allocation** (`Dynamic_asset_allocation.py`)
   - Performs rolling 126-day lookback window optimization
   - Solves Markowitz mean-variance problems with regime-adjusted risk aversion
   - Applies Kelly Criterion scaling for position sizing
   - Generates daily portfolio returns and cumulative performance

4. **Visualization** (`Data_Visualisation.py`)
   - Plots strategy cumulative returns vs benchmarks
   - Generates regime timeline visualization
   - Creates performance summary table
   - Outputs results to `outputs/` directory

## 🔧 Configuration Parameters

Key parameters in the codebase (modify in source files):

### Data_Extraction.py
```python
MARKET_TICKER = "SPY"                           # Primary market indicator
FACTOR_TICKERS = ["MTUM", "VLUE", "QUAL", "USMV"]  # Factor tickers
```

### Data_Train_HMM.py
```python
HMM_FEATURE_COLUMNS = [
    "SPY_Log_Return",                           # Log returns of SPY
    "SPY_Realized_Vol",                         # 20-day realized volatility
    "SPY_Vol_ZScore",                           # Standardized volatility
    "SPY_Drawdown_63"                           # Maximum drawdown (63-day window)
]
```

### Dynamic_asset_allocation.py
```python
MAX_FACTOR_WEIGHT = 0.75                        # Maximum single factor weight
STATIC_LOOKBACK_WINDOW = 126                    # Number of trading days for estimation
STATIC_RISK_AVERSION = 2.0                      # Base Markowitz risk aversion coefficient
KELLY_SCALE = 1.15                              # Kelly Criterion scaling factor
KELLY_FLOOR = 0.25                              # Minimum position size (Kelly floor)
```

## 📈 Output Files

Generated in the `outputs/` directory:

- `performance_summary.csv` - Summary statistics including:
  - Total Return
  - Annual Return
  - Volatility
  - Sharpe Ratio
  - Maximum Drawdown
  - Regime breakdown statistics

- Performance plots showing:
  - Cumulative returns (strategy vs benchmarks)
  - Regime identification timeline
  - Portfolio allocation evolution

## 🎓 Methodology

### Hidden Markov Model (HMM)
- **Type**: 2-state Gaussian HMM
- **Features**: Log returns, realized volatility, volatility Z-score, drawdown metrics
- **Training**: Uses 90% of historical data
- **States**: 
  - Risk-off (high volatility, market stress)
  - Risk-on (normal market conditions)

### Portfolio Optimization
- **Approach**: Regime-aware Markowitz mean-variance optimization
- **Risk Aversion**: 
  - 2.0x in risk-on regimes (aggressive)
  - 4.0x in risk-off regimes (conservative)
- **Constraints**:
  - Long-only positions (non-negative weights)
  - Individual factor weights ≤ 75%
  - Weights sum to 1.0
  - Kelly Criterion scaling with 25% minimum floor

### Performance Metrics
The strategy measures success through:
- Cumulative returns
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Win rate (% positive days)
- Regime-specific returns

## 📝 Usage Examples

### Running the Full Pipeline
```bash
python main.py
```

### Importing Individual Modules
```python
from Data_Extraction import build_data_pipeline
from Data_Train_HMM import train_and_decode_hmm
from Dynamic_asset_allocation import run_dynamic_allocation
from Data_Visualisation import plot_strategy_performance

# Build data
market_data = build_data_pipeline()

# Train HMM
market_data_with_regimes, hmm_model = train_and_decode_hmm(market_data)

# Run allocation strategy
portfolio_data, performance = run_dynamic_allocation(market_data_with_regimes)

# Generate visualizations
output_files = plot_strategy_performance(portfolio_data, performance)
```

## 🔬 Technical Details

### HMM Features Explained

1. **SPY_Log_Return**: Daily log returns of the S&P 500
   - Captures market direction and magnitude of moves

2. **SPY_Realized_Vol**: 20-day rolling realized volatility
   - Measures current market turbulence

3. **SPY_Vol_ZScore**: Standardized volatility (mean=0, std=1)
   - Identifies extreme volatility periods relative to history

4. **SPY_Drawdown_63**: Maximum drawdown over 63-day window
   - Captures downside risk and market stress severity

### Optimization Problem (Markowitz)

For each day with lookback window:
```
maximize: μ'w - 0.5 * λ * w'Σw

subject to:
  w ≥ 0                  (long-only)
  1'w = 1                (fully invested)
  w_i ≤ 0.75 ∀i          (weight caps)
  
where:
  w = portfolio weights
  μ = annualized expected returns
  Σ = annualized covariance matrix
  λ = risk aversion (regime-dependent)
```

## 📊 Expected Results

Historical backtesting (2015-2026) typically shows:
- Outperformance vs SPY buy-and-hold in certain periods
- Reduced volatility through diversification
- Dynamic adaptation to market regime changes
- Positive returns in most market conditions

Actual results depend on:
- Market data freshness and accuracy
- Hyperparameter settings
- Regime detection quality
- Optimization convergence

## 🛠️ Troubleshooting

### Data Download Issues
- Ensure internet connection is stable
- Check if Yahoo Finance is accessible
- Verify ticker symbols are correct

### HMM Training Issues
- Ensure sufficient historical data (at least 2+ years recommended)
- Check that HMM features contain no NaN values
- Verify feature distributions are reasonable

### Optimization Errors
- Ensure covariance matrix is positive definite
- Check that lookback window has sufficient data
- Verify weight constraints are feasible (max weight > 1/n assets)

## 📚 References

- **Hidden Markov Models**: Rabiner, L. R. (1989). "A tutorial on hidden Markov models"
- **Markowitz Optimization**: Markowitz, H. (1952). "Portfolio Selection"
- **Kelly Criterion**: Kelly Jr., J. L. (1956). "A new interpretation of information rate"
- **hmmlearn Documentation**: https://hmmlearn.readthedocs.io/
- **cvxpy Documentation**: https://www.cvxpy.org/

## ⚠️ Disclaimers

- **Past Performance**: Historical backtesting results do not guarantee future performance
- **Risk Disclosure**: Trading and investment strategies carry significant risk, including potential loss of principal
- **Not Investment Advice**: This project is for educational purposes and should not be considered financial advice
- **Model Limitations**: HMM regime detection is based on historical patterns and may not predict future regimes accurately
- **Data Quality**: Results depend on the accuracy and completeness of financial data from Yahoo Finance

## 📄 License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## 👤 Author

[Your Name/Team Name]

## 🤝 Contributing

[Add contribution guidelines if applicable]

## 📞 Contact

[Add contact information if applicable]

## 🔄 Version History

- **v1.0** (2026) - Initial release with HMM regime detection and dynamic allocation

---

**Last Updated**: May 2026
