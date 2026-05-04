import pandas as pd
import numpy as np
import yfinance as yf
import warnings

# Suppress yfinance warnings for cleaner output
warnings.filterwarnings('ignore')

def build_data_pipeline(start_date="2015-01-01", end_date="2026-01-01"):
    """
    Fetches market data, engineers features for HMM, and aligns the time series.
    """
    
    # 1. Define the assets
    # SPY: S&P 500 (Broad Market Index for Regime Detection)
    # MTUM: Momentum Factor
    # VLUE: Value Factor
    # QUAL: Quality Factor
    # USMV: Minimum Volatility Factor
    tickers = ['SPY', 'MTUM', 'VLUE', 'QUAL', 'USMV']
    
    print(f"Fetching historical data from {start_date} to {end_date}...")
    
    # 2. Fetch the data
    # We use 'Adj Close' to account for dividends and stock splits
    raw_prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust = False)['Adj Close']
    
    print("Engineering features...")
    
    # 3. Engineer Features
    # A. Calculate Daily Logarithmic Returns
    # Log returns are preferred because they are time-additive and more normally distributed
    log_returns = np.log(raw_prices / raw_prices.shift(1))
    
    # B. Calculate 20-Day Rolling Realized Volatility
    # The HMM will observe the broad market (SPY) volatility to detect regimes.
    # We multiply by np.sqrt(252) to annualize the daily standard deviation.
    volatility_window = 20
    realized_volatility = log_returns.rolling(window=volatility_window).std() * np.sqrt(252)
    
    # Combine the observable emissions the HMM will use to detect states
    hmm_features = pd.DataFrame({
        'SPY_Log_Return': log_returns['SPY'],
        'SPY_Realized_Vol': realized_volatility['SPY']
    })
    
    print("Cleaning and aligning data...")
    
    # 4. Clean and Structure Data
    # The rolling window will generate 20 days of NaNs at the start of the dataset.
    hmm_features_clean = hmm_features.dropna()
    log_returns_clean = log_returns.dropna()
    
    # Join the datasets ensuring all time series are perfectly aligned by date
    # This prevents look-ahead bias and alignment errors during backtesting
    aligned_dataset = hmm_features_clean.join(log_returns_clean.drop(columns=['SPY']), how='inner')
    
    return aligned_dataset

# Execute the pipeline
if __name__ == "__main__":
    market_data = build_data_pipeline()
    
    print("\n--- Pipeline Execution Complete ---")
    print(f"Total Trading Days Processed: {market_data.shape[0]}")
    print("\nFirst 5 rows of the structured dataset:")
    print(market_data.head())

