import numpy as np
import pandas as pd
import yfinance as yf

from config import DEFAULT_CONFIG

MARKET_TICKER = DEFAULT_CONFIG.market_ticker
FACTOR_TICKERS = DEFAULT_CONFIG.factor_tickers
ALL_TICKERS = [MARKET_TICKER, *FACTOR_TICKERS]


def _extract_adjusted_close(downloaded_data: pd.DataFrame) -> pd.DataFrame:
    """Return a clean adjusted-close price table from yfinance output."""
    if isinstance(downloaded_data.columns, pd.MultiIndex):
        if "Adj Close" in downloaded_data.columns.get_level_values(0):
            prices = downloaded_data["Adj Close"].copy()
        elif "Close" in downloaded_data.columns.get_level_values(0):
            prices = downloaded_data["Close"].copy()
        else:
            raise ValueError("Could not find adjusted close prices in downloaded market data.")
    else:
        prices = downloaded_data.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=MARKET_TICKER)

    missing_tickers = [ticker for ticker in ALL_TICKERS if ticker not in prices.columns]
    if missing_tickers:
        raise ValueError(f"Missing price history for tickers: {missing_tickers}")

    return prices[ALL_TICKERS].sort_index().dropna(how="all")


def build_data_pipeline(
    start_date: str = DEFAULT_CONFIG.start_date,
    end_date: str = DEFAULT_CONFIG.end_date,
) -> pd.DataFrame:
    """
    Fetch price history and build the aligned dataset used by the HMM and allocator.

    Output columns contain:
    - HMM emissions derived from `SPY`
    - factor log returns for `MTUM`, `VLUE`, `QUAL`, `USMV`
    """
    print(f"Fetching historical data from {start_date} to {end_date}...")

    downloaded_data = yf.download(
        ALL_TICKERS,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    prices = _extract_adjusted_close(downloaded_data)
    print("Engineering features...")

    log_returns = np.log(prices / prices.shift(1))
    realized_volatility = log_returns[MARKET_TICKER].rolling(window=20).std() * np.sqrt(252)
    volatility_mean = realized_volatility.rolling(window=63).mean()
    volatility_std = realized_volatility.rolling(window=63).std()
    volatility_zscore = (realized_volatility - volatility_mean) / volatility_std

    spy_growth = np.exp(log_returns[MARKET_TICKER].cumsum())
    rolling_peak = spy_growth.rolling(window=63, min_periods=20).max()
    drawdown_63 = spy_growth / rolling_peak - 1

    dataset = pd.DataFrame(
        {
            "SPY_Log_Return": log_returns[MARKET_TICKER],
            "SPY_Realized_Vol": realized_volatility,
            "SPY_Vol_ZScore": volatility_zscore,
            "SPY_Drawdown_63": drawdown_63,
        },
        index=prices.index,
    )

    for ticker in FACTOR_TICKERS:
        dataset[ticker] = log_returns[ticker]

    print("Cleaning and aligning data...")
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna().copy()
    dataset.index.name = "Date"

    return dataset


if __name__ == "__main__":
    market_data = build_data_pipeline()
    print("\n--- Pipeline Execution Complete ---")
    print(f"Total Trading Days Processed: {market_data.shape[0]}")
    print("\nFirst 5 rows of the structured dataset:")
    print(market_data.head())
