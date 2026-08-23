from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from config import DEFAULT_CONFIG


@pytest.fixture
def small_config():
    return replace(
        DEFAULT_CONFIG,
        hmm_states=2,
        hmm_warmup=60,
        hmm_refit_every=30,
        hmm_max_iterations=100,
        hmm_random_seeds=(42, 7),
        allocation_lookback=40,
        minimum_allocation_history=20,
    )


@pytest.fixture
def market_frame():
    rng = np.random.default_rng(123)
    number_of_rows = 150
    index = pd.bdate_range("2020-01-01", periods=number_of_rows)
    market_returns = np.r_[
        rng.normal(0.0005, 0.006, number_of_rows // 2),
        rng.normal(-0.0005, 0.018, number_of_rows // 2),
    ]
    frame = pd.DataFrame(index=index)
    frame["SPY_Log_Return"] = market_returns
    frame["SPY_Realized_Vol"] = (
        pd.Series(market_returns, index=index).rolling(15).std().bfill() * np.sqrt(252)
    )
    rolling_mean = frame["SPY_Realized_Vol"].rolling(30).mean()
    rolling_std = frame["SPY_Realized_Vol"].rolling(30).std()
    frame["SPY_Vol_ZScore"] = (
        (frame["SPY_Realized_Vol"] - rolling_mean) / rolling_std
    ).fillna(0)
    growth = np.exp(frame["SPY_Log_Return"].cumsum())
    frame["SPY_Drawdown_63"] = growth / growth.rolling(40, min_periods=1).max() - 1
    for ticker in DEFAULT_CONFIG.factor_tickers:
        frame[ticker] = rng.normal(0.0003, 0.01, number_of_rows)
    return frame
