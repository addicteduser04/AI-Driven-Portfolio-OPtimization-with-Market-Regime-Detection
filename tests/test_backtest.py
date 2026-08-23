import numpy as np
import pandas as pd

from Data_Train_HMM import train_and_decode_hmm
from Dynamic_asset_allocation import _maximum_drawdown, _strategy_metrics, run_dynamic_allocation


def test_warmup_signal_lag_and_benchmark_alignment(market_frame, small_config):
    decoded, _ = train_and_decode_hmm(market_frame, small_config)
    result, summary = run_dynamic_allocation(decoded, small_config)

    first_regime_date = decoded["Regime"].first_valid_index()
    first_trade_date = result["Dynamic_Net_Return"].first_valid_index()
    assert first_trade_date == decoded.index[decoded.index.get_loc(first_regime_date) + 1]
    assert result.loc[:first_regime_date, "Dynamic_Net_Return"].isna().all()

    return_columns = [
        "Dynamic_Net_Return",
        "Rolling_Markowitz_Return",
        "Equal_Weight_Return",
        "SPY_Buy_Hold_Return",
    ]
    for column in return_columns:
        assert result[column].dropna().index.equals(result["Dynamic_Net_Return"].dropna().index)
    assert summary["Observations"].nunique() == 1


def test_transaction_cost_arithmetic(market_frame, small_config):
    decoded, _ = train_and_decode_hmm(market_frame, small_config)
    result, _ = run_dynamic_allocation(decoded, small_config)
    valid = result["Dynamic_Net_Return"].notna()
    expected_cost = result.loc[valid, "Turnover"] * small_config.transaction_cost_bps / 10_000
    assert np.allclose(result.loc[valid, "Transaction_Cost"], expected_cost)
    assert np.allclose(
        result.loc[valid, "Dynamic_Net_Return"],
        result.loc[valid, "Dynamic_Gross_Return"] - expected_cost,
    )


def test_future_factor_returns_do_not_change_earlier_weights(market_frame, small_config):
    decoded, _ = train_and_decode_hmm(market_frame, small_config)
    original, _ = run_dynamic_allocation(decoded, small_config)
    changed = decoded.copy()
    changed.loc[changed.index[-10]:, small_config.factor_tickers] += 1.0
    changed_result, _ = run_dynamic_allocation(changed, small_config)

    weight_columns = [f"Weight_{ticker}" for ticker in small_config.factor_tickers]
    pd.testing.assert_frame_equal(
        original.iloc[:-10][weight_columns],
        changed_result.iloc[:-10][weight_columns],
    )


def test_known_cumulative_return_and_drawdown():
    index = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.Series([0.10, -0.10, 0.05], index=index)
    metrics = _strategy_metrics(returns, annual_risk_free_rate=0.0)
    expected_wealth = 1.10 * 0.90 * 1.05
    assert np.isclose(metrics["Final Wealth"], expected_wealth)
    assert np.isclose(metrics["Cumulative Return"], expected_wealth - 1)
    assert np.isclose(_maximum_drawdown(returns), -0.10)
