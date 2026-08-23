from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

import Data_Train_HMM as hmm_module
from Data_Train_HMM import (
    HMM_FEATURE_COLUMNS,
    _filter_one_observation,
    _map_states_to_regimes,
    train_and_decode_hmm,
)


def _simple_model(transition_matrix):
    model = GaussianHMM(n_components=2, covariance_type="full")
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.asarray(transition_matrix, dtype=float)
    model.means_ = np.array([[0.0], [2.0]])
    model.n_features = 1
    model.covars_ = np.array([[[1.0]], [[1.0]]])
    return model


def test_transition_matrix_changes_filtered_probability():
    previous = np.array([0.95, 0.05])
    observation = np.array([1.0])
    persistent = _simple_model([[0.99, 0.01], [0.01, 0.99]])
    switching = _simple_model([[0.05, 0.95], [0.95, 0.05]])

    first_result = _filter_one_observation(previous, observation, persistent)
    second_result = _filter_one_observation(previous, observation, switching)

    assert not np.allclose(first_result, second_result)


def test_future_rows_do_not_change_earlier_results(market_frame, small_config):
    original, _ = train_and_decode_hmm(market_frame, small_config)
    changed_data = market_frame.copy()
    changed_data.iloc[-15:, changed_data.columns.get_loc("SPY_Log_Return")] += 2.0
    changed, _ = train_and_decode_hmm(changed_data, small_config)

    columns = ["Regime", "State_0_Probability", "State_1_Probability"]
    pd.testing.assert_frame_equal(original.iloc[:-15][columns], changed.iloc[:-15][columns])


def test_first_prediction_starts_after_warmup(market_frame, small_config):
    result, _ = train_and_decode_hmm(market_frame, small_config)
    assert result["Regime"].iloc[: small_config.hmm_warmup].isna().all()
    assert result["Regime"].iloc[small_config.hmm_warmup:].notna().all()


def test_short_dataset_fails_cleanly(market_frame, small_config):
    with pytest.raises(ValueError, match="At least"):
        train_and_decode_hmm(market_frame.iloc[:30], small_config)


def test_state_mapping_orders_calm_and_stressed_states():
    frame = pd.DataFrame(
        {
            "SPY_Log_Return": [0.01, 0.01, -0.02, -0.02],
            "SPY_Realized_Vol": [0.1, 0.1, 0.4, 0.4],
            "SPY_Drawdown_63": [0.0, 0.0, -0.2, -0.2],
        }
    )
    probabilities = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    mapping = _map_states_to_regimes(frame, probabilities)
    assert mapping == {0: "Risk-On", 1: "Risk-Off"}


def test_convergence_failure_raises(monkeypatch, small_config):
    class FailedModel:
        monitor_ = SimpleNamespace(converged=False, history=[0.0, 1.0])
        tol = 0.001

        def fit(self, data):
            return self

        def score(self, data):
            return 0.0

    monkeypatch.setattr(hmm_module, "_build_hmm_model", lambda config, seed: FailedModel())
    with pytest.raises(RuntimeError, match="did not converge"):
        hmm_module._fit_hmm(np.zeros((80, len(HMM_FEATURE_COLUMNS))), small_config)
