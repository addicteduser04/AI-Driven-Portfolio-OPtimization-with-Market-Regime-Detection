import numpy as np
import pandas as pd

from Dynamic_asset_allocation import (
    _compute_kelly_exposure,
    _equal_weights,
    _solve_markowitz_weights,
)


def _inputs():
    names = ["A", "B", "C", "D"]
    means = pd.Series([0.08, 0.07, 0.06, 0.05], index=names)
    covariance = pd.DataFrame(np.eye(4) * 0.04, index=names, columns=names)
    return means, covariance


def test_markowitz_weights_are_valid():
    means, covariance = _inputs()
    weights = _solve_markowitz_weights(means, covariance, 2.0, 0.5)
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1)
    assert (weights >= 0).all()
    assert (weights <= 0.5 + 1e-8).all()


def test_equal_weight_fallback_is_valid():
    weights = _equal_weights(["A", "B", "C", "D"])
    assert np.isclose(weights.sum(), 1)
    assert np.allclose(weights, 0.25)


def test_kelly_is_zero_for_negative_edge():
    means, covariance = _inputs()
    means[:] = -0.01
    exposure = _compute_kelly_exposure(
        _equal_weights(list(means.index)), means, covariance, 0.02, 0.5, 1.0
    )
    assert exposure == 0


def test_kelly_is_zero_for_zero_variance():
    means, covariance = _inputs()
    covariance.iloc[:, :] = 0
    exposure = _compute_kelly_exposure(
        _equal_weights(list(means.index)), means, covariance, 0.02, 0.5, 1.0
    )
    assert exposure == 0


def test_kelly_is_capped():
    means, covariance = _inputs()
    covariance *= 0.00001
    exposure = _compute_kelly_exposure(
        _equal_weights(list(means.index)), means, covariance, 0.0, 0.5, 0.8
    )
    assert exposure == 0.8
