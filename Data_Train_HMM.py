"""Walk-forward Hidden Markov Model used to identify market regimes."""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_CONFIG, ProjectConfig

LOGGER = logging.getLogger(__name__)

HMM_FEATURE_COLUMNS = [
    "SPY_Log_Return",
    "SPY_Realized_Vol",
    "SPY_Vol_ZScore",
    "SPY_Drawdown_63",
]


def _validate_hmm_input(dataset: pd.DataFrame, config: ProjectConfig) -> None:
    missing = [column for column in HMM_FEATURE_COLUMNS if column not in dataset]
    if missing:
        raise ValueError(f"Missing HMM columns: {missing}")
    if not dataset.index.is_monotonic_increasing or not dataset.index.is_unique:
        raise ValueError("The dataset index must contain sorted, unique dates.")
    if not np.isfinite(dataset[HMM_FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("HMM features must contain only finite values.")
    if len(dataset) <= config.hmm_warmup:
        raise ValueError(
            f"At least {config.hmm_warmup + 1} rows are needed for the HMM; "
            f"received {len(dataset)}."
        )


def _build_hmm_model(config: ProjectConfig, random_seed: int) -> GaussianHMM:
    return GaussianHMM(
        n_components=config.hmm_states,
        covariance_type=config.hmm_covariance_type,
        min_covar=config.hmm_covariance_floor,
        n_iter=config.hmm_max_iterations,
        tol=1e-3,
        random_state=random_seed,
    )


def _fit_hmm(training_data: np.ndarray, config: ProjectConfig) -> GaussianHMM:
    """Try a few fixed seeds and return the best converged model."""
    converged_models: list[GaussianHMM] = []

    for seed in config.hmm_random_seeds:
        model = _build_hmm_model(config, seed)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                model.fit(training_data)
            except (ValueError, FloatingPointError) as error:
                LOGGER.warning("HMM fit failed with seed %s: %s", seed, error)
                continue

        for warning in caught_warnings:
            LOGGER.warning("HMM warning with seed %s: %s", seed, warning.message)

        score = model.score(training_data)
        history = list(model.monitor_.history)
        final_improvement = history[-1] - history[-2] if len(history) >= 2 else np.inf
        converged = model.monitor_.converged and -model.tol <= final_improvement < model.tol
        if converged and np.isfinite(score):
            converged_models.append(model)
        else:
            LOGGER.warning(
                "HMM did not converge with seed %s (final improvement %.6f).",
                seed,
                final_improvement,
            )

    if not converged_models:
        raise RuntimeError("The HMM did not converge with any configured random seed.")

    return max(converged_models, key=lambda fitted_model: fitted_model.score(training_data))


def _log_gaussian_density(observation: np.ndarray, model: GaussianHMM) -> np.ndarray:
    """Return the log emission probability of one observation for every state."""
    number_of_features = len(observation)
    log_probabilities = np.empty(model.n_components)

    for state in range(model.n_components):
        covariance = np.asarray(model.covars_[state], dtype=float)
        covariance = covariance + np.eye(number_of_features) * 1e-9
        difference = observation - model.means_[state]
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise RuntimeError("The fitted HMM produced an invalid covariance matrix.")
        squared_distance = difference @ np.linalg.solve(covariance, difference)
        log_probabilities[state] = -0.5 * (
            number_of_features * np.log(2 * np.pi) + log_determinant + squared_distance
        )

    return log_probabilities


def _filter_one_observation(
    previous_probability: np.ndarray | None,
    observation: np.ndarray,
    model: GaussianHMM,
) -> np.ndarray:
    """Apply one forward-filter step without using future observations."""
    prior = model.startprob_ if previous_probability is None else previous_probability @ model.transmat_
    log_emission = _log_gaussian_density(observation, model)

    # Subtracting the largest log value prevents numerical underflow.
    emission = np.exp(log_emission - np.max(log_emission))
    posterior = prior * emission
    total = posterior.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Could not calculate a valid HMM state probability.")
    return posterior / total


def _filter_sequence(observations: np.ndarray, model: GaussianHMM) -> np.ndarray:
    probabilities = np.empty((len(observations), model.n_components))
    previous_probability = None
    for position, observation in enumerate(observations):
        previous_probability = _filter_one_observation(previous_probability, observation, model)
        probabilities[position] = previous_probability
    return probabilities


def _map_states_to_regimes(
    training_frame: pd.DataFrame,
    state_probabilities: np.ndarray,
) -> dict[int, str]:
    """Order states from calm/positive (Risk-On) to stressed (Risk-Off)."""
    state_statistics = []

    for state in range(state_probabilities.shape[1]):
        weights = state_probabilities[:, state]
        weight_sum = weights.sum()
        if weight_sum <= 0:
            stress_score = float("inf")
        else:
            mean_return = np.average(training_frame["SPY_Log_Return"], weights=weights)
            volatility = np.average(training_frame["SPY_Realized_Vol"], weights=weights)
            drawdown = np.average(training_frame["SPY_Drawdown_63"], weights=weights)
            stress_score = volatility - mean_return - drawdown
        state_statistics.append((state, stress_score))

    ordered_states = [state for state, _ in sorted(state_statistics, key=lambda item: item[1])]
    if len(ordered_states) == 2:
        names = ["Risk-On", "Risk-Off"]
    else:
        names = ["Risk-On", "Neutral", "Risk-Off"]
    return dict(zip(ordered_states, names))


def train_and_decode_hmm(
    dataset: pd.DataFrame,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, GaussianHMM]:
    """Fit the HMM on past data and filter each new observation one date at a time."""
    config.validate()
    _validate_hmm_input(dataset, config)

    print(f"Training {config.hmm_states}-state HMM with walk-forward filtering...")
    result = dataset.copy()
    raw_features = result[HMM_FEATURE_COLUMNS].to_numpy()
    number_of_rows = len(result)

    probabilities = np.full((number_of_rows, config.hmm_states), np.nan)
    risk_off_probabilities = np.full(number_of_rows, np.nan)
    states = np.full(number_of_rows, np.nan)
    regimes = np.full(number_of_rows, None, dtype=object)

    current_model: GaussianHMM | None = None
    current_scaler: StandardScaler | None = None
    previous_probability: np.ndarray | None = None
    state_mapping: dict[int, str] = {}

    for position in range(config.hmm_warmup, number_of_rows):
        should_refit = current_model is None or (
            position - config.hmm_warmup
        ) % config.hmm_refit_every == 0

        if should_refit:
            training_start = 0
            if config.hmm_training_window is not None:
                training_start = max(0, position - config.hmm_training_window)

            training_frame = result.iloc[training_start:position]
            current_scaler = StandardScaler()
            scaled_training = current_scaler.fit_transform(
                training_frame[HMM_FEATURE_COLUMNS].to_numpy()
            )
            current_model = _fit_hmm(scaled_training, config)

            historical_probabilities = _filter_sequence(scaled_training, current_model)
            previous_probability = historical_probabilities[-1]
            state_mapping = _map_states_to_regimes(training_frame, historical_probabilities)

        if current_model is None or current_scaler is None:
            raise RuntimeError("Internal HMM initialization failed.")

        scaled_observation = current_scaler.transform(raw_features[position : position + 1])[0]
        previous_probability = _filter_one_observation(
            previous_probability, scaled_observation, current_model
        )
        probabilities[position] = previous_probability
        state = int(np.argmax(previous_probability))
        states[position] = state
        regimes[position] = state_mapping[state]
        risk_off_state = next(
            mapped_state for mapped_state, name in state_mapping.items() if name == "Risk-Off"
        )
        risk_off_probabilities[position] = previous_probability[risk_off_state]

    result["HMM_State"] = states
    result["Regime"] = regimes
    for state in range(config.hmm_states):
        result[f"State_{state}_Probability"] = probabilities[:, state]
    result["Risk_Off_Probability"] = risk_off_probabilities

    print(f"First tradable signal date: {result.index[config.hmm_warmup].date()}")
    print(f"Refit frequency: every {config.hmm_refit_every} trading days")
    return result, current_model


def _save_figure(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    return _save_figure(fig, path)


def save_hmm_matrices_as_images(
    hmm_model: GaussianHMM,
    feature_names: list[str] = HMM_FEATURE_COLUMNS,
    output_dir: str = "outputs/hmm",
) -> list[str]:
    """Save simple diagnostics for the final fitted HMM."""
    if not hasattr(hmm_model, "transmat_"):
        raise ValueError("Cannot plot an HMM before it has been fitted.")

    output_path = Path(output_dir)
    state_labels = [f"State {state}" for state in range(hmm_model.n_components)]
    files = [
        _heatmap(
            hmm_model.transmat_,
            state_labels,
            "HMM Transition Matrix",
            output_path / "hmm_transition_matrix.png",
        ),
        _heatmap(
            hmm_model.means_,
            feature_names,
            "Standardized HMM Emission Means",
            output_path / "hmm_emission_means.png",
        ),
    ]
    return files
