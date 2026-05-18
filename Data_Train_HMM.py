import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

HMM_FEATURE_COLUMNS = [
    "SPY_Log_Return",
    "SPY_Realized_Vol",
    "SPY_Vol_ZScore",
    "SPY_Drawdown_63",
]


def _build_hmm_model(n_states: int) -> GaussianHMM:
    """Create the Gaussian HMM with the project's fixed configuration."""
    return GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )


def _identify_risk_off_state(
    hmm_model: GaussianHMM,
    training_slice: np.ndarray,
    training_index: pd.Index,
    training_volatility: pd.Series,
) -> int:
    """Map the model's arbitrary state labels to the highest-volatility regime."""
    training_states = pd.Series(
        hmm_model.predict(training_slice),
        index=training_index,
        name="HMM_State",
    )
    state_volatility = training_volatility.groupby(training_states).mean().sort_values()
    return int(state_volatility.idxmax())


def train_and_decode_hmm(
    dataset: pd.DataFrame,
    n_states: int = 3,
    warmup_periods: int = 252,
    refit_every: int = 21,
) -> tuple[pd.DataFrame, GaussianHMM]:
    """
    Train a Gaussian HMM with a walk-forward expanding window to avoid lookahead bias.

    At each time step `t`, the model:
    - trains only on data strictly before `t`
    - refits on a fixed cadence for efficiency
    - scores only the current observation via `predict_proba`
    """
    print(f"Training {n_states}-state Hidden Markov Model with walk-forward decoding...")

    decoded_dataset = dataset.copy()
    model_input = decoded_dataset[HMM_FEATURE_COLUMNS].to_numpy()

    hmm_states = np.full(len(decoded_dataset), np.nan)
    risk_off_probability = np.full(len(decoded_dataset), np.nan)
    regimes = np.full(len(decoded_dataset), None, dtype=object)

    current_model: GaussianHMM | None = None
    current_risk_off_state: int | None = None

    for position in range(len(decoded_dataset)):
        if position < warmup_periods:
            continue

        should_refit = current_model is None or (position - warmup_periods) % refit_every == 0
        if should_refit:
            training_slice = model_input[:position]
            current_model = _build_hmm_model(n_states)
            current_model.fit(training_slice)
            current_risk_off_state = _identify_risk_off_state(
                hmm_model=current_model,
                training_slice=training_slice,
                training_index=decoded_dataset.index[:position],
                training_volatility=decoded_dataset["SPY_Realized_Vol"].iloc[:position],
            )

        posterior = current_model.predict_proba(model_input[position : position + 1])[0]
        current_state = int(np.argmax(posterior))

        hmm_states[position] = current_state
        risk_off_probability[position] = posterior[current_risk_off_state]
        regimes[position] = "Risk-Off" if current_state == current_risk_off_state else "Risk-On"

    decoded_dataset["HMM_State"] = hmm_states
    decoded_dataset["Regime"] = pd.Series(regimes, index=decoded_dataset.index, dtype="object")
    decoded_dataset["Risk_Off_Probability"] = risk_off_probability

    if current_model is None or current_risk_off_state is None:
        current_model = _build_hmm_model(n_states)
        print("\n--- HMM Training Incomplete ---")
        print("Not enough history to fit the walk-forward HMM.")
        return decoded_dataset, current_model

    print("\n--- HMM Training Complete ---")
    print(f"Warmup periods: {warmup_periods}")
    print(f"Refit cadence: every {refit_every} bars")
    print(f"HMM features: {', '.join(HMM_FEATURE_COLUMNS)}")
    print(f"Latest Risk-Off state identified as State {current_risk_off_state}")
    print("\nLatest Transition Matrix:")
    print(np.round(current_model.transmat_, 3))

    return decoded_dataset, current_model
