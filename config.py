"""Project settings kept in one place so every module uses the same values."""

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    start_date: str = "2015-01-01"
    end_date: str = "2026-01-01"
    market_ticker: str = "SPY"
    factor_tickers: list[str] = field(
        default_factory=lambda: ["MTUM", "VLUE", "QUAL", "USMV"]
    )

    hmm_states: int = 3
    hmm_warmup: int = 252
    hmm_refit_every: int = 21
    hmm_training_window: int | None = None  # None means an expanding window.
    hmm_covariance_type: str = "full"
    hmm_covariance_floor: float = 1e-4
    hmm_max_iterations: int = 300
    hmm_random_seeds: tuple[int, ...] = (42, 7, 21)

    allocation_lookback: int = 126
    minimum_allocation_history: int = 40
    risk_on_aversion: float = 2.0
    neutral_aversion: float = 3.0
    risk_off_aversion: float = 4.0
    maximum_factor_weight: float = 0.75
    covariance_regularization: float = 1e-6

    kelly_fraction: float = 0.50
    maximum_risky_exposure: float = 1.0
    annual_risk_free_rate: float = 0.02
    transaction_cost_bps: float = 5.0

    output_directory: str = "outputs"

    @property
    def all_tickers(self) -> list[str]:
        return [self.market_ticker, *self.factor_tickers]

    def validate(self) -> None:
        if self.hmm_states not in (2, 3):
            raise ValueError("hmm_states must be 2 or 3.")
        if self.hmm_warmup < 30:
            raise ValueError("hmm_warmup must be at least 30 observations.")
        if self.hmm_refit_every < 1:
            raise ValueError("hmm_refit_every must be positive.")
        if self.allocation_lookback < self.minimum_allocation_history:
            raise ValueError("allocation_lookback must be at least the minimum history.")
        if self.maximum_factor_weight * len(self.factor_tickers) < 1:
            raise ValueError("maximum_factor_weight makes the portfolio infeasible.")
        if not 0 <= self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be between 0 and 1.")
        if not 0 <= self.maximum_risky_exposure <= 1:
            raise ValueError("maximum_risky_exposure must be between 0 and 1.")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative.")


DEFAULT_CONFIG = ProjectConfig()
