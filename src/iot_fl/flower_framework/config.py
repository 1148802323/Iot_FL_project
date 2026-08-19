from dataclasses import dataclass


@dataclass
class FrameworkConfig:
    aggregation: str
    num_clients: int
    num_rounds: int
    failure_aware_alpha: float = 1.0
    failure_aware_v2_alpha: float = 1.0
    failure_aware_v3_lambda: float = 1.0
    failure_aware_v3_beta: float = 1.0
    failure_aware_v4_schedule: str = "linear"
    failure_aware_v4_lambda_max: float = 2.0
    failure_aware_v4_target_recall: float = 0.85
    failure_aware_v4_eta: float = 0.25

    def __post_init__(self) -> None:
        if self.num_clients <= 0:
            raise ValueError("num_clients must be greater than zero")

        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be greater than zero")

        if self.failure_aware_alpha < 0.0:
            raise ValueError(
                "failure_aware_alpha must be greater than or equal to zero"
            )

        if self.failure_aware_v2_alpha < 0.0:
            raise ValueError(
                "failure_aware_v2_alpha must be greater than or equal to zero"
            )

        if self.failure_aware_v3_lambda < 0.0:
            raise ValueError(
                "failure_aware_v3_lambda must be greater than or equal to zero"
            )

        if self.failure_aware_v3_beta < 0.0:
            raise ValueError(
                "failure_aware_v3_beta must be greater than or equal to zero"
            )

        normalized_schedule = (
            self.failure_aware_v4_schedule
            .strip()
            .lower()
        )

        if normalized_schedule not in {
            "fixed",
            "linear",
            "recall_adaptive",
        }:
            raise ValueError(
                "failure_aware_v4_schedule must be "
                "'fixed' or 'linear' or 'recall_adaptive'"
            )

        if self.failure_aware_v4_lambda_max < 0.0:
            raise ValueError(
                "failure_aware_v4_lambda_max must be "
                "greater than or equal to zero"
            )

        if not 0.0 <= self.failure_aware_v4_target_recall <= 1.0:
            raise ValueError(
                "failure_aware_v4_target_recall must be between 0 and 1"
            )

        if self.failure_aware_v4_eta < 0.0:
            raise ValueError(
                "failure_aware_v4_eta must be greater than or equal to zero"
            )