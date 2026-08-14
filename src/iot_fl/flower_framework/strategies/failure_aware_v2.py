from __future__ import annotations

from .base_failure_aware import BaseFailureAwareStrategy


class FailureAwareV2Strategy(BaseFailureAwareStrategy):
    """Failure-count FedAvg using absolute client failure count."""

    def __init__(
        self,
        alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if alpha < 0.0:
            raise ValueError(
                "alpha must be greater than or equal to zero."
            )

        self.alpha = float(alpha)

    def _calculate_client_weight(
        self,
        server_round: int,
        num_examples: int,
        client_failure_rate: float,
        global_failure_rate: float,
    ) -> float:
        client_failure_count = (
            num_examples * client_failure_rate
        )

        return (
            num_examples
            + self.alpha * client_failure_count
        )