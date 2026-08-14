from __future__ import annotations

from .base_failure_aware import BaseFailureAwareStrategy


class FailureAwareV1Strategy(BaseFailureAwareStrategy):
    """Static failure-aware FedAvg using relative failure lift."""

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

        if global_failure_rate > 0.0:
            failure_lift = (
                client_failure_rate
                / global_failure_rate
            )
        else:
            failure_lift = 0.0

        return (
            num_examples
            * (
                1.0
                + self.alpha
                * failure_lift
            )
        )