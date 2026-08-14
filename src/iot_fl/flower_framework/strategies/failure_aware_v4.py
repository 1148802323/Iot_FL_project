from __future__ import annotations



from .base_failure_aware import BaseFailureAwareStrategy


class FailureAwareV4Strategy(BaseFailureAwareStrategy):
    """Dynamic failure-aware FedAvg with fixed or linear lambda schedule."""

    def __init__(
        self,
        schedule: str = "linear",
        lambda_max: float = 2.0,
        total_rounds: int = 50,
        target_recall: float = 0.85,
        eta: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        normalized_schedule = schedule.strip().lower()

        if normalized_schedule not in {
            "fixed",
            "linear",
            "recall_adaptive",
        }:
            raise ValueError(
                "schedule must be 'fixed', 'linear', "
                "or 'recall_adaptive'."
            )

        if lambda_max < 0.0:
            raise ValueError(
                "lambda_max must be greater than or equal to zero."
            )

        if total_rounds <= 0:
            raise ValueError(
                "total_rounds must be greater than zero."
            )

        if not 0.0 <= target_recall <= 1.0:
            raise ValueError(
                "target_recall must be between 0 and 1."
            )

        if eta < 0.0:
            raise ValueError(
                "eta must be greater than or equal to zero."
            )

        self.schedule = normalized_schedule
        self.lambda_max = float(lambda_max)
        self.total_rounds = int(total_rounds)
        self.target_recall = float(target_recall)
        self.eta = float(eta)
        self.latest_validation_recall: float | None = None
        self.current_lambda = (
            self.lambda_max
            if self.schedule == "fixed"
            else 0.0
        )

    def _prepare_round(
        self,
        server_round: int,
        global_failure_rate: float,
    ) -> None:
        if self.schedule == "fixed":
            self.current_lambda = self.lambda_max
            return

        if self.schedule == "linear":
            self.current_lambda = (
                self.lambda_max
                * server_round
                / self.total_rounds
            )
        if self.schedule == "recall_adaptive":
            if self.latest_validation_recall is None:
                return

            self.current_lambda = float(
                max(
                    0.0,
                    min(
                        self.lambda_max,
                        self.current_lambda
                        + self.eta
                        * (
                            self.target_recall
                            - self.latest_validation_recall
                        ),
                    ),
                )
            )

    def evaluate(
            self,
            server_round: int,
            parameters,
    ):
        result = super().evaluate(
            server_round=server_round,
            parameters=parameters,
        )

        if result is None:
            return None

        loss, metrics = result

        recall = metrics.get("recall")

        if recall is not None:
            recall_value = float(recall)

            if not 0.0 <= recall_value <= 1.0:
                raise ValueError(
                    "Validation recall must be between 0 and 1."
                )

            self.latest_validation_recall = recall_value

        return loss, metrics

    def _calculate_client_weight(
        self,
        server_round: int,
        num_examples: int,
        client_failure_rate: float,
        global_failure_rate: float,
    ) -> float:
        return (
            num_examples
            * (
                1.0
                + self.current_lambda
                * client_failure_rate
            )
        )