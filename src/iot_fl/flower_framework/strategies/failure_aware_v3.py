from __future__ import annotations

from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


class FailureAwareV3Strategy(FedAvg):
    """V3: aggregate by sample count, failure rate, and local recall."""

    def __init__(
        self,
        *,
        failure_lambda: float = 1.0,
        beta: float = 1.0,
        **kwargs,
    ) -> None:
        if failure_lambda < 0 or beta < 0:
            raise ValueError(
                "failure_lambda and beta must be non-negative"
            )

        super().__init__(**kwargs)

        self.failure_lambda = float(failure_lambda)
        self.beta = float(beta)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[
            tuple[ClientProxy, FitRes] | BaseException
        ],
    ) -> tuple[Parameters | None, dict]:

        if not results:
            return None, {}

        if failures and not self.accept_failures:
            return None, {}

        weighted_updates = []

        for _, fit_result in results:
            if fit_result.num_examples <= 0:
                raise ValueError(
                    "Client num_examples must be greater than zero"
                )

            if "failure_rate" not in fit_result.metrics:
                raise ValueError(
                    "FailureAwareV3Strategy requires 'failure_rate'"
                )

            if "local_recall" not in fit_result.metrics:
                raise ValueError(
                    "FailureAwareV3Strategy requires 'local_recall'"
                )

            failure_rate = float(
                fit_result.metrics["failure_rate"]
            )
            local_recall = float(
                fit_result.metrics["local_recall"]
            )

            if not 0.0 <= failure_rate <= 1.0:
                raise ValueError(
                    "failure_rate must be between 0 and 1"
                )

            if not 0.0 <= local_recall <= 1.0:
                raise ValueError(
                    "local_recall must be between 0 and 1"
                )

            score = float(fit_result.num_examples) * (
                1.0
                + self.failure_lambda * failure_rate
                + self.beta * local_recall
            )

            weighted_updates.append(
                (
                    parameters_to_ndarrays(
                        fit_result.parameters
                    ),
                    score,
                )
            )

        aggregated_parameters = ndarrays_to_parameters(
            aggregate(weighted_updates)
        )

        return aggregated_parameters, {}