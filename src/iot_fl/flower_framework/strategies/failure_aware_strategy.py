from __future__ import annotations

from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


class FailureAwareStrategy(FedAvg):
    """Compatibility base for failure-aware strategies built on FedAvg."""


class PerformanceAwareStrategy(FailureAwareStrategy):
    """Aggregate updates by sample count, failure rate, and local recall."""

    def __init__(
        self,
        *,
        failure_lambda: float = 1.0,
        beta: float = 1.0,
        **kwargs,
    ) -> None:
        if failure_lambda < 0 or beta < 0:
            raise ValueError("failure_lambda and beta must be non-negative")
        super().__init__(**kwargs)
        self.failure_lambda = float(failure_lambda)
        self.beta = float(beta)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}

        weighted_updates = []
        for _client, fit_result in results:
            failure_rate = float(fit_result.metrics.get("failure_rate", 0.0))
            local_recall = float(fit_result.metrics.get("local_recall", 0.0))
            score = float(fit_result.num_examples) * (
                1.0 + self.failure_lambda * failure_rate + self.beta * local_recall
            )
            weighted_updates.append(
                (parameters_to_ndarrays(fit_result.parameters), score)
            )

        parameters = ndarrays_to_parameters(aggregate(weighted_updates))
        metrics: dict[str, bool | bytes | float | int | str] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_result.num_examples, fit_result.metrics)
                for _, fit_result in results
            ]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        return parameters, metrics
