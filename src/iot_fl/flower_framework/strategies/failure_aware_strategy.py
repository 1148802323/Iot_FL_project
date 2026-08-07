from __future__ import annotations

from flwr.common import Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from flwr.common import FitRes

class FailureAwareStrategy(FedAvg):
    """Pluggable strategy shell for future failure-aware aggregation."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, object]]:
        for _, fit_res in results:
            failure_rate = fit_res.metrics.get("failure_rate")

            if failure_rate is None:
                raise ValueError(
                    "FailureAwareStrategy requires client metric 'failure_rate'"
                )
            return super().aggregate_fit(
                server_round=server_round,
                results=results,
                failures=failures,
            )

