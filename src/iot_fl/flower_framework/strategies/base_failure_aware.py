from __future__ import annotations

from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class BaseFailureAwareStrategy(FedAvg):
    """Base strategy for failure-aware aggregation variants."""

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, object]]:

        if not results:
            return None, {}

        if failures and not self.accept_failures:
            return None, {}

        self._validate_results(results)

        total_examples = sum(
            fit_res.num_examples
            for _, fit_res in results
        )

        global_failure_rate = sum(
            fit_res.num_examples
            * float(fit_res.metrics["failure_rate"])
            for _, fit_res in results
        ) / total_examples

        self._prepare_round(
            server_round=server_round,
            global_failure_rate=global_failure_rate,
        )

        weighted_parameters = []
        total_aggregation_weight = 0.0

        for _, fit_res in results:
            client_failure_rate = float(
                fit_res.metrics["failure_rate"]
            )

            aggregation_weight = self._calculate_client_weight(
                server_round=server_round,
                num_examples=fit_res.num_examples,
                client_failure_rate=client_failure_rate,
                global_failure_rate=global_failure_rate,
            )

            if aggregation_weight <= 0.0:
                raise ValueError(
                    "Aggregation weight must be greater than zero."
                )

            client_parameters = parameters_to_ndarrays(
                fit_res.parameters
            )

            weighted_parameters.append(
                (client_parameters, aggregation_weight)
            )

            total_aggregation_weight += aggregation_weight

        aggregated_ndarrays = [
            sum(
                client_parameters[layer_index]
                * (
                        aggregation_weight
                        / total_aggregation_weight
                )
                for client_parameters, aggregation_weight
                in weighted_parameters
            )
            for layer_index in range(
                len(weighted_parameters[0][0])
            )
        ]

        aggregated_parameters = ndarrays_to_parameters(
            aggregated_ndarrays
        )

        return aggregated_parameters, {}

    def _validate_results(
        self,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> None:

        for _, fit_res in results:
            failure_rate = fit_res.metrics.get("failure_rate")

            if failure_rate is None:
                raise ValueError(
                    "Failure-aware strategies require client metric "
                    "'failure_rate'."
                )

            failure_rate = float(failure_rate)

            if fit_res.num_examples <= 0:
                raise ValueError(
                    "Client num_examples must be greater than zero."
                )

            if not 0.0 <= failure_rate <= 1.0:
                raise ValueError(
                    "Client failure_rate must be between 0 and 1."
                )



    def _prepare_round(
            self,
            server_round: int,
            global_failure_rate: float,
    ) -> None:
        pass



    def _calculate_client_weight(
        self,
        server_round: int,
        num_examples: int,
        client_failure_rate: float,
        global_failure_rate: float,
    ) -> float:
        raise NotImplementedError(
            "Subclasses must implement "
            "_calculate_client_weight()."
        )