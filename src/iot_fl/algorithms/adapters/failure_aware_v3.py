from __future__ import annotations

from typing import Any

from iot_fl.algorithms.base import (
    ExperimentConfig,
    FederatedAlgorithm,
)

from iot_fl.integration.flower_adapter import (
    run_flower_experiment,
)


class FailureAwareV3Adapter(FederatedAlgorithm):
    name = "failure_aware_v3"
    display_name = "Failure-Aware FedAvg V3"
    implementation_file = (
        "src/iot_fl/flower_framework/"
        "strategies/failure_aware_v3.py"
    )

    def _run(
        self,
        distribution: str,
        config: ExperimentConfig,
    ) -> dict[str, Any]:
        return run_flower_experiment(
            aggregation="failure_aware_v3",
            distribution=distribution,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            l2=config.l2,
            data_path=config.data_path,
            factory_root=config.factory_root,
            seed=config.seed,
            v3_lambda=float(
                config.extra.get("lambda", 1.0)
            ),
            v3_beta=float(
                config.extra.get("beta", 1.0)
            ),
        )