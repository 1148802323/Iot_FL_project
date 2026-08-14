from __future__ import annotations

from typing import Any

from iot_fl.algorithms.base import (
    ExperimentConfig,
    FederatedAlgorithm,
)
from iot_fl.integration.flower_adapter import (
    run_flower_experiment,
)


class DynamicFailureAwareAdapter(FederatedAlgorithm):
    name = "dynamic_failure_aware"
    display_name = "Dynamic Failure-Aware FedAvg"
    implementation_file = "src/iot_fl/integration/flower_adapter.py"

    def _run(
        self,
        distribution: str,
        config: ExperimentConfig,
    ) -> dict[str, Any]:
        schedule = str(
            config.extra.get("schedule", "linear")
        )
        lambda_max = float(
            config.extra.get("lambda_max", 2.0)
        )
        target_recall = float(
            config.extra.get("target_recall", 0.85)
        )
        eta = float(
            config.extra.get("eta", 0.25)
        )

        return run_flower_experiment(
            aggregation="failure_aware_v4",
            distribution=distribution,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            l2=config.l2,
            data_path=config.data_path,
            factory_root=config.factory_root,
            seed=config.seed,
            v4_schedule=schedule,
            v4_lambda_max=lambda_max,
            v4_target_recall=target_recall,
            v4_eta=eta,
        )