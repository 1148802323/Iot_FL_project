from __future__ import annotations

from typing import Any

from iot_fl.algorithms.base import (
    ExperimentConfig,
    FederatedAlgorithm,
)
from iot_fl.integration.flower_adapter import (
    run_flower_experiment,
)


class FedAvgAdapter(FederatedAlgorithm):
    name = "fedavg"
    display_name = "FedAvg"
    implementation_file = "src/iot_fl/integration/flower_adapter.py"

    def _run(
        self,
        distribution: str,
        config: ExperimentConfig,
    ) -> dict[str, Any]:
        return run_flower_experiment(
            aggregation="fedavg",
            distribution=distribution,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            l2=config.l2,
            data_path=config.data_path,
            factory_root=config.factory_root,
            seed=config.seed,
        )