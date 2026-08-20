from __future__ import annotations

from typing import Any

from iot_fl.algorithms.base import (
    ExperimentConfig,
    FederatedAlgorithm,
    normalize_strategy_payload,
    shared_train_validation_test,
)

import train_performance_aware_fedavg as implementation


class PerformanceAwareV3Adapter(FederatedAlgorithm):
    name = "performance_aware_v3"
    display_name = "Performance-Aware FedAvg V3"
    implementation_file = "src/train_performance_aware_fedavg.py"

    def _run(self, distribution: str, config: ExperimentConfig) -> dict[str, Any]:
        train_ids, x_validation, y_validation, x_test, y_test = shared_train_validation_test(
            implementation,
            config,
        )
        history, _threshold_table, payload = implementation.performance_aware_strategy(
            strategy=distribution,
            factory_root=config.factory_root,
            train_ids=train_ids,
            x_val=x_validation,
            y_val=y_validation,
            x_test=x_test,
            y_test=y_test,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            lr=config.learning_rate,
            l2=config.l2,
            failure_lambda=float(config.extra.get("failure_lambda", 1.0)),
            beta=float(config.extra.get("beta", 1.0)),
        )
        result = normalize_strategy_payload(
            algorithm=self.name,
            distribution=distribution,
            final=payload["final"],
            history=history,
        )
        result["model"] = payload["model"]
        return result
