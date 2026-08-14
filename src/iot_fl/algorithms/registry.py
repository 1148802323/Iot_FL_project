from __future__ import annotations

from typing import Any

from iot_fl.algorithms.adapters import (
    DynamicFailureAwareAdapter,
    FailureAwareV1Adapter,
    FailureAwareV2Adapter,
    FedAvgAdapter,
)
from iot_fl.algorithms.base import FederatedAlgorithm, SUPPORTED_DISTRIBUTIONS


ALGORITHM_REGISTRY: dict[str, FederatedAlgorithm] = {
    "fedavg": FedAvgAdapter(),
    "failure_aware_v1": FailureAwareV1Adapter(),
    "failure_aware_v2": FailureAwareV2Adapter(),
    "dynamic_failure_aware": DynamicFailureAwareAdapter(),
}


def list_algorithms() -> list[str]:
    return sorted(ALGORITHM_REGISTRY)


def get_algorithm(name: str) -> FederatedAlgorithm:
    try:
        return ALGORITHM_REGISTRY[name]
    except KeyError as error:
        supported = ", ".join(list_algorithms())
        raise ValueError(f"Unknown algorithm '{name}'. Supported algorithms: {supported}") from error


def list_distributions() -> list[str]:
    return list(SUPPORTED_DISTRIBUTIONS)


def run_experiment(
    algorithm: str = "fedavg",
    distribution: str = "highly_non_iid",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return get_algorithm(algorithm).run(distribution=distribution, config=config)

