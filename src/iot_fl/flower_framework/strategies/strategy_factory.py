from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedAvg

from iot_fl.flower_framework.strategies.failure_aware_v1 import (
    FailureAwareV1Strategy,
)

from iot_fl.flower_framework.strategies.failure_aware_v2 import (
    FailureAwareV2Strategy,
)

from iot_fl.flower_framework.strategies.failure_aware_v3 import (
    FailureAwareV3Strategy,
)

from iot_fl.flower_framework.strategies.failure_aware_v4 import (
    FailureAwareV4Strategy,
)

def create_strategy(
    aggregation: str,
    **strategy_kwargs: Any,
) -> FedAvg:
    normalized_name = aggregation.strip().lower()

    if normalized_name == "fedavg":
        return FedAvg(**strategy_kwargs)

    if normalized_name == "failure_aware_v1":
        return FailureAwareV1Strategy(**strategy_kwargs)

    if normalized_name == "failure_aware_v4":
        return FailureAwareV4Strategy(**strategy_kwargs)

    if normalized_name == "failure_aware_v2":
        return FailureAwareV2Strategy(**strategy_kwargs)

    if normalized_name == "failure_aware_v3":
        return FailureAwareV3Strategy(**strategy_kwargs)

    raise ValueError(
        f"Unsupported aggregation strategy: {aggregation}"
    )