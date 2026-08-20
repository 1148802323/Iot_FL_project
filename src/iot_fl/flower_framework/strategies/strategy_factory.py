from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedAvg

from iot_fl.flower_framework.strategies.failure_aware_strategy import (
    PerformanceAwareStrategy,
)

def create_strategy(
    aggregation: str,
    **strategy_kwargs: Any,
) -> FedAvg:
    normalized_name = aggregation.strip().lower()

    if normalized_name == "fedavg":
        return FedAvg(**strategy_kwargs)

    if normalized_name in {"performance_aware", "performance_aware_v3"}:
        return PerformanceAwareStrategy(**strategy_kwargs)

    raise ValueError(
        f"Unsupported aggregation strategy: {aggregation}"
    )
