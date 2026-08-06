from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedAvg

def create_strategy(
    aggregation: str,
    **strategy_kwargs: Any,
) -> FedAvg:
    normalized_name = aggregation.strip().lower()

    if normalized_name == "fedavg":
        return FedAvg(**strategy_kwargs)

    raise ValueError(
        f"Unsupported aggregation strategy: {aggregation}"
    )