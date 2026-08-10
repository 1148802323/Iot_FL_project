from __future__ import annotations

from iot_fl.algorithms.adapters.dynamic_failure_aware import DynamicFailureAwareAdapter
from iot_fl.algorithms.adapters.failure_aware_v1 import FailureAwareV1Adapter
from iot_fl.algorithms.adapters.failure_aware_v2 import FailureAwareV2Adapter
from iot_fl.algorithms.adapters.fedavg import FedAvgAdapter

__all__ = [
    "DynamicFailureAwareAdapter",
    "FailureAwareV1Adapter",
    "FailureAwareV2Adapter",
    "FedAvgAdapter",
]

