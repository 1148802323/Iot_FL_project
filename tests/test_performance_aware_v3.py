from __future__ import annotations

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from iot_fl.flower_framework.strategies.failure_aware_strategy import (
    PerformanceAwareStrategy,
)
from train_performance_aware_fedavg import _local_recall


def test_local_recall_handles_positive_and_empty_positive_sets() -> None:
    assert _local_recall(np.array([1, 1, 0]), np.array([0.7, 0.2, 0.9])) == 0.5
    assert _local_recall(np.array([0, 0]), np.array([0.8, 0.1])) == 0.0


def test_flower_strategy_uses_performance_aware_scores() -> None:
    status = Status(code=Code.OK, message="ok")
    results = [
        (
            None,
            FitRes(
                status,
                ndarrays_to_parameters([np.array([1.0])]),
                100,
                {"failure_rate": 0.1, "local_recall": 0.2},
            ),
        ),
        (
            None,
            FitRes(
                status,
                ndarrays_to_parameters([np.array([3.0])]),
                100,
                {"failure_rate": 0.1, "local_recall": 0.8},
            ),
        ),
    ]

    parameters, _metrics = PerformanceAwareStrategy(
        failure_lambda=1.0,
        beta=1.0,
    ).aggregate_fit(1, results, [])

    assert parameters is not None
    actual = float(parameters_to_ndarrays(parameters)[0][0])
    expected = (1.0 * 130.0 + 3.0 * 190.0) / 320.0
    assert actual == expected
