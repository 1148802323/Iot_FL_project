from __future__ import annotations

import numpy as np

from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from iot_fl.flower_framework.strategies.failure_aware_v3 import (
    FailureAwareV3Strategy,
)


def make_fit_res(
    parameters: np.ndarray,
    *,
    num_examples: int,
    failure_rate: float,
    local_recall: float,
) -> FitRes:
    return FitRes(
        status=Status(
            code=Code.OK,
            message="OK",
        ),
        parameters=ndarrays_to_parameters(
            [parameters]
        ),
        num_examples=num_examples,
        metrics={
            "failure_rate": failure_rate,
            "local_recall": local_recall,
        },
    )


def test_failure_aware_v3_weighted_aggregation() -> None:
    strategy = FailureAwareV3Strategy(
        failure_lambda=1.0,
        beta=1.0,
    )

    client_1 = make_fit_res(
        np.array([1.0]),
        num_examples=100,
        failure_rate=0.2,
        local_recall=0.1,
    )

    client_2 = make_fit_res(
        np.array([3.0]),
        num_examples=100,
        failure_rate=0.4,
        local_recall=0.5,
    )

    aggregated, _ = strategy.aggregate_fit(
        server_round=1,
        results=[
            (None, client_1),
            (None, client_2),
        ],
        failures=[],
    )

    assert aggregated is not None

    result = parameters_to_ndarrays(
        aggregated
    )[0]

    # Client 1:
    # 100 * (1 + 0.2 + 0.1) = 130
    #
    # Client 2:
    # 100 * (1 + 0.4 + 0.5) = 190
    #
    # Expected:
    # (1 * 130 + 3 * 190) / (130 + 190)
    expected = (
        1.0 * 130.0
        + 3.0 * 190.0
    ) / 320.0

    np.testing.assert_allclose(
        result,
        np.array([expected]),
    )