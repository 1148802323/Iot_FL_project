import pytest

from iot_fl.flower_framework.strategies.failure_aware_v1 import (
    FailureAwareV1Strategy,
)

import numpy as np

from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays


def test_v1_alpha_zero_matches_fedavg_weight():
    strategy = FailureAwareV1Strategy(alpha=0.0)

    weight = strategy._calculate_client_weight(
        server_round=1,
        num_examples=100,
        client_failure_rate=0.30,
        global_failure_rate=0.20,
    )

    assert weight == pytest.approx(100.0)


def test_v1_failure_aware_weight():
    strategy = FailureAwareV1Strategy(alpha=1.0)

    weight = strategy._calculate_client_weight(
        server_round=1,
        num_examples=100,
        client_failure_rate=0.30,
        global_failure_rate=0.20,
    )

    assert weight == pytest.approx(250.0)


def test_v1_rejects_negative_alpha():
    with pytest.raises(
        ValueError,
        match="alpha must be greater than or equal to zero",
    ):
        FailureAwareV1Strategy(alpha=-1.0)

def _make_fit_res(
    values: list[float],
    num_examples: int,
    failure_rate: float,
) -> FitRes:
    return FitRes(
        status=Status(
            code=Code.OK,
            message="OK",
        ),
        parameters=ndarrays_to_parameters(
            [np.array(values, dtype=float)]
        ),
        num_examples=num_examples,
        metrics={
            "failure_rate": failure_rate,
        },
    )


def test_v1_aggregate_fit_alpha_zero_matches_sample_weighted_average():
    strategy = FailureAwareV1Strategy(alpha=0.0)

    fit_res_a = _make_fit_res(
        values=[1.0, 3.0],
        num_examples=100,
        failure_rate=0.10,
    )

    fit_res_b = _make_fit_res(
        values=[5.0, 7.0],
        num_examples=300,
        failure_rate=0.30,
    )

    aggregated_parameters, metrics = strategy.aggregate_fit(
        server_round=1,
        results=[
            (None, fit_res_a),
            (None, fit_res_b),
        ],
        failures=[],
    )

    aggregated = parameters_to_ndarrays(
        aggregated_parameters
    )[0]

    expected = np.array(
        [4.0, 6.0],
        dtype=float,
    )

    assert np.allclose(
        aggregated,
        expected,
    )

    assert metrics == {}

def test_v1_aggregate_fit_failure_aware_changes_global_parameters():
    strategy = FailureAwareV1Strategy(alpha=1.0)

    fit_res_a = _make_fit_res(
        values=[1.0, 3.0],
        num_examples=100,
        failure_rate=0.10,
    )

    fit_res_b = _make_fit_res(
        values=[5.0, 7.0],
        num_examples=100,
        failure_rate=0.30,
    )

    aggregated_parameters, _ = strategy.aggregate_fit(
        server_round=1,
        results=[
            (None, fit_res_a),
            (None, fit_res_b),
        ],
        failures=[],
    )

    aggregated = parameters_to_ndarrays(
        aggregated_parameters
    )[0]

    expected = np.array(
        [3.5, 5.5],
        dtype=float,
    )

    assert np.allclose(
        aggregated,
        expected,
    )

def test_v1_rejects_missing_failure_rate():
    strategy = FailureAwareV1Strategy(alpha=1.0)

    fit_res = FitRes(
        status=Status(
            code=Code.OK,
            message="OK",
        ),
        parameters=ndarrays_to_parameters(
            [np.array([1.0, 2.0], dtype=float)]
        ),
        num_examples=100,
        metrics={},
    )

    with pytest.raises(
        ValueError,
        match="failure_rate",
    ):
        strategy.aggregate_fit(
            server_round=1,
            results=[
                (None, fit_res),
            ],
            failures=[],
        )


def test_v1_rejects_invalid_failure_rate():
    strategy = FailureAwareV1Strategy(alpha=1.0)

    fit_res = _make_fit_res(
        values=[1.0, 2.0],
        num_examples=100,
        failure_rate=1.5,
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        strategy.aggregate_fit(
            server_round=1,
            results=[
                (None, fit_res),
            ],
            failures=[],
        )


def test_v1_rejects_zero_num_examples():
    strategy = FailureAwareV1Strategy(alpha=1.0)

    fit_res = _make_fit_res(
        values=[1.0, 2.0],
        num_examples=0,
        failure_rate=0.2,
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        strategy.aggregate_fit(
            server_round=1,
            results=[
                (None, fit_res),
            ],
            failures=[],
        )

from flwr.server.strategy import FedAvg

from iot_fl.flower_framework.strategies.strategy_factory import (
    create_strategy,
)

def test_factory_creates_fedavg():
    strategy = create_strategy(
        aggregation="fedavg",
    )

    assert type(strategy) is FedAvg


def test_factory_creates_failure_aware_v1():
    strategy = create_strategy(
        aggregation="failure_aware_v1",
        alpha=1.0,
    )

    assert isinstance(
        strategy,
        FailureAwareV1Strategy,
    )

    assert strategy.alpha == pytest.approx(1.0)