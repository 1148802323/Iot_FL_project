from flwr.server.strategy import FedAvg

from iot_fl.flower_framework.strategies.failure_aware_v1 import (
    FailureAwareV1Strategy,
)

from iot_fl.flower_framework.strategies.failure_aware_v2 import (
    FailureAwareV2Strategy,
)

from iot_fl.flower_framework.strategies.failure_aware_v4 import (
    FailureAwareV4Strategy,
)

from iot_fl.flower_framework.strategies.strategy_factory import create_strategy


def test_create_fedavg_strategy():
    strategy = create_strategy("fedavg")

    assert isinstance(strategy, FedAvg)


def test_create_failure_aware_v1_strategy():
    strategy = create_strategy("failure_aware_v1")

    assert isinstance(
        strategy,
        FailureAwareV1Strategy,
    )

def test_factory_creates_failure_aware_v2():
    strategy = create_strategy(
        aggregation="failure_aware_v2",
        alpha=1.0,
    )

    assert isinstance(
        strategy,
        FailureAwareV2Strategy,
    )

def test_factory_creates_failure_aware_v4():
    strategy = create_strategy(
        aggregation="failure_aware_v4",
        schedule="linear",
        lambda_max=2.0,
        total_rounds=50,
    )

    assert isinstance(
        strategy,
        FailureAwareV4Strategy,
    )


def test_unsupported_strategy():
    try:
        create_strategy("unknown_strategy")
    except ValueError:
        return

    raise AssertionError("Expected ValueError for unsupported strategy")