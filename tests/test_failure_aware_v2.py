import pytest

from iot_fl.flower_framework.strategies.failure_aware_v2 import (
    FailureAwareV2Strategy,
)

def test_v2_alpha_zero_matches_fedavg_weight():
    strategy = FailureAwareV2Strategy(alpha=0.0)

    weight = strategy._calculate_client_weight(
        server_round=1,
        num_examples=100,
        client_failure_rate=0.30,
        global_failure_rate=0.20,
    )

    assert weight == pytest.approx(100.0)


def test_v2_failure_count_weight():
    strategy = FailureAwareV2Strategy(alpha=2.0)

    weight = strategy._calculate_client_weight(
        server_round=1,
        num_examples=100,
        client_failure_rate=0.30,
        global_failure_rate=0.20,
    )

    assert weight == pytest.approx(160.0)

def test_v2_rejects_negative_alpha():
    with pytest.raises(
        ValueError,
        match="alpha must be greater than or equal to zero",
    ):
        FailureAwareV2Strategy(alpha=-1.0)