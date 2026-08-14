import pytest

from iot_fl.flower_framework.strategies.failure_aware_v4 import (
    FailureAwareV4Strategy,
)

def test_v4_fixed_schedule_uses_lambda_max():
    strategy = FailureAwareV4Strategy(
        schedule="fixed",
        lambda_max=2.0,
        total_rounds=50,
    )

    strategy._prepare_round(
        server_round=10,
        global_failure_rate=0.2,
    )

    assert strategy.current_lambda == pytest.approx(2.0)

def test_v4_linear_schedule_updates_lambda():
    strategy = FailureAwareV4Strategy(
        schedule="linear",
        lambda_max=2.0,
        total_rounds=50,
    )

    strategy._prepare_round(
        server_round=10,
        global_failure_rate=0.2,
    )

    assert strategy.current_lambda == pytest.approx(0.4)

def test_v4_failure_aware_weight():
    strategy = FailureAwareV4Strategy(
        schedule="fixed",
        lambda_max=2.0,
        total_rounds=50,
    )

    strategy._prepare_round(
        server_round=1,
        global_failure_rate=0.2,
    )

    weight = strategy._calculate_client_weight(
        server_round=1,
        num_examples=100,
        client_failure_rate=0.30,
        global_failure_rate=0.20,
    )

    assert weight == pytest.approx(160.0)

def test_v4_lambda_zero_matches_fedavg_weight():
    strategy = FailureAwareV4Strategy(
        schedule="fixed",
        lambda_max=0.0,
        total_rounds=50,
    )

    strategy._prepare_round(
        server_round=1,
        global_failure_rate=0.2,
    )

    weight = strategy._calculate_client_weight(
        server_round=1,
        num_examples=100,
        client_failure_rate=0.30,
        global_failure_rate=0.20,
    )

    assert weight == pytest.approx(100.0)

def test_v4_rejects_invalid_schedule():
    with pytest.raises(
        ValueError,
        match="schedule",
    ):
        FailureAwareV4Strategy(
            schedule="unknown",
            lambda_max=2.0,
            total_rounds=50,
        )


def test_v4_rejects_negative_lambda_max():
    with pytest.raises(
        ValueError,
        match="lambda_max",
    ):
        FailureAwareV4Strategy(
            schedule="fixed",
            lambda_max=-1.0,
            total_rounds=50,
        )


def test_v4_rejects_invalid_total_rounds():
    with pytest.raises(
        ValueError,
        match="total_rounds",
    ):
        FailureAwareV4Strategy(
            schedule="linear",
            lambda_max=2.0,
            total_rounds=0,
        )

def test_v4_recall_adaptive_increases_lambda_when_recall_below_target():
    strategy = FailureAwareV4Strategy(
        schedule="recall_adaptive",
        lambda_max=2.0,
        total_rounds=50,
        target_recall=0.85,
        eta=0.25,
    )

    strategy.latest_validation_recall = 0.65

    strategy._prepare_round(
        server_round=2,
        global_failure_rate=0.2,
    )

    assert strategy.current_lambda == pytest.approx(0.05)

def test_v4_recall_adaptive_decreases_lambda_when_recall_above_target():
    strategy = FailureAwareV4Strategy(
        schedule="recall_adaptive",
        lambda_max=2.0,
        total_rounds=50,
        target_recall=0.85,
        eta=0.25,
    )

    strategy.current_lambda = 0.50
    strategy.latest_validation_recall = 0.95

    strategy._prepare_round(
        server_round=2,
        global_failure_rate=0.2,
    )

    assert strategy.current_lambda == pytest.approx(0.475)

def test_v4_recall_adaptive_clips_lambda_to_maximum():
    strategy = FailureAwareV4Strategy(
        schedule="recall_adaptive",
        lambda_max=1.0,
        total_rounds=50,
        target_recall=0.90,
        eta=1.0,
    )

    strategy.current_lambda = 0.95
    strategy.latest_validation_recall = 0.0

    strategy._prepare_round(
        server_round=2,
        global_failure_rate=0.2,
    )

    assert strategy.current_lambda == pytest.approx(1.0)

def test_v4_recall_adaptive_clips_lambda_to_zero():
    strategy = FailureAwareV4Strategy(
        schedule="recall_adaptive",
        lambda_max=2.0,
        total_rounds=50,
        target_recall=0.50,
        eta=1.0,
    )

    strategy.current_lambda = 0.10
    strategy.latest_validation_recall = 1.0

    strategy._prepare_round(
        server_round=2,
        global_failure_rate=0.2,
    )

    assert strategy.current_lambda == pytest.approx(0.0)