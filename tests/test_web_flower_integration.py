from iot_fl.algorithms.adapters.failure_aware_v1 import FailureAwareV1Adapter
from iot_fl.algorithms.base import ExperimentConfig


def test_v1_alpha_reaches_flower_adapter(monkeypatch):
    captured = {}

    def fake_run_flower_experiment(**kwargs):
        captured.update(kwargs)

        return {
            "algorithm": "failure_aware_v1",
            "distribution": "iid",
            "final": {},
            "history": [],
        }

    monkeypatch.setattr(
        "iot_fl.algorithms.adapters.failure_aware_v1.run_flower_experiment",
        fake_run_flower_experiment,
    )

    config = ExperimentConfig.from_mapping(
        {
            "rounds": 1,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "alpha": 1.5,
        }
    )

    adapter = FailureAwareV1Adapter()

    adapter._run(
        distribution="iid",
        config=config,
    )

    assert config.extra["alpha"] == 1.5
    assert captured["aggregation"] == "failure_aware_v1"
    assert captured["alpha"] == 1.5

from iot_fl.algorithms.adapters.failure_aware_v2 import FailureAwareV2Adapter


def test_v2_alpha_reaches_flower_adapter(monkeypatch):
    captured = {}

    def fake_run_flower_experiment(**kwargs):
        captured.update(kwargs)

        return {
            "algorithm": "failure_aware_v2",
            "distribution": "iid",
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "communication_cost": 0,
            "rounds": 1,
            "convergence_history": [],
        }

    monkeypatch.setattr(
        "iot_fl.algorithms.adapters.failure_aware_v2.run_flower_experiment",
        fake_run_flower_experiment,
    )

    config = ExperimentConfig.from_mapping(
        {
            "rounds": 1,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "alpha": 1.5,
        }
    )

    adapter = FailureAwareV2Adapter()

    adapter._run(
        distribution="iid",
        config=config,
    )

    assert config.extra["alpha"] == 1.5
    assert captured["aggregation"] == "failure_aware_v2"
    assert captured["v2_alpha"] == 1.5


from iot_fl.algorithms.adapters.dynamic_failure_aware import (
    DynamicFailureAwareAdapter,
)


def test_v4_parameters_reach_flower_adapter(monkeypatch):
    captured = {}

    def fake_run_flower_experiment(**kwargs):
        captured.update(kwargs)

        return {
            "algorithm": "failure_aware_v4",
            "distribution": "iid",
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "communication_cost": 0,
            "rounds": 2,
            "convergence_history": [],
        }

    monkeypatch.setattr(
        "iot_fl.algorithms.adapters.dynamic_failure_aware.run_flower_experiment",
        fake_run_flower_experiment,
    )

    config = ExperimentConfig.from_mapping(
        {
            "rounds": 2,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "schedule": "linear",
            "lambda_max": 2.0,
            "target_recall": 0.85,
            "eta": 0.25,
        }
    )

    adapter = DynamicFailureAwareAdapter()

    adapter._run(
        distribution="iid",
        config=config,
    )

    assert config.extra["schedule"] == "linear"
    assert config.extra["lambda_max"] == 2.0
    assert config.extra["target_recall"] == 0.85
    assert config.extra["eta"] == 0.25

    assert captured["aggregation"] == "failure_aware_v4"
    assert captured["v4_schedule"] == "linear"
    assert captured["v4_lambda_max"] == 2.0
    assert captured["v4_target_recall"] == 0.85
    assert captured["v4_eta"] == 0.25