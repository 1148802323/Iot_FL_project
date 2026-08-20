from __future__ import annotations

import pytest

from iot_fl.algorithms import get_algorithm, list_algorithms, run_experiment
from iot_fl.algorithms.base import STANDARD_RESULT_KEYS, FederatedAlgorithm
from iot_fl.algorithms.registry import ALGORITHM_REGISTRY


EXPECTED_ALGORITHMS = {
    "fedavg",
    "failure_aware_v1",
    "failure_aware_v2",
    "performance_aware_v3",
    "dynamic_failure_aware",
}


def test_all_algorithms_are_registered() -> None:
    assert set(list_algorithms()) == EXPECTED_ALGORITHMS


def test_invalid_algorithm_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown algorithm 'missing'"):
        get_algorithm("missing")


def test_invalid_distribution_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown distribution 'bad_split'"):
        run_experiment("fedavg", "bad_split", {"rounds": 1})


def test_each_adapter_exposes_same_interface() -> None:
    for adapter in ALGORITHM_REGISTRY.values():
        assert isinstance(adapter, FederatedAlgorithm)
        assert callable(adapter.run)
        assert isinstance(adapter.name, str)
        assert adapter.implementation_file.startswith("src/")


@pytest.mark.parametrize("algorithm", sorted(EXPECTED_ALGORITHMS))
def test_standard_result_schema_is_respected(monkeypatch: pytest.MonkeyPatch, algorithm: str) -> None:
    adapter = ALGORITHM_REGISTRY[algorithm]

    def fake_run(distribution: str, config: dict[str, object] | None = None) -> dict[str, object]:
        del config
        return {
            "algorithm": adapter.name,
            "distribution": distribution,
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "communication_cost": 5,
            "training_time": 0.01,
            "rounds": 1,
            "convergence_history": [{"round": 1, "val_f1": 0.75}],
        }

    monkeypatch.setattr(adapter, "run", fake_run)
    result = run_experiment(algorithm, "iid", {"rounds": 1, "local_epochs": 1})

    assert set(STANDARD_RESULT_KEYS).issubset(result)
    assert result["algorithm"] == algorithm
    assert result["distribution"] == "iid"
    assert isinstance(result["convergence_history"], list)


def test_fedavg_adapter_can_run_one_lightweight_round() -> None:
    result = run_experiment(
        algorithm="fedavg",
        distribution="iid",
        config={"rounds": 1, "local_epochs": 1},
    )

    assert set(STANDARD_RESULT_KEYS).issubset(result)
    assert result["algorithm"] == "fedavg"
    assert result["distribution"] == "iid"
    assert result["rounds"] == 1
    assert isinstance(result["convergence_history"], list)
    assert len(result["convergence_history"]) == 1
    assert result["accuracy"] is not None


def test_performance_aware_v3_can_run_one_lightweight_round() -> None:
    result = run_experiment(
        algorithm="performance_aware_v3",
        distribution="highly_non_iid",
        config={
            "rounds": 1,
            "local_epochs": 1,
            "failure_lambda": 1.0,
            "beta": 1.0,
        },
    )

    assert set(STANDARD_RESULT_KEYS).issubset(result)
    assert result["algorithm"] == "performance_aware_v3"
    assert result["distribution"] == "highly_non_iid"
    assert result["rounds"] == 1
    assert len(result["convergence_history"]) == 1
    assert result["model"]["beta"] == 1.0
