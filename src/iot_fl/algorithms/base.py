from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from iot_fl.config import FEATURES, ID_COL, STRATEGIES, TARGET


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_DISTRIBUTIONS = tuple(STRATEGIES)
STANDARD_RESULT_KEYS = (
    "algorithm",
    "distribution",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "communication_cost",
    "training_time",
    "rounds",
    "convergence_history",
)


@dataclass(frozen=True)
class ExperimentConfig:
    rounds: int = 50
    local_epochs: int = 5
    learning_rate: float = 0.08
    seed: int = 42
    l2: float = 0.001
    data_path: Path = PROJECT_ROOT / "data" / "processed" / "ai4i_clean_standardized.csv"
    factory_root: Path = PROJECT_ROOT / "data" / "factories"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None = None) -> "ExperimentConfig":
        raw = dict(values or {})
        if "lr" in raw and "learning_rate" not in raw:
            raw["learning_rate"] = raw.pop("lr")

        known_fields = {
            "rounds",
            "local_epochs",
            "learning_rate",
            "seed",
            "l2",
            "data_path",
            "factory_root",
        }
        known = {key: raw.pop(key) for key in list(raw) if key in known_fields}
        if "data_path" in known:
            known["data_path"] = Path(known["data_path"])
        if "factory_root" in known:
            known["factory_root"] = Path(known["factory_root"])
        return cls(**known, extra=raw)


class FederatedAlgorithm(ABC):
    name: str
    display_name: str
    implementation_file: str

    def run(self, distribution: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        validate_distribution(distribution)
        experiment_config = ExperimentConfig.from_mapping(config)
        start = perf_counter()
        result = self._run(distribution, experiment_config)
        result["training_time"] = result.get("training_time") or round(perf_counter() - start, 6)
        return ensure_standard_result(result)

    @abstractmethod
    def _run(self, distribution: str, config: ExperimentConfig) -> dict[str, Any]:
        raise NotImplementedError


def validate_distribution(distribution: str) -> None:
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        supported = ", ".join(SUPPORTED_DISTRIBUTIONS)
        raise ValueError(f"Unknown distribution '{distribution}'. Supported distributions: {supported}")


def load_ai4i_frame(data_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    missing = [name for name in (ID_COL, TARGET, *FEATURES) if name not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame


def shared_train_validation_test(
    module: Any,
    config: ExperimentConfig,
) -> tuple[set[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = load_ai4i_frame(config.data_path)
    train_idx, validation_idx, test_idx = module.stratified_split(
        frame[TARGET].to_numpy(dtype=int),
        0.6,
        0.2,
        config.seed,
    )
    train_ids = set(frame.loc[train_idx, ID_COL].astype(int).tolist())
    x_validation = frame.loc[validation_idx, FEATURES].to_numpy(dtype=float)
    y_validation = frame.loc[validation_idx, TARGET].to_numpy(dtype=int)
    x_test = frame.loc[test_idx, FEATURES].to_numpy(dtype=float)
    y_test = frame.loc[test_idx, TARGET].to_numpy(dtype=int)
    return train_ids, x_validation, y_validation, x_test, y_test


def normalize_strategy_payload(
    *,
    algorithm: str,
    distribution: str,
    final: dict[str, Any],
    history: pd.DataFrame,
    training_time: float | None = None,
) -> dict[str, Any]:
    communication_updates = final.get("communication_client_updates")
    return ensure_standard_result(
        {
            "algorithm": algorithm,
            "distribution": distribution,
            "accuracy": final.get("accuracy"),
            "precision": final.get("precision"),
            "recall": final.get("recall"),
            "f1": final.get("f1"),
            "communication_cost": communication_updates,
            "training_time": training_time,
            "rounds": final.get("rounds"),
            "convergence_history": dataframe_records(history),
            "method": final.get("method"),
            "threshold": final.get("threshold"),
            "communication_client_updates": communication_updates,
            "communication_sample_updates": final.get("communication_sample_updates"),
            "raw_final": to_jsonable(final),
        }
    )


def ensure_standard_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    for key in STANDARD_RESULT_KEYS:
        normalized.setdefault(key, None)
    normalized["convergence_history"] = normalized["convergence_history"] or []
    return to_jsonable(normalized)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return to_jsonable(frame.to_dict(orient="records"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = probability >= threshold
    tp = int(((y_true == 1) & prediction).sum())
    tn = int(((y_true == 0) & ~prediction).sum())
    fp = int(((y_true == 0) & prediction).sum())
    fn = int(((y_true == 1) & ~prediction).sum())
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_predictions": int(prediction.sum()),
    }


def tune_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    rows = [
        classification_metrics(y_true, probability, float(threshold))
        for threshold in np.linspace(0.05, 0.95, 91)
    ]
    table = pd.DataFrame(rows)
    best = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"])

