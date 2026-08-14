from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from iot_fl.algorithms.base import (
    ExperimentConfig,
    FederatedAlgorithm,
    classification_metrics,
    dataframe_records,
    load_ai4i_frame,
    tune_threshold,
    ensure_standard_result,
)
from iot_fl.config import ID_COL, TARGET

import train_dynamic_failure_aware_variant4 as implementation


class DynamicFailureAwareAdapter(FederatedAlgorithm):
    name = "dynamic_failure_aware"
    display_name = "Dynamic Failure-Aware FedAvg"
    implementation_file = "src/train_dynamic_failure_aware_variant4.py"

    def _run(self, distribution: str, config: ExperimentConfig) -> dict[str, Any]:
        frame = load_ai4i_frame(config.data_path)
        train_idx, validation_idx, test_idx = implementation.stratified_split(
            frame[TARGET].to_numpy(dtype=int),
            config.seed,
        )
        train_ids = set(frame.iloc[train_idx][ID_COL].astype(int))
        predictions, history, model = implementation.train_one(
            full=frame,
            factory_root=config.factory_root,
            strategy=distribution,
            seed=config.seed,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            lr=config.learning_rate,
            l2=config.l2,
            schedule=str(config.extra.get("schedule", "linear")),
            lambda_max=float(config.extra.get("lambda_max", 2.0)),
            target_recall=float(config.extra.get("target_recall", 0.85)),
            eta=float(config.extra.get("eta", 0.25)),
        )

        prediction_frame = pd.DataFrame(predictions)
        final_rows = prediction_frame[prediction_frame["round"] == 0]
        validation_probability = aligned_probability(
            final_rows[final_rows["split"] == "validation"],
            frame.iloc[validation_idx],
        )
        test_probability = aligned_probability(
            final_rows[final_rows["split"] == "test"],
            frame.iloc[test_idx],
        )
        y_validation = frame.iloc[validation_idx][TARGET].to_numpy(dtype=int)
        y_test = frame.iloc[test_idx][TARGET].to_numpy(dtype=int)
        threshold = tune_threshold(y_validation, validation_probability)
        final = classification_metrics(y_test, test_probability, threshold)

        clients = implementation.load_clients(config.factory_root / distribution, train_ids)
        communication_client_updates = config.rounds * len(clients)
        communication_sample_updates = config.rounds * sum(int(client["samples"]) for client in clients)
        final.update(
            {
                "method": "Dynamic Failure-Aware FedAvg",
                "rounds": config.rounds,
                "local_epochs": config.local_epochs,
                "communication_client_updates": communication_client_updates,
                "communication_sample_updates": communication_sample_updates,
            }
        )

        return ensure_standard_result(
            {
                "algorithm": self.name,
                "distribution": distribution,
                "accuracy": final["accuracy"],
                "precision": final["precision"],
                "recall": final["recall"],
                "f1": final["f1"],
                "communication_cost": communication_client_updates,
                "training_time": None,
                "rounds": config.rounds,
                "convergence_history": dataframe_records(pd.DataFrame(history)),
                "method": final["method"],
                "threshold": final["threshold"],
                "communication_client_updates": communication_client_updates,
                "communication_sample_updates": communication_sample_updates,
                "raw_final": final,
                "model": model,
            }
        )


def aligned_probability(submitted: pd.DataFrame, expected: pd.DataFrame) -> np.ndarray:
    if submitted[ID_COL].duplicated().any():
        duplicates = submitted.loc[submitted[ID_COL].duplicated(), ID_COL].head().tolist()
        raise ValueError(f"Duplicate prediction rows for UDI values: {duplicates}")

    expected_ids = set(expected[ID_COL].astype(int))
    submitted_ids = set(submitted[ID_COL].astype(int))
    missing = expected_ids - submitted_ids
    extra = submitted_ids - expected_ids
    if missing or extra:
        raise ValueError(f"Prediction UDI mismatch; missing={len(missing)}, extra={len(extra)}")

    merged = expected[[ID_COL]].merge(
        submitted[[ID_COL, "probability"]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    return merged["probability"].to_numpy(dtype=float)

