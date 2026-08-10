from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate binary classification metrics."""
    predictions = (probabilities >= threshold).astype(int)

    tp = int(((y_true == 1) & (predictions == 1)).sum())
    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())

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
        "positive_predictions": int(predictions.sum()),
    }


def tune_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Search thresholds from 0.05 to 0.95 and maximise F1."""
    rows = [
        classification_metrics(
            y_true=y_true,
            probabilities=probabilities,
            threshold=float(threshold),
        )
        for threshold in np.linspace(0.05, 0.95, 91)
    ]

    table = pd.DataFrame(rows)

    best = table.sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    ).iloc[0]

    return float(best["threshold"]), table