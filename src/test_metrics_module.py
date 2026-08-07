from __future__ import annotations

import numpy as np

from iot_fl.metrics import (
    classification_metrics,
    tune_threshold,
)


def main() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)

    probabilities = np.array(
        [0.10, 0.70, 0.80, 0.40],
        dtype=float,
    )

    result = classification_metrics(
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.5,
    )

    print("Metrics at threshold 0.5:")
    print(result)

    assert result["tp"] == 1
    assert result["tn"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1

    assert np.isclose(result["accuracy"], 0.5)
    assert np.isclose(result["precision"], 0.5)
    assert np.isclose(result["recall"], 0.5)
    assert np.isclose(result["f1"], 0.5)

    best_threshold, threshold_results = tune_threshold(
        y_true=y_true,
        probabilities=probabilities,
    )

    print("\nBest threshold:", best_threshold)
    print(
        "Number of thresholds tested:",
        len(threshold_results),
    )

    best_result = threshold_results.sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    ).iloc[0]

    print("Best result:")
    print(best_result)

    assert 0.05 <= best_threshold <= 0.95
    assert len(threshold_results) == 91
    assert best_result["f1"] >= result["f1"]

    print("\nMetrics module test passed.")


if __name__ == "__main__":
    main()