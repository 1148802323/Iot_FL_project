from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_failure_aware_fedavg as shared


FEATURES = shared.FEATURES
ID_COL = shared.ID_COL
TARGET = shared.TARGET
STRATEGIES = shared.STRATEGIES
stratified_split = shared.stratified_split


def _local_recall(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Return recall at 0.5, using zero when a client has no positive rows."""
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0
    true_positives = int(np.sum((y_true == 1) & (probability >= 0.5)))
    return true_positives / positives


def performance_aware_strategy(
    strategy: str,
    factory_root: Path,
    train_ids: set[int],
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int,
    local_epochs: int,
    lr: float,
    l2: float,
    failure_lambda: float,
    beta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train V3 with failure-rate and post-training local-recall weighting.

    Each client sends two scalar metrics in addition to its model update. Raw
    records remain local. The server assigns the score

        n_i * (1 + failure_lambda * r_i + beta * recall_i)

    where ``recall_i`` is measured on the client's local training partition
    after the current local update. Global validation and test labels never
    influence aggregation weights.
    """
    if rounds <= 0:
        raise ValueError("rounds must be greater than zero")
    if local_epochs <= 0:
        raise ValueError("local_epochs must be greater than zero")
    if failure_lambda < 0 or beta < 0:
        raise ValueError("failure_lambda and beta must be non-negative")

    clients = shared.load_clients(factory_root / strategy, train_ids)
    global_weights = np.zeros(len(FEATURES) + 1, dtype=float)
    total_rows = sum(int(client["rows"]) for client in clients)
    history_rows: list[dict[str, object]] = []
    final_client_statistics: list[dict[str, object]] = []

    for round_number in range(1, rounds + 1):
        local_updates: list[tuple[np.ndarray, float]] = []
        local_losses: list[float] = []
        local_recalls: list[float] = []
        round_statistics: list[dict[str, object]] = []

        for client in clients:
            x_local = np.asarray(client["x"], dtype=float)
            y_local = np.asarray(client["y"], dtype=int)
            local_weights, local_loss = shared.local_train(
                global_weights,
                x_local,
                y_local,
                lr=lr,
                epochs=local_epochs,
                l2=l2,
            )
            local_recall = _local_recall(
                y_local,
                shared.predict_proba(x_local, local_weights),
            )
            client_rows = int(client["rows"])
            failure_rate = float(client["failure_rate"])
            aggregation_score = client_rows * (
                1.0 + failure_lambda * failure_rate + beta * local_recall
            )
            local_updates.append((local_weights, aggregation_score))
            local_losses.append(float(local_loss))
            local_recalls.append(float(local_recall))
            round_statistics.append(
                {
                    "name": str(client["name"]),
                    "rows": client_rows,
                    "failure_rate": failure_rate,
                    "local_recall": float(local_recall),
                    "aggregation_score": float(aggregation_score),
                }
            )

        total_score = sum(score for _, score in local_updates)
        if total_score <= 0:
            raise ValueError("Performance-aware aggregation produced a non-positive total score")
        global_weights = sum(
            weights * (score / total_score) for weights, score in local_updates
        )

        validation_metrics = shared.metrics(
            y_val,
            shared.predict_proba(x_val, global_weights),
            0.5,
        )
        test_metrics = shared.metrics(
            y_test,
            shared.predict_proba(x_test, global_weights),
            0.5,
        )
        history_rows.append(
            {
                "strategy": strategy,
                "round": round_number,
                "participating_clients": len(clients),
                "local_epochs": local_epochs,
                "client_samples": total_rows,
                "failure_lambda": failure_lambda,
                "beta": beta,
                "mean_local_recall": round(float(np.mean(local_recalls)), 6),
                "min_local_recall": round(float(np.min(local_recalls)), 6),
                "max_local_recall": round(float(np.max(local_recalls)), 6),
                "mean_client_loss": round(float(np.mean(local_losses)), 6),
                "val_accuracy": validation_metrics["accuracy"],
                "val_precision": validation_metrics["precision"],
                "val_recall": validation_metrics["recall"],
                "val_f1": validation_metrics["f1"],
                "test_accuracy_at_0_5": test_metrics["accuracy"],
                "test_precision_at_0_5": test_metrics["precision"],
                "test_recall_at_0_5": test_metrics["recall"],
                "test_f1_at_0_5": test_metrics["f1"],
            }
        )
        final_client_statistics = round_statistics

    validation_probability = shared.predict_proba(x_val, global_weights)
    threshold, threshold_table = shared.tune_threshold(y_val, validation_probability)
    final = shared.metrics(
        y_test,
        shared.predict_proba(x_test, global_weights),
        threshold,
    )
    final.update(
        {
            "strategy": strategy,
            "method": "Performance-Aware FedAvg V3",
            "rounds": rounds,
            "local_epochs": local_epochs,
            "failure_lambda": failure_lambda,
            "beta": beta,
            "clients": len(clients),
            "train_samples": total_rows,
            "communication_client_updates": rounds * len(clients),
            "communication_sample_updates": rounds * total_rows,
        }
    )
    threshold_table.insert(0, "strategy", strategy)
    model_payload = {
        "strategy": strategy,
        "method": "performance_aware_fedavg_v3_weighted_logistic_regression",
        "features": FEATURES,
        "intercept": float(global_weights[0]),
        "coefficients": {
            feature: float(value)
            for feature, value in zip(FEATURES, global_weights[1:])
        },
        "threshold": threshold,
        "failure_lambda": failure_lambda,
        "beta": beta,
        "local_recall_threshold": 0.5,
        "aggregation_formula": (
            "client_samples * (1 + failure_lambda * client_failure_rate "
            "+ beta * client_local_recall)"
        ),
        "clients_at_final_round": final_client_statistics,
    }
    return pd.DataFrame(history_rows), threshold_table, {"final": final, "model": model_payload}


def _comparison_with_fedavg(
    results: pd.DataFrame,
    baseline_path: Path,
) -> pd.DataFrame | None:
    if not baseline_path.exists():
        return None
    baseline = pd.read_csv(baseline_path)
    rows: list[dict[str, object]] = []
    for _, candidate in results.iterrows():
        matched = baseline[baseline["strategy"] == candidate["strategy"]]
        if matched.empty:
            continue
        reference = matched.iloc[0]
        row: dict[str, object] = {"strategy": candidate["strategy"]}
        for metric_name in ("accuracy", "precision", "recall", "f1"):
            baseline_value = float(reference[metric_name])
            candidate_value = float(candidate[metric_name])
            row[f"fedavg_{metric_name}"] = baseline_value
            row[f"performance_aware_{metric_name}"] = candidate_value
            row[f"delta_{metric_name}"] = candidate_value - baseline_value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run V3 Performance-Aware FedAvg across AI4I factory splits."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "processed" / "ai4i_clean_standardized.csv",
    )
    parser.add_argument(
        "--factory-root",
        type=Path,
        default=project_root / "data" / "factories",
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--failure-lambda", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reports = project_root / "reports"
    figures = project_root / "figures"
    processed = project_root / "data" / "processed"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.data)
    missing = [column for column in (ID_COL, TARGET, *FEATURES) if column not in full.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    train_idx, validation_idx, test_idx = shared.stratified_split(
        full[TARGET].to_numpy(dtype=int), 0.6, 0.2, args.seed
    )
    train_ids = set(full.loc[train_idx, ID_COL].astype(int).tolist())
    x_validation = full.loc[validation_idx, FEATURES].to_numpy(dtype=float)
    y_validation = full.loc[validation_idx, TARGET].to_numpy(dtype=int)
    x_test = full.loc[test_idx, FEATURES].to_numpy(dtype=float)
    y_test = full.loc[test_idx, TARGET].to_numpy(dtype=int)

    histories: list[pd.DataFrame] = []
    thresholds: list[pd.DataFrame] = []
    finals: list[dict[str, object]] = []
    models: dict[str, object] = {}
    for strategy in STRATEGIES:
        history, threshold_table, payload = performance_aware_strategy(
            strategy=strategy,
            factory_root=args.factory_root,
            train_ids=train_ids,
            x_val=x_validation,
            y_val=y_validation,
            x_test=x_test,
            y_test=y_test,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            l2=args.l2,
            failure_lambda=args.failure_lambda,
            beta=args.beta,
        )
        histories.append(history)
        thresholds.append(threshold_table)
        finals.append(payload["final"])
        models[strategy] = payload["model"]

    history_frame = pd.concat(histories, ignore_index=True)
    threshold_frame = pd.concat(thresholds, ignore_index=True)
    results_frame = pd.DataFrame(finals)
    history_frame.to_csv(reports / "performance_aware_fedavg_history.csv", index=False)
    threshold_frame.to_csv(
        reports / "performance_aware_fedavg_threshold_tuning.csv", index=False
    )
    results_frame.to_csv(reports / "performance_aware_fedavg_results.csv", index=False)
    (processed / "performance_aware_fedavg_models.json").write_text(
        json.dumps(models, indent=2), encoding="utf-8"
    )

    shared.draw_convergence(
        history_frame,
        figures / "performance_aware_fedavg_convergence.png",
        title="Performance-Aware FedAvg V3 Convergence",
    )
    shared.draw_final_metrics(
        results_frame,
        figures / "performance_aware_fedavg_final_metrics.png",
        title="Performance-Aware FedAvg V3 Final Test Metrics",
    )
    shared.draw_loss_curve(
        history_frame,
        figures / "performance_aware_fedavg_client_loss.png",
        title="Performance-Aware FedAvg V3 Mean Client Loss",
    )
    comparison = _comparison_with_fedavg(
        results_frame, reports / "fedavg_baseline_results.csv"
    )
    if comparison is not None and not comparison.empty:
        comparison.to_csv(
            reports / "performance_aware_vs_fedavg_results.csv", index=False
        )

    print("Performance-Aware FedAvg V3 complete.")
    print(results_frame.to_string(index=False))


if __name__ == "__main__":
    main()