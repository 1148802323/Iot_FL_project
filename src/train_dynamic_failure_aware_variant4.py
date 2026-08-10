from __future__ import annotations

"""Part 4: dynamic failure-aware FedAvg with evaluator-compatible exports.

This is a standalone contribution.  It does not import or modify the existing
FedAvg/V1/V2 implementations.  Three schedules are provided for ablation:
fixed, linear, and recall_adaptive.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Machine failure"
ID_COLUMN = "UDI"
STRATEGIES = ("iid", "moderate_non_iid", "highly_non_iid")
FEATURES = (
    "air_temperature_k_z", "process_temperature_k_z",
    "rotational_speed_rpm_z", "torque_nm_z", "tool_wear_min_z",
    "temperature_gap_k_z", "power_proxy_z",
    "Type_H", "Type_L", "Type_M",
)


def stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in np.unique(y):
        idx = np.where(y == label)[0].copy()
        rng.shuffle(idx)
        n_train = int(round(len(idx) * 0.60))
        n_val = int(round(len(idx) * 0.20))
        parts[0].extend(idx[:n_train])
        parts[1].extend(idx[n_train:n_train + n_val])
        parts[2].extend(idx[n_train + n_val:])
    for part in parts:
        rng.shuffle(part)
    return tuple(np.asarray(part, dtype=int) for part in parts)  # type: ignore[return-value]


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def weighted_loss_gradient(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    probability = sigmoid(x @ weights)
    positive_weight = max(float((y == 0).sum()) / max(int((y == 1).sum()), 1), 1.0)
    sample_weight = np.where(y == 1, positive_weight, 1.0)
    error = (probability - y) * sample_weight
    gradient = x.T @ error / sample_weight.sum()
    gradient[1:] += l2 * weights[1:]
    eps = 1e-12
    loss = -np.average(
        y * np.log(probability + eps) + (1 - y) * np.log(1 - probability + eps),
        weights=sample_weight,
    ) + 0.5 * l2 * float(weights[1:] @ weights[1:])
    return float(loss), gradient


def local_train(
    x: np.ndarray, y: np.ndarray, initial: np.ndarray, epochs: int, lr: float, l2: float
) -> tuple[np.ndarray, float]:
    weights = initial.copy()
    loss = 0.0
    for _ in range(epochs):
        loss, gradient = weighted_loss_gradient(x, y, weights, l2)
        weights -= lr * gradient
    return weights, loss


def recall_at_half(y: np.ndarray, probability: np.ndarray) -> float:
    predicted = probability >= 0.5
    tp = int(((y == 1) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    return tp / max(tp + fn, 1)


def schedule_lambda(
    mode: str, round_number: int, rounds: int, lambda_max: float,
    previous: float, validation_recall: float, target_recall: float, eta: float,
) -> float:
    if mode == "fixed":
        return lambda_max
    if mode == "linear":
        return lambda_max * round_number / max(rounds, 1)
    if mode == "recall_adaptive":
        return float(np.clip(previous + eta * (target_recall - validation_recall), 0.0, lambda_max))
    raise ValueError(f"Unknown schedule: {mode}")


def load_clients(directory: Path, train_ids: set[int]) -> list[dict[str, object]]:
    clients = []
    for path in sorted(directory.glob("factory_*.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame[ID_COLUMN].astype(int).isin(train_ids)]
        if frame.empty:
            continue
        x = add_intercept(frame.loc[:, FEATURES].to_numpy(dtype=float))
        y = frame[TARGET].to_numpy(dtype=int)
        clients.append({
            "name": path.stem, "x": x, "y": y, "samples": len(y),
            "failures": int(y.sum()),
        })
    if not clients:
        raise ValueError(f"No training clients found in {directory}")
    return clients


def train_one(
    full: pd.DataFrame, factory_root: Path, strategy: str, seed: int,
    rounds: int, local_epochs: int, lr: float, l2: float, schedule: str,
    lambda_max: float, target_recall: float, eta: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    train_idx, validation_idx, test_idx = stratified_split(full[TARGET].to_numpy(dtype=int), seed)
    train_ids = set(full.iloc[train_idx][ID_COLUMN].astype(int))
    clients = load_clients(factory_root / strategy, train_ids)
    x_val = add_intercept(full.iloc[validation_idx].loc[:, FEATURES].to_numpy(dtype=float))
    y_val = full.iloc[validation_idx][TARGET].to_numpy(dtype=int)
    x_test = add_intercept(full.iloc[test_idx].loc[:, FEATURES].to_numpy(dtype=float))
    weights = np.zeros(x_val.shape[1], dtype=float)
    current_lambda = 0.0 if schedule != "fixed" else lambda_max
    history: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for round_number in range(1, rounds + 1):
        recall_before = recall_at_half(y_val, sigmoid(x_val @ weights))
        current_lambda = schedule_lambda(
            schedule, round_number, rounds, lambda_max, current_lambda,
            recall_before, target_recall, eta,
        )
        local_models, aggregation_scores, losses = [], [], []
        for client in clients:
            local, loss = local_train(
                client["x"], client["y"], weights, local_epochs, lr, l2  # type: ignore[arg-type]
            )
            failures = int(client["failures"])
            samples = int(client["samples"])
            failure_rate = failures / max(samples, 1)
            score = samples * (1.0 + current_lambda * failure_rate)
            local_models.append(local)
            aggregation_scores.append(score)
            losses.append(loss)
        normalised = np.asarray(aggregation_scores) / np.sum(aggregation_scores)
        weights = np.sum(np.asarray(local_models) * normalised[:, None], axis=0)
        val_probability = sigmoid(x_val @ weights)
        val_recall = recall_at_half(y_val, val_probability)
        history.append({
            "seed": seed, "strategy": strategy, "round": round_number,
            "schedule": schedule, "lambda": current_lambda,
            "validation_recall_at_0_5": val_recall,
            "mean_client_loss": float(np.mean(losses)),
        })
        for row_index, probability in zip(validation_idx, val_probability):
            prediction_rows.append({
                "UDI": int(full.iloc[row_index][ID_COLUMN]), "seed": seed,
                "strategy": strategy, "split": "validation",
                "probability": float(probability), "round": round_number,
            })

    for split, indices, matrix in (
        ("validation", validation_idx, x_val), ("test", test_idx, x_test)
    ):
        for row_index, probability in zip(indices, sigmoid(matrix @ weights)):
            prediction_rows.append({
                "UDI": int(full.iloc[row_index][ID_COLUMN]), "seed": seed,
                "strategy": strategy, "split": split,
                "probability": float(probability), "round": 0,
            })
    model = {
        "seed": seed, "strategy": strategy, "schedule": schedule,
        "lambda_final": current_lambda, "lambda_max": lambda_max,
        "target_recall": target_recall, "eta": eta,
        "intercept": float(weights[0]),
        "coefficients": {name: float(value) for name, value in zip(FEATURES, weights[1:])},
    }
    return prediction_rows, history, model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train Part 4 dynamic failure-aware FedAvg.")
    parser.add_argument("--data", type=Path, default=root / "data/processed/ai4i_clean_standardized.csv")
    parser.add_argument("--factory-root", type=Path, default=root / "data/factories")
    parser.add_argument("--output-dir", type=Path, default=root / "part4_outputs")
    parser.add_argument("--seeds", default="42,52,62,72,82")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--schedule", choices=("fixed", "linear", "recall_adaptive"), default="linear")
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--eta", type=float, default=0.25)
    args = parser.parse_args()

    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    strategies = tuple(item.strip() for item in args.strategies.split(",") if item.strip())
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    full = pd.read_csv(args.data)
    missing = [name for name in (ID_COLUMN, TARGET, *FEATURES) if name not in full.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    all_predictions, all_history, models = [], [], []
    for seed in seeds:
        for strategy in strategies:
            predictions, history, model = train_one(
                full, args.factory_root, strategy, seed, args.rounds, args.local_epochs,
                args.lr, args.l2, args.schedule, args.lambda_max, args.target_recall, args.eta,
            )
            all_predictions.extend(predictions)
            all_history.extend(history)
            models.append(model)
            print(f"complete seed={seed} strategy={strategy} schedule={args.schedule}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_predictions).to_csv(args.output_dir / "part4_predictions.csv", index=False)
    pd.DataFrame(all_history).to_csv(args.output_dir / "part4_training_history.csv", index=False)
    (args.output_dir / "part4_models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
    manifest = {
        "algorithm": "dynamic_failure_aware_fedavg_part4", "seeds": seeds,
        "strategies": strategies, "rounds": args.rounds, "local_epochs": args.local_epochs,
        "learning_rate": args.lr, "l2": args.l2, "schedule": args.schedule,
        "lambda_max": args.lambda_max, "target_recall": args.target_recall, "eta": args.eta,
    }
    (args.output_dir / "part4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
