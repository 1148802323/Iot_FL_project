from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TARGET = "Machine failure"
ID_COL = "UDI"
FEATURES = [
    "air_temperature_k_z",
    "process_temperature_k_z",
    "rotational_speed_rpm_z",
    "torque_nm_z",
    "tool_wear_min_z",
    "temperature_gap_k_z",
    "power_proxy_z",
    "Type_H",
    "Type_L",
    "Type_M",
]
STRATEGIES = ["iid", "moderate_non_iid", "highly_non_iid"]
COLORS = {
    "iid": (42, 157, 143),
    "moderate_non_iid": (225, 111, 86),
    "highly_non_iid": (130, 90, 160),
    "ink": (28, 35, 39),
    "muted": (88, 96, 102),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def stratified_split(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0].copy()
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_ratio))
        n_val = int(round(len(idx) * val_ratio))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())
    for arr in (train_idx, val_idx, test_idx):
        rng.shuffle(arr)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def sample_weights(y: np.ndarray) -> np.ndarray:
    pos = float(y.sum())
    neg = float((y == 0).sum())
    if pos == 0 or neg == 0:
        return np.ones(len(y), dtype=float)
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))


def weighted_log_loss(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> float:
    eps = 1e-8
    loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return float(np.average(loss, weights=weights))


def local_train(
    initial: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
) -> tuple[np.ndarray, float]:
    xb = add_intercept(x)
    w = initial.copy()
    sw = sample_weights(y)
    for _ in range(epochs):
        p = sigmoid(xb @ w)
        grad = (xb.T @ ((p - y) * sw)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    final_loss = weighted_log_loss(y, sigmoid(xb @ w), sw)
    return w, final_loss


def predict_proba(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return sigmoid(add_intercept(x) @ weights)


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
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
        "positive_predictions": int(y_pred.sum()),
    }


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = [metrics(y_true, y_prob, float(t)) for t in np.linspace(0.05, 0.95, 91)]
    table = pd.DataFrame(rows)
    best = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def load_clients(strategy_dir: Path, train_ids: set[int]) -> list[dict[str, object]]:
    clients = []
    for path in sorted(strategy_dir.glob("factory_*.csv")):
        df = pd.read_csv(path)
        df = df[df[ID_COL].isin(train_ids)].copy()
        if df.empty:
            continue
        clients.append(
            {
                "name": path.stem,
                "rows": len(df),
                "failure_count": int(df[TARGET].sum()),
                "x": df[FEATURES].to_numpy(dtype=float),
                "y": df[TARGET].to_numpy(dtype=int),
            }
        )
    if not clients:
        raise ValueError(f"No training clients found in {strategy_dir}")
    return clients


def failure_count_strategy(
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
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    clients = load_clients(factory_root / strategy, train_ids)
    global_w = np.zeros(len(FEATURES) + 1, dtype=float)
    total_rows = sum(int(c["rows"]) for c in clients)
    total_failures = sum(int(c["failure_count"]) for c in clients)
    global_failure_count = total_failures
    history_rows = []

    for round_num in range(1, rounds + 1):
        client_weights = []
        client_losses = []
        for client in clients:
            w, loss = local_train(
                global_w,
                client["x"],  # type: ignore[arg-type]
                client["y"],  # type: ignore[arg-type]
                lr=lr,
                epochs=local_epochs,
                l2=l2,
            )
            client_rows = int(client["rows"])
            client_failure_count = int(client["failure_count"])
            failure_bonus = alpha * client_failure_count
            aggregation_weight = (
                client_rows
                + failure_bonus
            )
            client_weights.append((w, aggregation_weight))
            client_losses.append(loss)
        total_aggregation_weight = sum(weight for _, weight in client_weights)
        global_w = sum(w * (weight / total_aggregation_weight) for w, weight in client_weights)

        val_prob = predict_proba(x_val, global_w)
        test_prob = predict_proba(x_test, global_w)
        val_metrics = metrics(y_val, val_prob, 0.5)
        test_metrics = metrics(y_test, test_prob, 0.5)
        history_rows.append(
            {
                "strategy": strategy,
                "round": round_num,
                "participating_clients": len(clients),
                "local_epochs": local_epochs,
                "client_samples": total_rows,
                "alpha": alpha,
                "global_failure_count": int(global_failure_count),
                "mean_client_loss": round(float(np.mean(client_losses)), 6),
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "test_accuracy_at_0_5": test_metrics["accuracy"],
                "test_precision_at_0_5": test_metrics["precision"],
                "test_recall_at_0_5": test_metrics["recall"],
                "test_f1_at_0_5": test_metrics["f1"],
            }
        )

    val_prob = predict_proba(x_val, global_w)
    threshold, threshold_table = tune_threshold(y_val, val_prob)
    test_prob = predict_proba(x_test, global_w)
    final = metrics(y_test, test_prob, threshold)
    final.update(
        {
            "strategy": strategy,
            "method": "Failure-Count FedAvg",
            "rounds": rounds,
            "local_epochs": local_epochs,
            "alpha": alpha,
            "clients": len(clients),
            "train_samples": total_rows,
            "communication_client_updates": rounds * len(clients),
            "communication_sample_updates": rounds * total_rows,
        }
    )
    threshold_table.insert(0, "strategy", strategy)
    model_payload = {
        "strategy": strategy,
        "method": "failure_count_fedavg_weighted_logistic_regression",
        "features": FEATURES,
        "intercept": float(global_w[0]),
        "coefficients": {feature: float(value) for feature, value in zip(FEATURES, global_w[1:])},
        "threshold": threshold,
        "alpha": alpha,
        "global_failure_count": int(global_failure_count),
        "aggregation_formula": "client_samples + alpha * client_failure_count",
        "clients": [
            {"name": str(c["name"]), "rows": int(c["rows"]), "failure_count": int(c["failure_count"])}
            for c in clients
        ],
    }
    return pd.DataFrame(history_rows), threshold_table, {"final": final, "model": model_payload}


def draw_convergence(history: pd.DataFrame, out_path: Path) -> None:
    width, height = 1250, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((75, 35), "Failure-Count FedAvg Convergence", fill=COLORS["ink"], font=font(34, True))
    left, right, top, bottom = 115, 1140, 130, 610
    draw.line((left, top, left, bottom, right, bottom), fill=COLORS["muted"], width=2)
    draw.text((25, top + 25), "Val F1", fill=COLORS["ink"], font=font(18))
    draw.text((right - 70, bottom + 25), "Round", fill=COLORS["ink"], font=font(18))
    all_rounds = history["round"].to_numpy(dtype=float)
    max_y = max(float(history["val_f1"].max()), 0.05)
    for tick in np.linspace(0, max_y, 6):
        py = bottom - (tick / max_y) * (bottom - top - 25)
        draw.line((left - 5, py, right, py), fill=(225, 229, 232), width=1)
        draw.text((48, py - 9), f"{tick:.3f}", fill=COLORS["muted"], font=font(14))
    for tick in np.linspace(all_rounds.min(), all_rounds.max(), 6):
        px = left + (tick - all_rounds.min()) / max(all_rounds.max() - all_rounds.min(), 1e-6) * (right - left)
        draw.line((px, bottom, px, bottom + 5), fill=COLORS["muted"], width=1)
        draw.text((px - 15, bottom + 12), f"{int(round(tick))}", fill=COLORS["muted"], font=font(14))
    for label_index, strategy in enumerate(STRATEGIES):
        sub = history[history["strategy"] == strategy]
        if sub.empty:
            continue
        xs = sub["round"].to_numpy(dtype=float)
        ys = sub["val_f1"].to_numpy(dtype=float)
        points = []
        for x, y in zip(xs, ys):
            px = left + (x - xs.min()) / max(xs.max() - xs.min(), 1e-6) * (right - left)
            py = bottom - (y / max_y) * (bottom - top - 25)
            points.append((px, py))
        draw.line(points, fill=COLORS[strategy], width=4)
        for px, py in points[:: max(1, len(points) // 10)]:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=COLORS[strategy])
        last_x, last_y = points[-1]
        draw.text((last_x - 102, last_y - 34 + label_index * 18), f"{ys[-1]:.4f}", fill=COLORS[strategy], font=font(15, True))
    legend_x = 780
    for i, strategy in enumerate(STRATEGIES):
        y = 54 + i * 30
        draw.rectangle((legend_x, y, legend_x + 22, y + 14), fill=COLORS[strategy])
        draw.text((legend_x + 32, y - 5), strategy.replace("_", " ").title(), fill=COLORS["ink"], font=font(18))
    img.save(out_path)


def draw_final_metrics(results: pd.DataFrame, out_path: Path) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    width, height = 1300, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((75, 35), "Failure-Count FedAvg Final Test Metrics", fill=COLORS["ink"], font=font(34, True))
    left, right, top, bottom = 130, 1210, 145, 640
    draw.line((left, top, left, bottom, right, bottom), fill=COLORS["muted"], width=2)
    bar_w, item_gap, group_gap = 58, 12, 76
    x = left + 35
    for _, row in results.iterrows():
        strategy = row["strategy"]
        draw.text((x - 10, bottom + 28), str(strategy).replace("_", "\n"), fill=COLORS["ink"], font=font(17))
        for j, metric_name in enumerate(metric_cols):
            value = float(row[metric_name])
            x0 = x + j * (bar_w + item_gap)
            x1 = x0 + bar_w
            y0 = bottom - value * (bottom - top - 20)
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=COLORS[strategy])
            draw.text((x0 + 4, y0 - 25), f"{value:.2f}", fill=COLORS["ink"], font=font(15))
            draw.text((x0, bottom + 108), metric_name[:4], fill=COLORS["muted"], font=font(14))
        x += len(metric_cols) * (bar_w + item_gap) + group_gap
    draw.text((32, top + 22), "Score", fill=COLORS["ink"], font=font(18))
    img.save(out_path)


def draw_loss_curve(history: pd.DataFrame, out_path: Path) -> None:
    width, height = 1250, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((75, 35), "Failure-Count FedAvg Mean Client Loss", fill=COLORS["ink"], font=font(34, True))
    left, right, top, bottom = 115, 1140, 130, 610
    draw.line((left, top, left, bottom, right, bottom), fill=COLORS["muted"], width=2)
    y_min = float(history["mean_client_loss"].min())
    y_max = float(history["mean_client_loss"].max())
    span = max(y_max - y_min, 1e-6)
    all_rounds = history["round"].to_numpy(dtype=float)
    for tick in np.linspace(y_min, y_max, 6):
        py = bottom - ((tick - y_min) / span) * (bottom - top - 25)
        draw.line((left - 5, py, right, py), fill=(225, 229, 232), width=1)
        draw.text((42, py - 9), f"{tick:.3f}", fill=COLORS["muted"], font=font(14))
    for tick in np.linspace(all_rounds.min(), all_rounds.max(), 6):
        px = left + (tick - all_rounds.min()) / max(all_rounds.max() - all_rounds.min(), 1e-6) * (right - left)
        draw.line((px, bottom, px, bottom + 5), fill=COLORS["muted"], width=1)
        draw.text((px - 15, bottom + 12), f"{int(round(tick))}", fill=COLORS["muted"], font=font(14))
    for label_index, strategy in enumerate(STRATEGIES):
        sub = history[history["strategy"] == strategy]
        xs = sub["round"].to_numpy(dtype=float)
        ys = sub["mean_client_loss"].to_numpy(dtype=float)
        points = []
        for x, y in zip(xs, ys):
            px = left + (x - xs.min()) / max(xs.max() - xs.min(), 1e-6) * (right - left)
            py = bottom - ((y - y_min) / span) * (bottom - top - 25)
            points.append((px, py))
        draw.line(points, fill=COLORS[strategy], width=4)
        last_x, last_y = points[-1]
        draw.text((last_x - 110, last_y - 32 + label_index * 18), f"{ys[-1]:.4f}", fill=COLORS[strategy], font=font(15, True))
    draw.text((28, top + 24), "Loss", fill=COLORS["ink"], font=font(18))
    draw.text((right - 70, bottom + 25), "Round", fill=COLORS["ink"], font=font(18))
    img.save(out_path)


def draw_baseline_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    width, height = 1350, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((75, 35), "Failure-Count FedAvg vs Standard FedAvg", fill=COLORS["ink"], font=font(32, True))
    left, right, top, bottom = 130, 1260, 145, 640
    draw.line((left, top, left, bottom, right, bottom), fill=COLORS["muted"], width=2)
    for tick in np.linspace(0, 1, 6):
        py = bottom - tick * (bottom - top - 20)
        draw.line((left - 5, py, right, py), fill=(225, 229, 232), width=1)
        draw.text((58, py - 9), f"{tick:.2f}", fill=COLORS["muted"], font=font(14))

    fedavg_color = (112, 128, 144)
    failure_color = (42, 157, 143)
    bar_w, pair_gap, metric_gap, group_gap = 26, 6, 24, 64
    x = left + 28
    for strategy in STRATEGIES:
        sub = comparison[comparison["strategy"] == strategy]
        if sub.empty:
            continue
        row = sub.iloc[0]
        draw.text((x - 6, bottom + 30), strategy.replace("_", "\n"), fill=COLORS["ink"], font=font(16))
        for i, metric_name in enumerate(metric_cols):
            base = float(row[f"fedavg_{metric_name}"])
            aware = float(row[f"failure_count_{metric_name}"])
            mx = x + i * (bar_w * 2 + pair_gap + metric_gap)
            for value, color, offset in [(base, fedavg_color, 0), (aware, failure_color, bar_w + pair_gap)]:
                y0 = bottom - value * (bottom - top - 20)
                draw.rounded_rectangle((mx + offset, y0, mx + offset + bar_w, bottom), radius=5, fill=color)
            draw.text((mx - 2, bottom + 112), metric_name[:4], fill=COLORS["muted"], font=font(13))
            delta = aware - base
            delta_color = failure_color if delta >= 0 else (225, 111, 86)
            draw.text((mx - 2, bottom + 132), f"{delta:+.3f}", fill=delta_color, font=font(13, True))
        x += len(metric_cols) * (bar_w * 2 + pair_gap + metric_gap) + group_gap

    legend_x = 875
    draw.rectangle((legend_x, 58, legend_x + 22, 72), fill=fedavg_color)
    draw.text((legend_x + 32, 52), "Standard FedAvg", fill=COLORS["ink"], font=font(17))
    draw.rectangle((legend_x, 88, legend_x + 22, 102), fill=failure_color)
    draw.text((legend_x + 32, 82), "Failure-Count FedAvg", fill=COLORS["ink"], font=font(17))
    draw.text((35, top + 22), "Score", fill=COLORS["ink"], font=font(18))
    draw.text((left + 10, bottom + 165), "Delta labels show Failure-Count minus Standard FedAvg.", fill=COLORS["muted"], font=font(16))
    img.save(out_path)


def compare_with_fedavg_baseline(results_df: pd.DataFrame, baseline_path: Path) -> pd.DataFrame | None:
    if not baseline_path.exists():
        return None
    baseline = pd.read_csv(baseline_path)
    rows = []
    for strategy in STRATEGIES:
        base = baseline[baseline["strategy"] == strategy]
        aware = results_df[results_df["strategy"] == strategy]
        if base.empty or aware.empty:
            continue
        base_row = base.iloc[0]
        aware_row = aware.iloc[0]
        row = {"strategy": strategy}
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            row[f"fedavg_{metric_name}"] = float(base_row[metric_name])
            row[f"failure_count_{metric_name}"] = float(aware_row[metric_name])
            row[f"delta_{metric_name}"] = float(aware_row[metric_name]) - float(base_row[metric_name])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run Variant 2 failure-count FedAvg across AI4I factory splits."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ai4i_clean_standardized.csv"
    )
    parser.add_argument(
        "--factory-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "factories"
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Strength of failure-count aggregation. 0.0 is equivalent to sample-size FedAvg.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reports = PROJECT_ROOT / "reports"
    figures = PROJECT_ROOT / "figures"
    processed = PROJECT_ROOT / "data" / "processed"

    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.data)
    missing = [c for c in FEATURES if c not in full.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    train_idx, val_idx, test_idx = stratified_split(full[TARGET].to_numpy(dtype=int), 0.6, 0.2, args.seed)
    train_ids = set(full.loc[train_idx, ID_COL].astype(int).tolist())
    x_val = full.loc[val_idx, FEATURES].to_numpy(dtype=float)
    y_val = full.loc[val_idx, TARGET].to_numpy(dtype=int)
    x_test = full.loc[test_idx, FEATURES].to_numpy(dtype=float)
    y_test = full.loc[test_idx, TARGET].to_numpy(dtype=int)

    histories = []
    thresholds = []
    finals = []
    models = {}
    for strategy in STRATEGIES:
        history, threshold_table, payload = failure_count_strategy(
            strategy=strategy,
            factory_root=args.factory_root,
            train_ids=train_ids,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            l2=args.l2,
            alpha=args.alpha,
        )
        histories.append(history)
        thresholds.append(threshold_table)
        finals.append(payload["final"])
        models[strategy] = payload["model"]

    history_df = pd.concat(histories, ignore_index=True)
    threshold_df = pd.concat(thresholds, ignore_index=True)
    results_df = pd.DataFrame(finals)
    ordered = [
        "strategy",
        "method",
        "rounds",
        "local_epochs",
        "alpha",
        "clients",
        "train_samples",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "positive_predictions",
        "communication_client_updates",
        "communication_sample_updates",
    ]
    results_df = results_df[ordered]

    history_df.to_csv(reports / "failure_count_fedavg_history.csv", index=False)
    threshold_df.to_csv(reports / "failure_count_fedavg_threshold_tuning.csv", index=False)
    results_df.to_csv(reports / "failure_count_fedavg_results.csv", index=False)
    (processed / "failure_count_fedavg_models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")

    draw_convergence(history_df, figures / "failure_count_fedavg_convergence.png")
    draw_final_metrics(results_df, figures / "failure_count_fedavg_final_metrics.png")
    draw_loss_curve(history_df, figures / "failure_count_fedavg_client_loss.png")
    comparison = compare_with_fedavg_baseline(results_df, reports / "fedavg_baseline_results.csv")
    if comparison is not None and not comparison.empty:
        comparison.to_csv(reports / "failure_count_vs_fedavg_results.csv", index=False)
        draw_baseline_comparison(comparison, figures / "failure_count_vs_fedavg_metrics.png")

    print("Failure-Count FedAvg complete.")
    print(results_df.to_string(index=False))
    if comparison is not None and not comparison.empty:
        print("\nComparison with standard FedAvg:")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
