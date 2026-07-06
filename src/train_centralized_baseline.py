from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TARGET = "Machine failure"
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
PALETTE = {
    "baseline": (130, 90, 160),
    "logistic": (42, 157, 143),
    "accent": (225, 111, 86),
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
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


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


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def weighted_log_loss(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray) -> float:
    eps = 1e-8
    loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return float(np.average(loss, weights=sample_weight))


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
    class_weight: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    xb = add_intercept(x)
    w = np.zeros(xb.shape[1], dtype=float)
    if class_weight:
        pos = max(float(y.sum()), 1.0)
        neg = max(float((y == 0).sum()), 1.0)
        weights = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    else:
        weights = np.ones(len(y), dtype=float)

    history = []
    for epoch in range(1, epochs + 1):
        p = sigmoid(xb @ w)
        grad = (xb.T @ ((p - y) * weights)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "weighted_log_loss": weighted_log_loss(y, p, weights),
                    "mean_probability": float(p.mean()),
                }
            )
    return w, pd.DataFrame(history)


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
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        row = metrics(y_true, y_prob, float(threshold))
        rows.append(row)
    table = pd.DataFrame(rows)
    best = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def draw_metrics_chart(results: pd.DataFrame, out_path: Path) -> None:
    metrics_cols = ["accuracy", "precision", "recall", "f1"]
    width, height = 1200, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((80, 35), "Centralized Baseline Metrics", fill=PALETTE["ink"], font=font(36, True))
    left, right, top, bottom = 130, 1120, 140, 610
    draw.line((left, top, left, bottom, right, bottom), fill=PALETTE["muted"], width=2)
    filtered = results[results["split"] == "test"].copy()
    bar_w = 82
    group_gap = 72
    item_gap = 16
    x = left + 40
    for _, row in filtered.iterrows():
        model = row["model"]
        color = PALETTE["logistic"] if "Logistic" in model else PALETTE["baseline"]
        draw.text((x, bottom + 28), str(model).replace(" ", "\n"), fill=PALETTE["ink"], font=font(18))
        for j, col in enumerate(metrics_cols):
            value = float(row[col])
            x0 = x + j * (bar_w + item_gap)
            x1 = x0 + bar_w
            y0 = bottom - value * (bottom - top - 20)
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=color)
            draw.text((x0 + 10, y0 - 28), f"{value:.2f}", fill=PALETTE["ink"], font=font(16))
            draw.text((x0 + 2, bottom + 95), col[:4], fill=PALETTE["muted"], font=font(15))
        x += len(metrics_cols) * (bar_w + item_gap) + group_gap
    draw.text((35, top + 20), "Score", fill=PALETTE["ink"], font=font(18))
    img.save(out_path)


def draw_confusion_matrix(row: pd.Series, out_path: Path) -> None:
    width, height = 760, 640
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 35), "Logistic Regression Test Confusion Matrix", fill=PALETTE["ink"], font=font(28, True))
    cells = [
        ("TN", int(row["tn"]), (180, 223, 219)),
        ("FP", int(row["fp"]), (246, 196, 171)),
        ("FN", int(row["fn"]), (246, 196, 171)),
        ("TP", int(row["tp"]), (180, 223, 219)),
    ]
    x0, y0, cell = 180, 150, 180
    labels = [("Pred 0", x0 + 42, 112), ("Pred 1", x0 + cell + 42, 112), ("True 0", 70, y0 + 70), ("True 1", 70, y0 + cell + 70)]
    for text, x, y in labels:
        draw.text((x, y), text, fill=PALETTE["muted"], font=font(20, True))
    for i, (name, value, color) in enumerate(cells):
        cx = x0 + (i % 2) * cell
        cy = y0 + (i // 2) * cell
        draw.rectangle((cx, cy, cx + cell - 5, cy + cell - 5), fill=color, outline=(120, 128, 132), width=2)
        draw.text((cx + 62, cy + 52), name, fill=PALETTE["ink"], font=font(24, True))
        draw.text((cx + 58, cy + 92), str(value), fill=PALETTE["ink"], font=font(26, True))
    img.save(out_path)


def draw_training_curve(history: pd.DataFrame, out_path: Path) -> None:
    width, height = 1050, 650
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 35), "Weighted Logistic Training Loss", fill=PALETTE["ink"], font=font(30, True))
    left, right, top, bottom = 110, 980, 125, 545
    draw.line((left, top, left, bottom, right, bottom), fill=PALETTE["muted"], width=2)
    xs = history["epoch"].to_numpy(dtype=float)
    ys = history["weighted_log_loss"].to_numpy(dtype=float)
    if len(xs) > 1:
        min_y, max_y = float(ys.min()), float(ys.max())
        span = max(max_y - min_y, 1e-6)
        y_ticks = np.linspace(min_y, max_y, 6)
        for tick in y_ticks:
            py = bottom - (tick - min_y) / span * (bottom - top - 20)
            draw.line((left - 5, py, right, py), fill=(225, 229, 232), width=1)
            draw.text((42, py - 9), f"{tick:.3f}", fill=PALETTE["muted"], font=font(14))
        x_ticks = np.linspace(xs.min(), xs.max(), 6)
        for tick in x_ticks:
            px = left + (tick - xs.min()) / max(xs.max() - xs.min(), 1e-6) * (right - left)
            draw.line((px, bottom, px, bottom + 5), fill=PALETTE["muted"], width=1)
            draw.text((px - 18, bottom + 12), f"{int(round(tick))}", fill=PALETTE["muted"], font=font(14))
        points = []
        for x, y in zip(xs, ys):
            px = left + (x - xs.min()) / max(xs.max() - xs.min(), 1e-6) * (right - left)
            py = bottom - (y - min_y) / span * (bottom - top - 20)
            points.append((px, py))
        draw.line(points, fill=PALETTE["accent"], width=4)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=PALETTE["accent"])
        last_x, last_y = points[-1]
        draw.text((last_x - 135, last_y - 30), f"epoch {int(xs[-1])}, loss {ys[-1]:.4f}", fill=PALETTE["ink"], font=font(15, True))
    draw.text((28, top + 25), "Loss", fill=PALETTE["ink"], font=font(18))
    draw.text((right - 60, bottom + 26), "Epoch", fill=PALETTE["ink"], font=font(18))
    img.save(out_path)


def draw_threshold_curve(threshold_table: pd.DataFrame, out_path: Path) -> None:
    width, height = 1150, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 35), "Centralized Threshold Tuning", fill=PALETTE["ink"], font=font(30, True))
    left, right, top, bottom = 115, 1040, 125, 590
    draw.line((left, top, left, bottom, right, bottom), fill=PALETTE["muted"], width=2)
    xs = threshold_table["threshold"].to_numpy(dtype=float)
    metric_specs = [
        ("precision", (41, 98, 120)),
        ("recall", (225, 111, 86)),
        ("f1", (42, 157, 143)),
    ]
    for tick in np.linspace(0, 1, 6):
        py = bottom - tick * (bottom - top - 25)
        draw.line((left - 5, py, right, py), fill=(225, 229, 232), width=1)
        draw.text((52, py - 9), f"{tick:.2f}", fill=PALETTE["muted"], font=font(14))
    for tick in np.linspace(float(xs.min()), float(xs.max()), 6):
        px = left + (tick - xs.min()) / max(xs.max() - xs.min(), 1e-6) * (right - left)
        draw.line((px, bottom, px, bottom + 5), fill=PALETTE["muted"], width=1)
        draw.text((px - 18, bottom + 12), f"{tick:.2f}", fill=PALETTE["muted"], font=font(14))
    for label_index, (metric_name, color) in enumerate(metric_specs):
        ys = threshold_table[metric_name].to_numpy(dtype=float)
        points = []
        for x, y in zip(xs, ys):
            px = left + (x - xs.min()) / max(xs.max() - xs.min(), 1e-6) * (right - left)
            py = bottom - y * (bottom - top - 25)
            points.append((px, py))
        draw.line(points, fill=color, width=4)
        best_idx = int(np.argmax(ys))
        bx, by = points[best_idx]
        draw.ellipse((bx - 5, by - 5, bx + 5, by + 5), fill=color)
        label = f"{metric_name} {ys[best_idx]:.4f} @ {xs[best_idx]:.2f}"
        label_x = bx + 8
        if label_x > right - 190:
            label_x = bx - 170
        draw.text((label_x, by - 28 + label_index * 18), label, fill=color, font=font(15, True))
    legend_x = 760
    for i, (metric_name, color) in enumerate(metric_specs):
        y = 54 + i * 28
        draw.rectangle((legend_x, y, legend_x + 22, y + 14), fill=color)
        draw.text((legend_x + 32, y - 5), metric_name.title(), fill=PALETTE["ink"], font=font(17))
    draw.text((30, top + 25), "Score", fill=PALETTE["ink"], font=font(18))
    draw.text((right - 95, bottom + 35), "Threshold", fill=PALETTE["ink"], font=font(18))
    img.save(out_path)


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Train a centralized baseline for AI4I machine failure prediction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ai4i_clean_standardized.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    args = parser.parse_args()

    reports = PROJECT_ROOT / "reports"
    figures = PROJECT_ROOT / "figures"
    processed = PROJECT_ROOT / "data" / "processed"

    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    train_idx, val_idx, test_idx = stratified_split(df[TARGET].to_numpy(dtype=int), 0.6, 0.2, args.seed)
    x = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)

    weights, history = train_logistic_regression(
        x[train_idx],
        y[train_idx],
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        class_weight=True,
    )

    val_prob = predict_proba(x[val_idx], weights)
    threshold, threshold_table = tune_threshold(y[val_idx], val_prob)
    test_prob = predict_proba(x[test_idx], weights)

    rows = []
    majority_prob = np.zeros(len(y))
    for split_name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        majority = metrics(y[idx], majority_prob[idx], 0.5)
        majority.update({"model": "Majority Class", "split": split_name})
        rows.append(majority)
        probs = predict_proba(x[idx], weights)
        logistic = metrics(y[idx], probs, threshold)
        logistic.update({"model": "Weighted Logistic Regression", "split": split_name})
        rows.append(logistic)

    results = pd.DataFrame(rows)
    ordered = ["model", "split", "threshold", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn", "positive_predictions"]
    results = results[ordered]
    results.to_csv(reports / "centralized_baseline_results.csv", index=False)
    threshold_table.to_csv(reports / "centralized_threshold_tuning.csv", index=False)
    history.to_csv(reports / "centralized_training_history.csv", index=False)

    predictions = df.loc[test_idx, ["UDI", "Product ID", "Type", TARGET, "failure_mode"]].copy()
    predictions["failure_probability"] = test_prob
    predictions["prediction"] = (test_prob >= threshold).astype(int)
    predictions.to_csv(processed / "centralized_test_predictions.csv", index=False)

    model_payload = {
        "model": "weighted_logistic_regression",
        "features": FEATURES,
        "intercept": float(weights[0]),
        "coefficients": {feature: float(value) for feature, value in zip(FEATURES, weights[1:])},
        "threshold": threshold,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "l2": args.l2,
        "note": "Failure mode columns TWF/HDF/PWF/OSF/RNF were excluded to avoid label leakage.",
    }
    (processed / "centralized_logistic_model.json").write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    draw_metrics_chart(results, figures / "centralized_baseline_metrics.png")
    logistic_test = results[(results["model"] == "Weighted Logistic Regression") & (results["split"] == "test")].iloc[0]
    draw_confusion_matrix(logistic_test, figures / "centralized_confusion_matrix.png")
    draw_training_curve(history, figures / "centralized_training_curve.png")
    draw_threshold_curve(threshold_table, figures / "centralized_threshold_curve.png")

    print("Centralized baseline complete.")
    print(results[results["split"] == "test"].to_string(index=False))
    print(f"Best validation F1 threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
