from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
PALETTE = [
    (41, 98, 120),
    (225, 111, 86),
    (42, 157, 143),
    (233, 196, 106),
    (130, 90, 160),
    (110, 138, 72),
]


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


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"UDI", "Product ID", *FEATURE_COLUMNS, TARGET, *FAILURE_MODES}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temperature_gap [K]"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["power_proxy"] = out["Rotational speed [rpm]"] * out["Torque [Nm]"]
    out["failure_mode"] = "None"
    for mode in FAILURE_MODES:
        out.loc[out[mode] == 1, "failure_mode"] = mode
    return out


def clean_and_standardize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = add_engineered_columns(df)
    numeric = NUMERIC_FEATURES + ["temperature_gap [K]", "power_proxy"]
    scaler_rows = []
    for col in numeric:
        mean = float(clean[col].mean())
        std = float(clean[col].std(ddof=0))
        scaler_rows.append({"feature": col, "mean": mean, "std": std})
        clean[f"{safe_name(col)}_z"] = (clean[col] - mean) / (std if std else 1.0)
    type_dummies = pd.get_dummies(clean["Type"], prefix="Type", dtype=int)
    clean = pd.concat([clean, type_dummies], axis=1)
    return clean, pd.DataFrame(scaler_rows)


def eda_tables(df: pd.DataFrame, clean: pd.DataFrame, out_dir: Path) -> dict:
    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_positive": int(df[TARGET].sum()),
        "target_negative": int((df[TARGET] == 0).sum()),
        "target_positive_rate": round(float(df[TARGET].mean()), 4),
    }
    pd.DataFrame({"missing_count": df.isna().sum(), "missing_rate": df.isna().mean()}).to_csv(
        out_dir / "missing_values.csv"
    )
    df.dtypes.astype(str).rename("dtype").to_csv(out_dir / "field_types.csv")
    df[NUMERIC_FEATURES].describe().T.to_csv(out_dir / "numeric_summary.csv")
    df["Type"].value_counts().rename_axis("Type").reset_index(name="count").to_csv(
        out_dir / "type_distribution.csv", index=False
    )
    df[TARGET].value_counts().sort_index().rename_axis(TARGET).reset_index(name="count").to_csv(
        out_dir / "target_distribution.csv", index=False
    )
    clean["failure_mode"].value_counts().rename_axis("failure_mode").reset_index(name="count").to_csv(
        out_dir / "failure_mode_distribution.csv", index=False
    )
    with (out_dir / "eda_overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)
    return overview


def stratified_iid_split(df: pd.DataFrame, n_clients: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    buckets = {i: [] for i in range(n_clients)}
    strata = df.groupby(["Type", TARGET], sort=False)
    for _, group in strata:
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        for client_id, split in enumerate(np.array_split(idx, n_clients)):
            buckets[client_id].extend(split.tolist())
    return {f"factory_{i + 1:02d}": df.loc[sorted(rows)].copy() for i, rows in buckets.items()}


def weighted_partition(df: pd.DataFrame, n_clients: int, weights: pd.Series, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    buckets = {i: [] for i in range(n_clients)}
    weights = weights / weights.sum()
    for _, group in df.groupby(["Type", TARGET], sort=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        counts = rng.multinomial(len(idx), weights.to_numpy())
        start = 0
        for client_id, count in enumerate(counts):
            buckets[client_id].extend(idx[start : start + count].tolist())
            start += count
    return {f"factory_{i + 1:02d}": df.loc[sorted(rows)].copy() for i, rows in buckets.items()}


def moderate_non_iid_split(df: pd.DataFrame, n_clients: int, seed: int) -> dict[str, pd.DataFrame]:
    quality_score = df["Type"].map({"L": 1.8, "M": 1.0, "H": 0.65}).fillna(1.0)
    wear_score = 1 + (df["Tool wear [min]"] / max(float(df["Tool wear [min]"].max()), 1.0))
    weights = pd.Series(np.linspace(0.75, 1.25, n_clients), index=range(n_clients))
    weighted = {}
    rng = np.random.default_rng(seed)
    buckets = {i: [] for i in range(n_clients)}
    for _, group in df.groupby([TARGET], sort=False):
        idx = group.index.to_numpy().copy()
        probs = []
        for client_id in range(n_clients):
            client_bias = (quality_score.loc[idx] * wear_score.loc[idx] * weights.loc[client_id]).to_numpy()
            probs.append(client_bias / client_bias.sum())
        scores = np.vstack(probs).T
        assignment = [rng.choice(n_clients, p=row / row.sum()) for row in scores]
        for row_idx, client_id in zip(idx, assignment):
            buckets[int(client_id)].append(int(row_idx))
    for client_id, rows in buckets.items():
        weighted[f"factory_{client_id + 1:02d}"] = df.loc[sorted(rows)].copy()
    return weighted


def highly_non_iid_split(df: pd.DataFrame, n_clients: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    buckets = {i: [] for i in range(n_clients)}
    dominant_modes = ["HDF", "PWF", "OSF", "TWF", "RNF"]
    mode_to_client = {mode: i % n_clients for i, mode in enumerate(dominant_modes)}
    for mode, group in df.groupby("failure_mode", sort=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        if mode == "None":
            for client_id, split in enumerate(np.array_split(idx, n_clients)):
                buckets[client_id].extend(split.tolist())
            continue
        dominant = mode_to_client.get(mode, rng.integers(0, n_clients))
        main_count = int(round(len(idx) * 0.8))
        buckets[dominant].extend(idx[:main_count].tolist())
        remaining = idx[main_count:]
        for client_id, split in enumerate(np.array_split(remaining, n_clients)):
            buckets[client_id].extend(split.tolist())
    return {f"factory_{i + 1:02d}": df.loc[sorted(rows)].copy() for i, rows in buckets.items()}


def export_partitions(partitions: dict[str, dict[str, pd.DataFrame]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for strategy, clients in partitions.items():
        strategy_dir = out_dir / strategy
        ensure_dirs(strategy_dir)
        for client, data in clients.items():
            data.to_csv(strategy_dir / f"{client}.csv", index=False)
            row = {
                "strategy": strategy,
                "factory": client,
                "rows": len(data),
                "failure_rate": round(float(data[TARGET].mean()), 4) if len(data) else 0.0,
            }
            for t, count in data["Type"].value_counts().to_dict().items():
                row[f"type_{t}"] = int(count)
            for mode, count in data["failure_mode"].value_counts().to_dict().items():
                row[f"mode_{mode}"] = int(count)
            rows.append(row)
    summary = pd.DataFrame(rows).fillna(0)
    summary.to_csv(out_dir / "factory_partition_summary.csv", index=False)
    return summary


def draw_bar_chart(
    data: dict[str, int | float],
    title: str,
    out_path: Path,
    width: int = 1200,
    height: int = 760,
    y_label: str = "Count",
) -> None:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, small_font = font(36, True), font(22), font(18)
    margin = 105
    top = 100
    bottom = height - 125
    left = 130
    right = width - 70
    draw.text((left, 35), title, fill=(28, 35, 39), font=title_font)
    draw.line((left, top, left, bottom, right, bottom), fill=(80, 88, 92), width=2)
    max_val = max(float(v) for v in data.values()) if data else 1
    bar_area = right - left
    bar_w = max(24, int(bar_area / max(len(data), 1) * 0.58))
    gap = (bar_area - bar_w * len(data)) / max(len(data), 1)
    for i, (label, value) in enumerate(data.items()):
        x0 = left + gap / 2 + i * (bar_w + gap)
        x1 = x0 + bar_w
        h = (float(value) / max_val) * (bottom - top - 20)
        y0 = bottom - h
        color = PALETTE[i % len(PALETTE)]
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=color)
        draw.text((x0, y0 - 28), f"{int(value)}", fill=(28, 35, 39), font=small_font)
        draw.text((x0, bottom + 18), str(label), fill=(28, 35, 39), font=label_font)
    draw.text((28, top + 20), y_label, fill=(28, 35, 39), font=small_font)
    img.save(out_path)


def draw_heatmap(corr: pd.DataFrame, title: str, out_path: Path) -> None:
    labels = [safe_name(c).replace("_", "\n") for c in corr.columns]
    n = len(labels)
    cell = 88
    left = 250
    top = 150
    width = left + n * cell + 100
    height = top + n * cell + 150
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 40), title, fill=(28, 35, 39), font=font(34, True))
    for i, row_label in enumerate(labels):
        draw.text((35, top + i * cell + 28), row_label, fill=(28, 35, 39), font=font(15))
    for j, col_label in enumerate(labels):
        draw.text((left + j * cell + 8, 95), col_label, fill=(28, 35, 39), font=font(15))
    for i in range(n):
        for j in range(n):
            val = float(corr.iloc[i, j])
            intensity = int(245 - abs(val) * 150)
            color = (intensity, 80 + int(110 * (val > 0)), 80 + int(130 * (val < 0)))
            x0, y0 = left + j * cell, top + i * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color)
            draw.text((x0 + 20, y0 + 32), f"{val:.2f}", fill=(20, 25, 28), font=font(15, True))
    img.save(out_path)


def draw_client_distribution(summary: pd.DataFrame, strategy: str, out_path: Path) -> None:
    subset = summary[summary["strategy"] == strategy].copy()
    factories = subset["factory"].tolist()
    failure = subset["failure_rate"].astype(float).tolist()
    rows = subset["rows"].astype(int).tolist()
    img = Image.new("RGB", (1250, 780), "white")
    draw = ImageDraw.Draw(img)
    draw.text((80, 38), f"{strategy.replace('_', ' ').title()} Factory Distribution", fill=(28, 35, 39), font=font(34, True))
    left, right, top, bottom = 130, 1160, 135, 610
    draw.line((left, top, left, bottom, right, bottom), fill=(80, 88, 92), width=2)
    max_rows = max(rows) if rows else 1
    bar_w = 96
    gap = (right - left - bar_w * len(factories)) / max(len(factories), 1)
    for i, factory in enumerate(factories):
        x0 = left + gap / 2 + i * (bar_w + gap)
        x1 = x0 + bar_w
        h = rows[i] / max_rows * (bottom - top - 35)
        y0 = bottom - h
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=PALETTE[i % len(PALETTE)])
        draw.text((x0 - 2, y0 - 30), str(rows[i]), fill=(28, 35, 39), font=font(17))
        draw.text((x0 - 18, bottom + 18), factory.replace("factory_", "F"), fill=(28, 35, 39), font=font(19))
        draw.text((x0 - 8, bottom + 48), f"fail {failure[i]:.1%}", fill=(75, 82, 87), font=font(16))
    img.save(out_path)


def create_figures(df: pd.DataFrame, clean: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    draw_bar_chart(df["Type"].value_counts().sort_index().to_dict(), "Product Type Distribution", out_dir / "type_distribution.png")
    draw_bar_chart(
        df[TARGET].value_counts().sort_index().rename(index={0: "Normal", 1: "Failure"}).to_dict(),
        "Machine Failure Class Distribution",
        out_dir / "class_distribution.png",
    )
    draw_bar_chart(
        clean["failure_mode"].value_counts().to_dict(),
        "Failure Mode Distribution",
        out_dir / "failure_mode_distribution.png",
    )
    corr_cols = NUMERIC_FEATURES + ["temperature_gap [K]", "power_proxy", TARGET]
    draw_heatmap(clean[corr_cols].corr(numeric_only=True), "Numeric Feature Correlation", out_dir / "correlation_heatmap.png")
    for strategy in summary["strategy"].unique():
        draw_client_distribution(summary, strategy, out_dir / f"client_distribution_{strategy}.png")


def write_summary(overview: dict, summary: pd.DataFrame, report_dir: Path) -> None:
    iid = summary[summary["strategy"] == "iid"]
    moderate = summary[summary["strategy"] == "moderate_non_iid"]
    high = summary[summary["strategy"] == "highly_non_iid"]
    text = f"""# AI4I 2020 Dataset Summary

## Dataset Profile
- Source file: `data/raw/ai4i2020.csv`
- Scale: {overview['rows']:,} rows x {overview['columns']} original columns.
- Target: `Machine failure`, with {overview['target_positive']:,} positive samples and {overview['target_negative']:,} normal samples.
- Class imbalance: failure rate is {overview['target_positive_rate']:.2%}.
- Quality categories: `L`, `M`, `H`; physical sensor fields include temperature, rotational speed, torque, and tool wear.
- Missing values: see `reports/missing_values.csv`; this dataset has no missing values in the original AI4I file if all counts are zero.

## Cleaning And Standardization
- Preserved identifiers (`UDI`, `Product ID`) for traceability.
- Added `failure_mode`, `temperature_gap [K]`, and `power_proxy`.
- Standardized all numeric sensor features into z-score columns.
- Added one-hot columns for product type.

## Federated Factory Splits
- IID: stratified by product type and machine failure label. Factory row range: {int(iid['rows'].min())}-{int(iid['rows'].max())}.
- Moderate Non-IID: biased by product type and tool wear while preserving both target classes across factories. Factory row range: {int(moderate['rows'].min())}-{int(moderate['rows'].max())}.
- Highly Non-IID: concentrates failure modes and normal samples into dominant factories to create strong client heterogeneity. Factory row range: {int(high['rows'].min())}-{int(high['rows'].max())}.

## Presentation Figures
- `figures/class_distribution.png`
- `figures/type_distribution.png`
- `figures/failure_mode_distribution.png`
- `figures/correlation_heatmap.png`
- `figures/client_distribution_iid.png`
- `figures/client_distribution_moderate_non_iid.png`
- `figures/client_distribution_highly_non_iid.png`
"""
    (report_dir / "dataset_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Prepare AI4I 2020 EDA, cleaning, plots, and factory splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "raw" / "ai4i2020.csv"
    )
    parser.add_argument("--clients", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    processed_dir = project_root / "data" / "processed"
    factory_dir = project_root / "data" / "factories"
    figure_dir = project_root / "figures"
    report_dir = project_root / "reports"

    ensure_dirs(processed_dir, factory_dir, figure_dir, report_dir)

    df = load_data(args.input)
    clean, scaler = clean_and_standardize(df)
    clean.to_csv(processed_dir / "ai4i_clean_standardized.csv", index=False)
    scaler.to_csv(processed_dir / "standardization_parameters.csv", index=False)
    overview = eda_tables(df, clean, report_dir)

    partitions = {
        "iid": stratified_iid_split(clean, args.clients, args.seed),
        "moderate_non_iid": moderate_non_iid_split(clean, args.clients, args.seed + 1),
        "highly_non_iid": highly_non_iid_split(clean, args.clients, args.seed + 2),
    }
    partition_summary = export_partitions(partitions, factory_dir)
    create_figures(df, clean, partition_summary, figure_dir)
    write_summary(overview, partition_summary, report_dir)
    print(f"Prepared AI4I dataset with {args.clients} factories per strategy.")
    print(f"Clean data: {processed_dir / 'ai4i_clean_standardized.csv'}")
    print(f"Factory splits: {factory_dir}")
    print(f"Figures: {figure_dir}")
    print(f"Summary: {report_dir / 'dataset_summary.md'}")


if __name__ == "__main__":
    main()
