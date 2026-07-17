from __future__ import annotations

"""Standalone evaluator for predictions produced by teammates' FL algorithms.

This file does not import or modify teammates' code. Each algorithm supplies a
CSV containing validation/test probabilities. This evaluator owns ground-truth
labels, threshold selection, Non-IID client metrics, statistics and reporting.

Required CSV columns:
    UDI, seed, strategy, split, probability

Where split is "validation" or "test". Optional column "round" may contain
positive communication-round numbers for validation rows. Final rows must have
round blank/0. If the round column is absent, all rows are treated as final.

Example:
    python standalone_non_iid_evaluator.py ^
      --prediction fedavg=outputs/fedavg_predictions.csv ^
      --prediction new_method=outputs/new_method_predictions.csv ^
      --baseline fedavg
"""

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET = "Machine failure"
ID_COLUMN = "UDI"
DEFAULT_STRATEGIES = ("iid", "moderate_non_iid", "highly_non_iid")
DEFAULT_SEEDS = (42, 52, 62, 72, 82)
REQUIRED_PREDICTION_COLUMNS = {"UDI", "seed", "strategy", "split", "probability"}


def stratified_split(
    y: np.ndarray, train_ratio: float, validation_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label in np.unique(y):
        indices = np.where(y == label)[0].copy()
        rng.shuffle(indices)
        train_count = int(round(len(indices) * train_ratio))
        validation_count = int(round(len(indices) * validation_ratio))
        train.extend(indices[:train_count])
        validation.extend(indices[train_count : train_count + validation_count])
        test.extend(indices[train_count + validation_count :])
    for indices in (train, validation, test):
        rng.shuffle(indices)
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-probability, kind="mergesort")
    ranked = y_true[order]
    cumulative_true_positive = np.cumsum(ranked == 1)
    precision = cumulative_true_positive / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    prediction = probability >= threshold
    tp = int(((y_true == 1) & prediction).sum())
    tn = int(((y_true == 0) & ~prediction).sum())
    fp = int(((y_true == 0) & prediction).sum())
    fn = int(((y_true == 1) & ~prediction).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "pr_auc": average_precision(y_true, probability),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_support": int((y_true == 1).sum()),
        "negative_support": int((y_true == 0).sum()),
    }


def tune_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = [
        metrics(y_true, probability, float(threshold))
        for threshold in np.linspace(0.05, 0.95, 91)
    ]
    best = pd.DataFrame(candidates).sort_values(
        ["f1", "recall", "precision"], ascending=False
    ).iloc[0]
    return float(best["threshold"])


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=prediction_file.csv")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name or not all(character.isalnum() or character in "._-" for character in name):
        raise argparse.ArgumentTypeError("Algorithm NAME may use letters, numbers, dot, dash, underscore")
    return name, Path(path)


def validate_probability_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["UDI"] = pd.to_numeric(result["UDI"], errors="raise").astype(int)
    result["seed"] = pd.to_numeric(result["seed"], errors="raise").astype(int)
    result["strategy"] = result["strategy"].astype(str).str.lower().str.strip()
    result["split"] = result["split"].astype(str).str.lower().str.strip()
    result["probability"] = pd.to_numeric(result["probability"], errors="raise").astype(float)
    if not np.isfinite(result["probability"]).all():
        raise ValueError(f"{source} contains NaN/infinite probabilities")
    if ((result["probability"] < 0) | (result["probability"] > 1)).any():
        raise ValueError(f"{source} probabilities must be in [0, 1]")
    invalid_split = sorted(set(result["split"]) - {"validation", "test"})
    if invalid_split:
        raise ValueError(f"{source} has invalid split values: {invalid_split}")
    if "round" in result.columns:
        result["round"] = pd.to_numeric(result["round"], errors="coerce").fillna(0).astype(int)
        if (result["round"] < 0).any():
            raise ValueError(f"{source} round values cannot be negative")
    else:
        result["round"] = 0
    return result


def expected_split_tables(full_data: pd.DataFrame, seeds: tuple[int, ...]) -> dict[tuple[int, str], pd.DataFrame]:
    result: dict[tuple[int, str], pd.DataFrame] = {}
    labels = full_data[TARGET].to_numpy(dtype=int)
    for seed in seeds:
        _, validation_idx, test_idx = stratified_split(labels, 0.6, 0.2, seed)
        result[(seed, "validation")] = full_data.iloc[validation_idx][[ID_COLUMN, TARGET]].copy()
        result[(seed, "test")] = full_data.iloc[test_idx][[ID_COLUMN, TARGET]].copy()
    return result


def aligned_probabilities(
    submitted: pd.DataFrame,
    expected: pd.DataFrame,
    identity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if submitted["UDI"].duplicated().any():
        duplicates = submitted.loc[submitted["UDI"].duplicated(), "UDI"].head().tolist()
        raise ValueError(f"{identity}: duplicate UDI values, examples={duplicates}")
    expected_ids = set(expected[ID_COLUMN].astype(int))
    submitted_ids = set(submitted["UDI"].astype(int))
    missing = expected_ids - submitted_ids
    extra = submitted_ids - expected_ids
    if missing or extra:
        raise ValueError(
            f"{identity}: UDI mismatch; missing={len(missing)}, extra={len(extra)}"
        )
    merged = expected.merge(submitted[["UDI", "probability"]], on="UDI", how="left", validate="one_to_one")
    return (
        merged["UDI"].to_numpy(dtype=int),
        merged[TARGET].to_numpy(dtype=int),
        merged["probability"].to_numpy(dtype=float),
    )


def factory_membership(factory_directory: Path, ordered_ids: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    ownership = np.zeros(len(ordered_ids), dtype=int)
    for path in sorted(factory_directory.glob("factory_*.csv")):
        ids = set(pd.read_csv(path, usecols=[ID_COLUMN])[ID_COLUMN].astype(int))
        mask = np.asarray([int(identifier) in ids for identifier in ordered_ids], dtype=bool)
        if mask.any():
            masks[path.stem] = mask
            ownership += mask.astype(int)
    if not masks or not np.all(ownership == 1):
        raise ValueError(f"Factory mapping is incomplete or overlapping in {factory_directory}")
    return masks


def heterogeneity(factory_directory: Path, train_ids: set[int]) -> dict[str, float]:
    sizes = []
    rates = []
    for path in sorted(factory_directory.glob("factory_*.csv")):
        frame = pd.read_csv(path, usecols=[ID_COLUMN, TARGET])
        frame = frame[frame[ID_COLUMN].isin(train_ids)]
        if frame.empty:
            continue
        sizes.append(len(frame))
        rates.append(float(frame[TARGET].mean()))
    size_values = np.asarray(sizes, dtype=float)
    rate_values = np.asarray(rates, dtype=float)
    divergences = []
    distributions = np.column_stack([1 - rate_values, rate_values])
    for left in range(len(distributions)):
        for right in range(left + 1, len(distributions)):
            p = distributions[left]
            q = distributions[right]
            midpoint = (p + q) / 2
            kl_p = np.sum(np.where(p > 0, p * np.log(p / np.maximum(midpoint, 1e-12)), 0))
            kl_q = np.sum(np.where(q > 0, q * np.log(q / np.maximum(midpoint, 1e-12)), 0))
            divergences.append(0.5 * (kl_p + kl_q))
    return {
        "client_quantity_cv": float(size_values.std() / max(size_values.mean(), 1e-12)),
        "client_failure_rate_std": float(rate_values.std()),
        "client_failure_rate_range": float(rate_values.max() - rate_values.min()),
        "mean_pairwise_label_js_divergence": float(np.mean(divergences)) if divergences else 0.0,
    }


def per_client_metrics(
    algorithm: str,
    seed: int,
    strategy: str,
    ids: np.ndarray,
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    factory_directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    masks = factory_membership(factory_directory, ids)
    rows = []
    for client, mask in masks.items():
        rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "strategy": strategy,
                "client": client,
                "test_samples": int(mask.sum()),
                **metrics(y_true[mask], probability[mask], threshold),
            }
        )
    table = pd.DataFrame(rows)
    applicable = table[table["positive_support"] > 0]
    if applicable.empty:
        applicable = table
    aggregate: dict[str, float | int] = {
        "evaluated_clients": len(table),
        "clients_with_test_failures": len(applicable),
    }
    for metric_name in ("recall", "f1", "pr_auc", "balanced_accuracy"):
        values = applicable[metric_name].to_numpy(dtype=float)
        aggregate[f"client_macro_{metric_name}"] = float(np.nanmean(values))
        aggregate[f"client_worst_{metric_name}"] = float(np.nanmin(values))
        aggregate[f"client_{metric_name}_std"] = float(np.nanstd(values))
        aggregate[f"client_{metric_name}_gap"] = float(np.nanmax(values) - np.nanmin(values))
    return rows, aggregate


def convergence_history(
    rows: pd.DataFrame,
    expected_validation: pd.DataFrame,
    identity: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    history = []
    for round_number in sorted(value for value in rows["round"].unique() if value > 0):
        round_rows = rows[rows["round"] == round_number]
        _, labels, probabilities = aligned_probabilities(
            round_rows, expected_validation, f"{identity} round={round_number}"
        )
        round_metrics = metrics(labels, probabilities, 0.5)
        history.append(
            {
                **identity,
                "round": int(round_number),
                "val_recall_at_0_5": round_metrics["recall"],
                "val_f1_at_0_5": round_metrics["f1"],
                "val_pr_auc": round_metrics["pr_auc"],
            }
        )
    if not history:
        return history, float("nan")
    best = max(float(row["val_f1_at_0_5"]) for row in history)
    target = 0.95 * best
    convergence = next(int(row["round"]) for row in history if float(row["val_f1_at_0_5"]) >= target)
    return history, float(convergence)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "pr_auc",
        "client_macro_recall", "client_worst_recall", "client_recall_std",
        "client_macro_f1", "client_worst_f1", "client_f1_std",
        "client_macro_pr_auc", "client_worst_pr_auc", "client_pr_auc_std",
        "convergence_round",
    ]
    summary = raw.groupby(["strategy", "algorithm"], sort=False)[metric_names].agg(["mean", "std", "count"])
    summary.columns = [f"{metric_name}_{stat}" for metric_name, stat in summary.columns]
    return summary.reset_index()


def paired_bootstrap(raw: pd.DataFrame, baseline: str, samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260717)
    rows = []
    compared_metrics = [
        "recall", "f1", "f2", "pr_auc", "client_macro_recall", "client_worst_recall",
        "client_macro_f1", "client_worst_f1", "client_macro_pr_auc", "client_worst_pr_auc",
        "convergence_round",
    ]
    for strategy in raw["strategy"].unique():
        base = raw[(raw.strategy == strategy) & (raw.algorithm == baseline)].set_index("seed")
        for algorithm in raw["algorithm"].unique():
            if algorithm == baseline:
                continue
            candidate = raw[(raw.strategy == strategy) & (raw.algorithm == algorithm)].set_index("seed")
            common = sorted(set(base.index) & set(candidate.index))
            for metric_name in compared_metrics:
                if not common:
                    continue
                left = candidate.loc[common, metric_name].to_numpy(dtype=float)
                right = base.loc[common, metric_name].to_numpy(dtype=float)
                valid = np.isfinite(left) & np.isfinite(right)
                delta = left[valid] - right[valid]
                if not len(delta):
                    continue
                bootstrap = np.asarray([
                    rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(samples)
                ])
                rows.append({
                    "strategy": strategy, "algorithm": algorithm, "baseline": baseline,
                    "metric": metric_name, "paired_runs": len(delta),
                    "mean_delta": float(delta.mean()),
                    "ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap, 0.975)),
                })
    return pd.DataFrame(rows)


def non_iid_scorecard(paired: pd.DataFrame, baseline: str) -> pd.DataFrame:
    core = ("f1", "pr_auc", "client_macro_f1", "client_worst_f1")
    rows = []
    for strategy in ("moderate_non_iid", "highly_non_iid"):
        subset = paired[paired.strategy == strategy]
        for algorithm in subset["algorithm"].unique():
            table = subset[subset.algorithm == algorithm].set_index("metric")
            if not all(metric_name in table.index for metric_name in core):
                continue
            record: dict[str, Any] = {"strategy": strategy, "algorithm": algorithm, "baseline": baseline}
            means = []
            positive_ci = 0
            for metric_name in core:
                row = table.loc[metric_name]
                delta = float(row["mean_delta"])
                low = float(row["ci95_low"])
                high = float(row["ci95_high"])
                record[f"{metric_name}_delta"] = delta
                record[f"{metric_name}_ci95_low"] = low
                record[f"{metric_name}_ci95_high"] = high
                means.append(delta)
                positive_ci += int(low > 0)
            if all(value >= 0 for value in means) and positive_ci >= 2:
                verdict = "supported_improvement"
            elif all(value >= 0 for value in means):
                verdict = "promising_but_inconclusive"
            else:
                verdict = "tradeoff_or_not_supported"
            record["positive_ci_count"] = positive_ci
            record["verdict"] = verdict
            rows.append(record)
    return pd.DataFrame(rows)


def evaluate(
    data_path: Path,
    factory_root: Path,
    prediction_inputs: list[tuple[str, Path]],
    baseline: str,
    seeds: tuple[int, ...],
    strategies: tuple[str, ...],
    output_directory: Path,
    bootstrap_samples: int,
) -> dict[str, Path]:
    full_data = pd.read_csv(data_path)
    required_data = {ID_COLUMN, TARGET}
    if not required_data.issubset(full_data.columns):
        raise ValueError(f"Dataset must contain {sorted(required_data)}")
    if len(full_data[ID_COLUMN].unique()) != len(full_data):
        raise ValueError("Dataset UDI must be unique")
    named_frames: dict[str, tuple[Path, pd.DataFrame]] = {}
    for name, path in prediction_inputs:
        if name in named_frames:
            raise ValueError(f"Duplicate algorithm name: {name}")
        named_frames[name] = (path.resolve(), validate_probability_frame(pd.read_csv(path), path))
    if baseline not in named_frames:
        raise ValueError(f"Baseline '{baseline}' was not supplied")
    expected = expected_split_tables(full_data, seeds)
    labels = full_data[TARGET].to_numpy(dtype=int)
    raw_rows = []
    client_rows = []
    history_rows = []
    errors = []
    for algorithm, (source, predictions) in named_frames.items():
        for seed in seeds:
            train_idx, _, _ = stratified_split(labels, 0.6, 0.2, seed)
            train_ids = set(full_data.iloc[train_idx][ID_COLUMN].astype(int))
            for strategy in strategies:
                identity = {"algorithm": algorithm, "seed": seed, "strategy": strategy}
                try:
                    group = predictions[(predictions.seed == seed) & (predictions.strategy == strategy)]
                    final_rows = group[group["round"] == 0]
                    validation_rows = final_rows[final_rows.split == "validation"]
                    test_rows = final_rows[final_rows.split == "test"]
                    _, y_validation, validation_probability = aligned_probabilities(
                        validation_rows, expected[(seed, "validation")], f"{identity} validation"
                    )
                    test_ids, y_test, test_probability = aligned_probabilities(
                        test_rows, expected[(seed, "test")], f"{identity} test"
                    )
                    threshold = tune_threshold(y_validation, validation_probability)
                    global_metrics = metrics(y_test, test_probability, threshold)
                    per_client, aggregate = per_client_metrics(
                        algorithm, seed, strategy, test_ids, y_test, test_probability,
                        threshold, factory_root / strategy,
                    )
                    client_rows.extend(per_client)
                    history, convergence = convergence_history(
                        group[(group.split == "validation") & (group["round"] > 0)],
                        expected[(seed, "validation")], identity,
                    )
                    history_rows.extend(history)
                    raw_rows.append({
                        **identity, "source": str(source), "convergence_round": convergence,
                        **heterogeneity(factory_root / strategy, train_ids),
                        **global_metrics, **aggregate,
                    })
                    print(
                        f"OK algorithm={algorithm} seed={seed} strategy={strategy} "
                        f"F1={float(global_metrics['f1']):.4f} PR-AUC={float(global_metrics['pr_auc']):.4f}"
                    )
                except Exception as error:
                    errors.append({
                        **identity, "source": str(source), "error_type": type(error).__name__,
                        "error_message": str(error),
                    })
                    print(f"ERROR {identity}: {error}")
    if not raw_rows:
        raise RuntimeError("No complete algorithm run could be evaluated")
    output_directory.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(raw_rows)
    client = pd.DataFrame(client_rows)
    history = pd.DataFrame(history_rows)
    error_table = pd.DataFrame(
        errors,
        columns=["algorithm", "seed", "strategy", "source", "error_type", "error_message"],
    )
    summary = summarize(raw)
    paired = paired_bootstrap(raw, baseline, bootstrap_samples)
    scorecard = non_iid_scorecard(paired, baseline)
    paths = {
        "raw": output_directory / "standalone_raw.csv",
        "client_metrics": output_directory / "standalone_client_metrics.csv",
        "history": output_directory / "standalone_history.csv",
        "summary": output_directory / "standalone_summary.csv",
        "paired": output_directory / "standalone_paired_bootstrap.csv",
        "non_iid_scorecard": output_directory / "standalone_non_iid_scorecard.csv",
        "errors": output_directory / "standalone_errors.csv",
        "manifest": output_directory / "standalone_manifest.json",
    }
    raw.to_csv(paths["raw"], index=False)
    client.to_csv(paths["client_metrics"], index=False)
    history.to_csv(paths["history"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired"], index=False)
    scorecard.to_csv(paths["non_iid_scorecard"], index=False)
    error_table.to_csv(paths["errors"], index=False)
    manifest = {
        "data": str(data_path.resolve()),
        "factory_root": str(factory_root.resolve()),
        "algorithms": {name: str(path.resolve()) for name, path in prediction_inputs},
        "baseline": baseline,
        "seeds": seeds,
        "strategies": strategies,
        "successful_runs": len(raw),
        "failed_runs": len(error_table),
        "test_threshold_policy": "validation F1 maximization then locked test evaluation",
        "important_limitation": "Prediction files are trusted to come from models that did not train on test rows.",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def make_request_template(
    data_path: Path, output: Path, seeds: tuple[int, ...], strategies: tuple[str, ...]
) -> None:
    data = pd.read_csv(data_path)
    expected = expected_split_tables(data, seeds)
    rows = []
    for seed in seeds:
        for strategy in strategies:
            for split in ("validation", "test"):
                for identifier in expected[(seed, split)][ID_COLUMN]:
                    rows.append({
                        "UDI": int(identifier), "seed": seed, "strategy": strategy,
                        "split": split, "probability": "",
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Prediction request template written to {output}")


def make_demo_predictions(
    data_path: Path, output: Path, seeds: tuple[int, ...], strategies: tuple[str, ...]
) -> None:
    """Create interface-only probabilities; these are not a scientific model."""
    data = pd.read_csv(data_path)
    numeric = data.select_dtypes(include=[np.number]).drop(columns=[ID_COLUMN, TARGET], errors="ignore")
    signal = numeric.iloc[:, : min(5, numeric.shape[1])].mean(axis=1).to_numpy(dtype=float)
    signal = (signal - signal.mean()) / max(signal.std(), 1e-12)
    probability_by_id = dict(zip(data[ID_COLUMN].astype(int), 1 / (1 + np.exp(-signal))))
    expected = expected_split_tables(data, seeds)
    rows = []
    for seed in seeds:
        for strategy in strategies:
            for split in ("validation", "test"):
                for identifier in expected[(seed, split)][ID_COLUMN]:
                    rows.append({
                        "UDI": int(identifier), "seed": seed, "strategy": strategy,
                        "split": split, "probability": probability_by_id[int(identifier)],
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Demo predictions written to {output}; do not use them as a research result.")


def self_test() -> None:
    labels = np.asarray([0] * 97 + [1] * 3)
    probabilities = np.zeros(100)
    result = metrics(labels, probabilities, 0.5)
    assert math.isclose(float(result["accuracy"]), 0.97)
    assert math.isclose(float(result["balanced_accuracy"]), 0.5)
    assert math.isclose(float(result["recall"]), 0.0)
    try:
        validate_probability_frame(
            pd.DataFrame({
                "UDI": [1], "seed": [42], "strategy": ["iid"],
                "split": ["test"], "probability": [1.2],
            }),
            Path("invalid.csv"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid probability was not rejected")
    print("Self-test passed.")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Standalone Non-IID evaluator for teammates' prediction CSV files."
    )
    parser.add_argument(
        "--prediction", action="append", type=parse_named_path, default=[],
        metavar="NAME=FILE.csv", help="Repeat for every algorithm, including the baseline.",
    )
    parser.add_argument("--baseline", default="fedavg")
    parser.add_argument(
        "--data", type=Path,
        default=root / "data" / "processed" / "ai4i_clean_standardized.csv",
    )
    parser.add_argument("--factory-root", type=Path, default=root / "data" / "factories")
    parser.add_argument("--output-directory", type=Path, default=root / "reports" / "standalone_evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--make-request-template", type=Path)
    parser.add_argument("--make-demo-predictions", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    strategies = tuple(value.lower() for value in args.strategies)
    if args.self_test:
        self_test()
        return
    if args.make_request_template:
        make_request_template(args.data, args.make_request_template, seeds, strategies)
        return
    if args.make_demo_predictions:
        make_demo_predictions(args.data, args.make_demo_predictions, seeds, strategies)
        return
    if not args.prediction:
        parser.error("Provide --prediction NAME=FILE.csv (repeat for baseline and candidates)")
    paths = evaluate(
        args.data, args.factory_root, args.prediction, args.baseline, seeds, strategies,
        args.output_directory, args.bootstrap_samples,
    )
    print("\nEvaluation complete:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
