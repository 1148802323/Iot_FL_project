from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from flwr.server.strategy.aggregate import aggregate

from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from iot_fl.flower_framework.strategies.failure_aware_v1 import (
    FailureAwareV1Strategy,
)

from iot_fl.flower_framework.strategies.failure_aware_v2 import (
    FailureAwareV2Strategy,
)

from iot_fl.flower_framework.strategies.failure_aware_v4 import (
    FailureAwareV4Strategy,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iot_fl.config import FEATURES, ID_COL, STRATEGIES, TARGET
from iot_fl.data_utils import load_clients, stratified_split
from iot_fl.flower_framework.flower_client import AI4IFlowerClient
from iot_fl.metrics import classification_metrics
from iot_fl.model import (
    local_train,
    predict_proba,
    sample_weights,
    weighted_log_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate numerical consistency between the original FedAvg "
            "baseline logic and the Flower framework layer."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ai4i_clean_standardized.csv"
        ),
    )
    parser.add_argument(
        "--factory-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "factories",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=STRATEGIES,
        default="iid",
    )
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parameter-tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-12,
    )
    return parser.parse_args()


def max_abs_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(first, dtype=float)
                - np.asarray(second, dtype=float)
            )
        )
    )


def evaluate_parameters(
    x_val: np.ndarray,
    y_val: np.ndarray,
    parameters: np.ndarray,
) -> dict[str, float]:
    probabilities = predict_proba(x_val, parameters)
    loss = weighted_log_loss(
        y_val,
        probabilities,
        sample_weights(y_val),
    )
    metric_values = classification_metrics(
        y_val,
        probabilities,
        threshold=0.5,
    )
    return {
        "loss": float(loss),
        "accuracy": float(metric_values["accuracy"]),
        "precision": float(metric_values["precision"]),
        "recall": float(metric_values["recall"]),
        "f1": float(metric_values["f1"]),
    }


def main() -> None:
    args = parse_args()

    if args.local_epochs <= 0:
        raise ValueError("local-epochs must be greater than zero")
    if args.lr <= 0:
        raise ValueError("lr must be greater than zero")
    if args.l2 < 0:
        raise ValueError("l2 must be non-negative")
    if args.parameter_tolerance < 0:
        raise ValueError("parameter-tolerance must be non-negative")
    if args.metric_tolerance < 0:
        raise ValueError("metric-tolerance must be non-negative")

    full = pd.read_csv(args.data)

    required_columns = [ID_COL, TARGET, *FEATURES]
    missing_columns = [
        column
        for column in required_columns
        if column not in full.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    train_idx, val_idx, _ = stratified_split(
        full[TARGET].to_numpy(dtype=int),
        train_ratio=0.6,
        val_ratio=0.2,
        seed=args.seed,
    )

    train_ids = set(
        full.iloc[train_idx][ID_COL]
        .astype(int)
        .tolist()
    )

    x_val = full.iloc[val_idx][FEATURES].to_numpy(
        dtype=float
    )
    y_val = full.iloc[val_idx][TARGET].to_numpy(
        dtype=int
    )

    clients = load_clients(
        strategy_directory=(
            args.factory_root / args.strategy
        ),
        train_ids=train_ids,
    )

    initial_parameters = np.zeros(
        len(FEATURES) + 1,
        dtype=float,
    )

    baseline_updates: list[tuple[np.ndarray, int]] = []
    flower_updates: list[tuple[list[np.ndarray], int]] = []
    failure_aware_results = []
    client_results: list[dict[str, Any]] = []

    fit_config = {
        "learning_rate": float(args.lr),
        "local_epochs": int(args.local_epochs),
        "l2": float(args.l2),
    }

    for client_data in clients:
        client_name = str(client_data["name"])
        x_train = np.asarray(client_data["x"], dtype=float)
        y_train = np.asarray(client_data["y"], dtype=int)
        num_examples = len(y_train)

        baseline_parameters, baseline_loss = local_train(
            initial_parameters=initial_parameters,
            x=x_train,
            y=y_train,
            learning_rate=args.lr,
            epochs=args.local_epochs,
            l2=args.l2,
        )

        flower_client = AI4IFlowerClient(
            client_name=client_name,
            x_train=x_train,
            y_train=y_train,
            initial_parameters=initial_parameters,
        )

        flower_parameter_list, flower_examples, flower_metrics = (
            flower_client.fit(
                parameters=[initial_parameters.copy()],
                config=fit_config,
            )
        )

        fit_res = FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=ndarrays_to_parameters(
                flower_parameter_list
            ),
            num_examples=flower_examples,
            metrics={
                "failure_rate": float(
                    flower_metrics["failure_rate"]
                ),
            },
        )

        failure_aware_results.append(
            (None, fit_res)
        )

        flower_parameters = np.asarray(
            flower_parameter_list[0],
            dtype=float,
        )
        flower_loss = float(flower_metrics["local_loss"])

        parameter_difference = max_abs_difference(
            baseline_parameters,
            flower_parameters,
        )
        loss_difference = abs(
            float(baseline_loss) - flower_loss
        )

        baseline_updates.append(
            (baseline_parameters, num_examples)
        )
        flower_updates.append(
            (flower_parameter_list, flower_examples)
        )

        client_results.append(
            {
                "client_name": client_name,
                "num_examples": num_examples,
                "baseline_loss": float(baseline_loss),
                "flower_loss": flower_loss,
                "loss_abs_difference": loss_difference,
                "parameter_max_abs_difference": (
                    parameter_difference
                ),
                "parameters_match": (
                    parameter_difference
                    <= args.parameter_tolerance
                ),
                "loss_matches": (
                    loss_difference
                    <= args.metric_tolerance
                ),
            }
        )

    total_examples = sum(
        num_examples
        for _, num_examples in baseline_updates
    )

    baseline_global_parameters = sum(
        parameters * (num_examples / total_examples)
        for parameters, num_examples in baseline_updates
    )

    flower_aggregated = aggregate(flower_updates)
    if len(flower_aggregated) != 1:
        raise ValueError(
            "Expected one aggregated Flower parameter array"
        )

    flower_global_parameters = np.asarray(
        flower_aggregated[0],
        dtype=float,
    )

    global_parameter_difference = max_abs_difference(
        baseline_global_parameters,
        flower_global_parameters,
    )

    failure_aware_strategy = FailureAwareV1Strategy(
        alpha=0.0,
    )

    failure_aware_parameters, _ = (
        failure_aware_strategy.aggregate_fit(
            server_round=1,
            results=failure_aware_results,
            failures=[],
        )
    )

    if failure_aware_parameters is None:
        raise ValueError(
            "FailureAwareV1Strategy returned no parameters"
        )

    failure_aware_arrays = parameters_to_ndarrays(
        failure_aware_parameters
    )

    if len(failure_aware_arrays) != 1:
        raise ValueError(
            "Expected one Failure-Aware parameter array"
        )

    failure_aware_global_parameters = np.asarray(
        failure_aware_arrays[0],
        dtype=float,
    )

    failure_aware_difference = max_abs_difference(
        flower_global_parameters,
        failure_aware_global_parameters,
    )

    failure_aware_consistency_pass = (
        failure_aware_difference
        <= args.parameter_tolerance
    )

    failure_aware_v2_strategy = FailureAwareV2Strategy(
        alpha=0.0,
    )

    failure_aware_v2_parameters, _ = (
        failure_aware_v2_strategy.aggregate_fit(
            server_round=1,
            results=failure_aware_results,
            failures=[],
        )
    )

    if failure_aware_v2_parameters is None:
        raise ValueError(
            "FailureAwareV2Strategy returned no parameters"
        )

    failure_aware_v2_arrays = parameters_to_ndarrays(
        failure_aware_v2_parameters
    )

    if len(failure_aware_v2_arrays) != 1:
        raise ValueError(
            "Expected one Failure-Aware V2 parameter array"
        )

    failure_aware_v2_global_parameters = np.asarray(
        failure_aware_v2_arrays[0],
        dtype=float,
    )

    failure_aware_v2_difference = max_abs_difference(
        flower_global_parameters,
        failure_aware_v2_global_parameters,
    )

    failure_aware_v2_consistency_pass = (
        failure_aware_v2_difference
        <= args.parameter_tolerance
    )

    failure_aware_v4_strategy = FailureAwareV4Strategy(
        schedule="fixed",
        lambda_max=0.0,
        total_rounds=1,
    )

    failure_aware_v4_parameters, _ = (
        failure_aware_v4_strategy.aggregate_fit(
            server_round=1,
            results=failure_aware_results,
            failures=[],
        )
    )

    if failure_aware_v4_parameters is None:
        raise ValueError(
            "FailureAwareV4Strategy returned no parameters"
        )

    failure_aware_v4_arrays = parameters_to_ndarrays(
        failure_aware_v4_parameters
    )

    if len(failure_aware_v4_arrays) != 1:
        raise ValueError(
            "Expected one Failure-Aware V4 parameter array"
        )

    failure_aware_v4_global_parameters = np.asarray(
        failure_aware_v4_arrays[0],
        dtype=float,
    )

    failure_aware_v4_difference = max_abs_difference(
        flower_global_parameters,
        failure_aware_v4_global_parameters,
    )

    failure_aware_v4_consistency_pass = (
        failure_aware_v4_difference
        <= args.parameter_tolerance
    )

    baseline_validation = evaluate_parameters(
        x_val,
        y_val,
        baseline_global_parameters,
    )
    flower_validation = evaluate_parameters(
        x_val,
        y_val,
        flower_global_parameters,
    )

    metric_differences = {
        metric_name: abs(
            baseline_validation[metric_name]
            - flower_validation[metric_name]
        )
        for metric_name in baseline_validation
    }

    clients_pass = all(
        result["parameters_match"]
        and result["loss_matches"]
        for result in client_results
    )
    aggregation_pass = (
        global_parameter_difference
        <= args.parameter_tolerance
    )
    metrics_pass = all(
        difference <= args.metric_tolerance
        for difference in metric_differences.values()
    )
    overall_pass = (
            clients_pass
            and aggregation_pass
            and metrics_pass
            and failure_aware_consistency_pass
            and failure_aware_v2_consistency_pass
            and failure_aware_v4_consistency_pass
    )

    report = {
        "validation": (
            "Baseline Consistency Validation"
        ),
        "failure_aware_v2_alpha_zero_max_abs_difference": (
            failure_aware_v2_difference
        ),
        "failure_aware_v4_lambda_zero_max_abs_difference": (
            failure_aware_v4_difference
        ),
        "strategy": args.strategy,
        "seed": args.seed,
        "learning_rate": args.lr,
        "local_epochs": args.local_epochs,
        "l2": args.l2,
        "num_clients": len(clients),
        "train_samples": total_examples,
        "validation_samples": len(y_val),
        "parameter_tolerance": (
            args.parameter_tolerance
        ),
        "metric_tolerance": args.metric_tolerance,
        "checks": {
            "single_client_training": clients_pass,
            "fedavg_aggregation": aggregation_pass,
            "validation_metrics": metrics_pass,
            "failure_aware_v1_alpha_zero": (
                failure_aware_consistency_pass
            ),
            "failure_aware_v2_alpha_zero": (
                failure_aware_v2_consistency_pass
            ),
            "failure_aware_v4_lambda_zero": (
                failure_aware_v4_consistency_pass
            ),
            "overall": overall_pass,
        },
        "global_parameter_max_abs_difference": (
            global_parameter_difference
        ),
        "failure_aware_v1_alpha_zero_max_abs_difference": (
            failure_aware_difference
        ),
        "baseline_validation": baseline_validation,
        "flower_validation": flower_validation,
        "validation_metric_abs_differences": (
            metric_differences
        ),
        "clients": client_results,
    }

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "baseline_consistency"
        / args.strategy
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "round1_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    np.savez(
        output_dir / "round1_parameters.npz",
        initial_parameters=initial_parameters,
        baseline_global_parameters=(
            baseline_global_parameters
        ),
        flower_global_parameters=(
            flower_global_parameters
        ),
    )

    client_table = pd.DataFrame(client_results)
    client_table.to_csv(
        output_dir / "round1_client_comparison.csv",
        index=False,
    )

    print("=" * 72)
    print("STEP 5.2 - BASELINE CONSISTENCY VALIDATION")
    print("=" * 72)
    print(f"Strategy: {args.strategy}")
    print(f"Clients: {len(clients)}")
    print(f"Training samples: {total_examples}")
    print(
        "Maximum global parameter difference: "
        f"{global_parameter_difference:.16e}"
    )
    print("Validation metric differences:")
    for metric_name, difference in metric_differences.items():
        print(
            f"  {metric_name}: "
            f"{difference:.16e}"
        )
    print("-" * 72)
    print(
        "Single-client training: "
        f"{'PASS' if clients_pass else 'FAIL'}"
    )
    print(
        "FedAvg aggregation: "
        f"{'PASS' if aggregation_pass else 'FAIL'}"
    )
    print(
        "Validation metrics: "
        f"{'PASS' if metrics_pass else 'FAIL'}"
    )
    print(
        "FailureAware V1 alpha=0 consistency: "
        f"{'PASS' if failure_aware_consistency_pass else 'FAIL'}"
    )

    print(
        "FedAvg vs V1(alpha=0) maximum parameter difference: "
        f"{failure_aware_difference:.16e}"
    )

    print(
        "FailureAware V2 alpha=0 consistency: "
        f"{'PASS' if failure_aware_v2_consistency_pass else 'FAIL'}"
    )

    print(
        "FedAvg vs V2(alpha=0) maximum parameter difference: "
        f"{failure_aware_v2_difference:.16e}"
    )

    print(
        "FailureAware V4 lambda=0 consistency: "
        f"{'PASS' if failure_aware_v4_consistency_pass else 'FAIL'}"
    )

    print(
        "FedAvg vs V4(lambda=0) maximum parameter difference: "
        f"{failure_aware_v4_difference:.16e}"
    )
    print(
        "OVERALL RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )
    print("-" * 72)
    print(f"JSON report: {report_path}")
    print(f"Parameter archive: {output_dir / 'round1_parameters.npz'}")
    print(
        "Client comparison: "
        f"{output_dir / 'round1_client_comparison.csv'}"
    )

    if not overall_pass:
        raise SystemExit(
            "Consistency validation failed. "
            "Inspect the generated report before continuing."
        )


if __name__ == "__main__":
    main()
