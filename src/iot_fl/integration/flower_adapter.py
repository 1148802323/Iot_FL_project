from __future__ import annotations

from pathlib import Path

import pandas as pd
from flwr.simulation import run_simulation

from iot_fl.config import FEATURES, ID_COL, TARGET
from iot_fl.data_utils import load_clients, stratified_split
from iot_fl.flower_framework.client_app import create_client_app
from iot_fl.flower_framework.config import FrameworkConfig
from iot_fl.flower_framework.server_app import create_server_app


SUPPORTED_AGGREGATIONS = {
    "fedavg",
    "failure_aware_v1",
    "failure_aware_v2",
    "failure_aware_v3",
    "failure_aware_v4",
}


def run_flower_experiment(
    *,
    aggregation: str,
    distribution: str,
    rounds: int,
    local_epochs: int,
    learning_rate: float,
    l2: float,
    data_path: Path,
    factory_root: Path,
    seed: int = 42,
    alpha: float | None = None,
    v2_alpha: float | None = None,
    v3_lambda: float = 1.0,
    v3_beta: float = 1.0,
    v4_schedule: str | None = None,
    v4_lambda_max: float | None = None,
    v4_target_recall: float | None = None,
    v4_eta: float | None = None,
) -> dict:
    """
    Web-to-Flower integration entry point.

    WEB-Step 4 currently enables FedAvg only.
    The frozen Flower Framework is reused without modification.
    """

    normalized_aggregation = aggregation.strip().lower()

    if normalized_aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Aggregation '{aggregation}' is not enabled in the current "
            "Web integration stage."
        )

    data_path = Path(data_path)
    factory_root = Path(factory_root)

    full = pd.read_csv(data_path)

    required_columns = [
        ID_COL,
        TARGET,
        *FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in full.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    train_idx, val_idx, _test_idx = stratified_split(
        full[TARGET].to_numpy(dtype=int),
        train_ratio=0.6,
        val_ratio=0.2,
        seed=seed,
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

    strategy_directory = (
        factory_root / distribution
    )

    clients = load_clients(
        strategy_directory=strategy_directory,
        train_ids=train_ids,
    )

    framework_config_kwargs = {
        "aggregation": normalized_aggregation,
        "num_clients": len(clients),
        "num_rounds": rounds,
    }

    if normalized_aggregation == "failure_aware_v1":
        if alpha is None:
            raise ValueError(
                "failure_aware_v1 requires parameter 'alpha'."
            )

        framework_config_kwargs["failure_aware_alpha"] = alpha

    if normalized_aggregation == "failure_aware_v2":
        if v2_alpha is None:
            raise ValueError(
                "failure_aware_v2 requires parameter 'v2_alpha'."
            )

        framework_config_kwargs["failure_aware_v2_alpha"] = v2_alpha

    if normalized_aggregation == "failure_aware_v3":
        framework_config_kwargs[
            "failure_aware_v3_lambda"
        ] = float(v3_lambda)

        framework_config_kwargs[
            "failure_aware_v3_beta"
        ] = float(v3_beta)

    if normalized_aggregation == "failure_aware_v4":
        missing_parameters = [
            name
            for name, value in {
                "v4_schedule": v4_schedule,
                "v4_lambda_max": v4_lambda_max,
                "v4_target_recall": v4_target_recall,
                "v4_eta": v4_eta,
            }.items()
            if value is None
        ]

        if missing_parameters:
            raise ValueError(
                "failure_aware_v4 requires parameters: "
                + ", ".join(missing_parameters)
            )

        framework_config_kwargs[
            "failure_aware_v4_schedule"
        ] = str(v4_schedule)

        framework_config_kwargs[
            "failure_aware_v4_lambda_max"
        ] = float(v4_lambda_max)

        framework_config_kwargs[
            "failure_aware_v4_target_recall"
        ] = float(v4_target_recall)

        framework_config_kwargs[
            "failure_aware_v4_eta"
        ] = float(v4_eta)

    framework_config = FrameworkConfig(
        **framework_config_kwargs
    )

    client_app = create_client_app(
        clients=clients,
    )

    convergence_history: list[dict] = []

    def collect_result(
            server_round: int,
            loss: float,
            metrics: dict,
    ) -> None:
        convergence_history.append(
            {
                "round": int(server_round),
                "loss": float(loss),
                **metrics,
            }
        )

    server_app = create_server_app(
        num_features=len(FEATURES),
        x_val=x_val,
        y_val=y_val,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        l2=l2,
        config=framework_config,
        result_callback=collect_result,
    )

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(clients),
    )

    if not convergence_history:
        raise RuntimeError(
            "Flower simulation completed without evaluation results."
        )

    final = convergence_history[-1]

    return {
        "algorithm": normalized_aggregation,
        "distribution": distribution,
        "rounds": rounds,
        "accuracy": float(final["accuracy"]),
        "precision": float(final["precision"]),
        "recall": float(final["recall"]),
        "f1": float(final["f1"]),
        "threshold": float(final["threshold"]),
        "loss": float(final["loss"]),
        "convergence_history": convergence_history,
        "communication_cost": len(clients) * rounds,
    }