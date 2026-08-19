from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
from pathlib import Path

import pandas as pd

from flwr.simulation import run_simulation

from iot_fl.config import (
    FEATURES,
    ID_COL,
    STRATEGIES,
    TARGET,
)

from iot_fl.data_utils import (
    load_clients,
    stratified_split,
)

from iot_fl.flower_framework.client_app import (
    create_client_app,
)

from iot_fl.flower_framework.server_app import (
    create_server_app,
)

from iot_fl.flower_framework.config import FrameworkConfig

PROJECT_ROOT = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AI4I federated learning experiment with Flower."
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

    parser.add_argument(
        "--aggregation",
        type=str,
        default="fedavg",

    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Failure-aware strength for failure_aware_v1.",
    )

    parser.add_argument(
        "--v2-alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--v3-lambda",
        type=float,
        default=1.0,
        help="Failure-rate weight for failure_aware_v3.",
    )

    parser.add_argument(
        "--v3-beta",
        type=float,
        default=1.0,
        help="Local-recall weight for failure_aware_v3.",
    )

    parser.add_argument(
        "--v4-schedule",
        type=str,
        choices=("fixed", "linear","recall_adaptive"),
        default="linear",
    )

    parser.add_argument(
        "--v4-lambda-max",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--v4-target-recall",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--v4-eta",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--l2",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    full = pd.read_csv(args.data)

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

    train_idx, val_idx, test_idx = stratified_split(
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

    x_test = full.iloc[test_idx][FEATURES].to_numpy(
        dtype=float
    )
    y_test = full.iloc[test_idx][TARGET].to_numpy(
        dtype=int
    )

    strategy_directory = (
        args.factory_root / args.strategy
    )

    clients = load_clients(
        strategy_directory=strategy_directory,
        train_ids=train_ids,
    )

    config = FrameworkConfig(
        aggregation=args.aggregation,
        num_clients=len(clients),
        num_rounds=args.rounds,
        failure_aware_alpha=args.alpha,
        failure_aware_v2_alpha=args.v2_alpha,
        failure_aware_v3_lambda=args.v3_lambda,
        failure_aware_v3_beta=args.v3_beta,
        failure_aware_v4_schedule=args.v4_schedule,
        failure_aware_v4_lambda_max=args.v4_lambda_max,
        failure_aware_v4_target_recall=args.v4_target_recall,
        failure_aware_v4_eta=args.v4_eta,
    )

    print(f"[Config] {config}")

    client_app = create_client_app(
        clients=clients,
    )

    server_app = create_server_app(
        num_features=len(FEATURES),
        x_val=x_val,
        y_val=y_val,
        config=config,
        learning_rate=args.lr,
        local_epochs=args.local_epochs,
        l2=args.l2,


    )


    print("Flower data preparation complete.")
    print(f"Strategy: {args.strategy}")
    print(f"Training clients: {len(clients)}")
    print(f"Training IDs: {len(train_ids)}")
    print(f"Validation rows: {len(y_val)}")
    print(f"Test rows: {len(y_test)}")

    print("Starting Flower simulation...")

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(clients),
    )

    print("Flower simulation complete.")

if __name__ == "__main__":
    main()