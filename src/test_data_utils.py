from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from iot_fl.config import ID_COL, TARGET
from iot_fl.data_utils import load_clients, stratified_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai4i_clean_standardized.csv"
)

FACTORY_ROOT = PROJECT_ROOT / "data" / "factories"


def main() -> None:
    dataframe = pd.read_csv(DATA_PATH)

    y = dataframe[TARGET].to_numpy(dtype=int)

    train_indices, validation_indices, test_indices = (
        stratified_split(
            y=y,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42,
        )
    )

    print("Total samples:", len(dataframe))
    print("Train samples:", len(train_indices))
    print("Validation samples:", len(validation_indices))
    print("Test samples:", len(test_indices))

    # 检查三组索引没有重复，也没有遗漏。
    all_indices = np.concatenate(
        [
            train_indices,
            validation_indices,
            test_indices,
        ]
    )

    assert len(all_indices) == len(dataframe)
    assert len(np.unique(all_indices)) == len(dataframe)

    train_ids = set(
        dataframe.iloc[train_indices][ID_COL]
        .astype(int)
        .tolist()
    )

    clients = load_clients(
        strategy_directory=FACTORY_ROOT / "iid",
        train_ids=train_ids,
    )

    print("\nNumber of clients:", len(clients))

    total_client_rows = 0

    for client in clients:
        print(
            client["name"],
            "| rows:",
            client["rows"],
            "| failure rate:",
            round(client["failure_rate"], 4),
            "| x shape:",
            client["x"].shape,
            "| y shape:",
            client["y"].shape,
        )

        assert client["x"].shape[0] == client["rows"]
        assert client["y"].shape[0] == client["rows"]
        assert client["x"].shape[1] == 10

        total_client_rows += client["rows"]

    print("\nTotal client training rows:", total_client_rows)

    assert len(clients) == 5
    assert total_client_rows == len(train_indices)

    print("\nData utilities test passed.")


if __name__ == "__main__":
    main()