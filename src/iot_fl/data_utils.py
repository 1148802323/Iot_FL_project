from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from iot_fl.config import FEATURES, ID_COL, TARGET


class ClientDataset(TypedDict):
    """Data and metadata belonging to one federated client."""

    name: str
    rows: int
    failure_rate: float
    x: np.ndarray
    y: np.ndarray


def stratified_split(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train, validation and test indices.

    Positive and negative samples are split separately so that each subset
    has approximately the same class distribution.
    """
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0].copy()
        rng.shuffle(label_indices)

        train_count = int(round(len(label_indices) * train_ratio))
        validation_count = int(round(len(label_indices) * val_ratio))

        train_indices.extend(
            label_indices[:train_count].tolist()
        )

        validation_indices.extend(
            label_indices[
                train_count:train_count + validation_count
            ].tolist()
        )

        test_indices.extend(
            label_indices[
                train_count + validation_count:
            ].tolist()
        )

    for indices in (
        train_indices,
        validation_indices,
        test_indices,
    ):
        rng.shuffle(indices)

    return (
        np.array(train_indices),
        np.array(validation_indices),
        np.array(test_indices),
    )


def load_clients(
    strategy_directory: Path,
    train_ids: set[int],
) -> list[ClientDataset]:
    """
    Load all factory clients belonging to one data-distribution strategy.

    Only rows belonging to the global training split are retained.
    """
    clients: list[ClientDataset] = []

    for factory_path in sorted(
        strategy_directory.glob("factory_*.csv")
    ):
        dataframe = pd.read_csv(factory_path)

        client_dataframe = dataframe[
            dataframe[ID_COL].isin(train_ids)
        ].copy()

        if client_dataframe.empty:
            continue

        clients.append(
            {
                "name": factory_path.stem,
                "rows": len(client_dataframe),
                "failure_rate": float(
                    client_dataframe[TARGET].mean()
                ),
                "x": client_dataframe[
                    FEATURES
                ].to_numpy(dtype=float),
                "y": client_dataframe[
                    TARGET
                ].to_numpy(dtype=int),
            }
        )

    if not clients:
        raise ValueError(
            f"No training clients found in {strategy_directory}"
        )

    return clients