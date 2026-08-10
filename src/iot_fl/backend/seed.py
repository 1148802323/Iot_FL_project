from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from iot_fl.backend.config import PROJECT_ROOT
from iot_fl.backend.database import SessionLocal, init_db
from iot_fl.backend.models import Client, Factory
from iot_fl.config import ID_COL, STRATEGIES, TARGET
from iot_fl.data_utils import stratified_split


DEFAULT_FACTORIES = [
    ("factory_01", "Simulated factory client 1"),
    ("factory_02", "Simulated factory client 2"),
    ("factory_03", "Simulated factory client 3"),
    ("factory_04", "Simulated factory client 4"),
    ("factory_05", "Simulated factory client 5"),
]


def ensure_default_factories(db: Session) -> None:
    for name, description in DEFAULT_FACTORIES:
        exists = db.scalar(select(Factory).where(Factory.name == name))
        if exists is None:
            db.add(Factory(name=name, description=description))
    db.commit()


def _split_ids(project_root: Path) -> tuple[set[int], set[int]]:
    dataset_path = project_root / "data" / "processed" / "ai4i_clean_standardized.csv"
    if not dataset_path.exists():
        return set(), set()

    dataframe = pd.read_csv(dataset_path)
    train_idx, validation_idx, _ = stratified_split(
        dataframe[TARGET].to_numpy(dtype=int),
        train_ratio=0.6,
        val_ratio=0.2,
        seed=42,
    )
    train_ids = set(dataframe.loc[train_idx, ID_COL].astype(int).tolist())
    validation_ids = set(dataframe.loc[validation_idx, ID_COL].astype(int).tolist())
    return train_ids, validation_ids


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_factory_clients(
    db: Session,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Create or update client records from existing factory CSV partitions."""
    ensure_default_factories(db)

    factory_root = project_root / "data" / "factories"
    train_ids, validation_ids = _split_ids(project_root)

    for distribution_type in STRATEGIES:
        distribution_dir = factory_root / distribution_type
        if not distribution_dir.exists():
            continue

        for csv_path in sorted(distribution_dir.glob("factory_*.csv")):
            dataframe = pd.read_csv(csv_path)
            factory_name = csv_path.stem
            factory = db.scalar(select(Factory).where(Factory.name == factory_name))
            if factory is None:
                factory = Factory(
                    name=factory_name,
                    description=f"Simulated factory client {factory_name[-2:]}",
                )
                db.add(factory)
                db.flush()

            if train_ids:
                train_rows = int(dataframe[ID_COL].astype(int).isin(train_ids).sum())
            else:
                train_rows = len(dataframe)

            validation_rows = (
                int(dataframe[ID_COL].astype(int).isin(validation_ids).sum())
                if validation_ids
                else 0
            )
            failure_count = int(dataframe[TARGET].sum())
            failure_ratio = (
                float(dataframe[TARGET].mean())
                if len(dataframe) > 0
                else 0.0
            )

            client = db.scalar(
                select(Client).where(
                    Client.factory_id == factory.id,
                    Client.distribution_type == distribution_type,
                )
            )
            if client is None:
                client = Client(
                    factory_id=factory.id,
                    distribution_type=distribution_type,
                )
                db.add(client)

            client.name = f"{factory_name}_{distribution_type}"
            client.dataset_path = _relative_path(csv_path, project_root)
            client.train_rows = train_rows
            client.validation_rows = validation_rows
            client.failure_count = failure_count
            client.failure_ratio = failure_ratio
            client.status = "active"

    db.commit()


def seed_factories() -> None:
    init_db()
    with SessionLocal() as db:
        ensure_factory_clients(db)


if __name__ == "__main__":
    seed_factories()
    print("Seeded factories and clients")
