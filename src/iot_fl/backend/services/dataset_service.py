from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

import prepare_ai4i_dataset as preparation
from iot_fl.backend.config import PROJECT_ROOT
from iot_fl.backend.models import UploadedDataset, User
from iot_fl.config import FEATURES, ID_COL, STRATEGIES, TARGET


UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
RAW_FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


class DatasetValidationError(ValueError):
    pass


def list_datasets(db: Session, current_user: User) -> list[UploadedDataset]:
    statement = select(UploadedDataset).order_by(
        UploadedDataset.created_at.desc(),
        UploadedDataset.id.desc(),
    )
    if current_user.role != "admin":
        statement = statement.where(UploadedDataset.user_id == current_user.id)
    return list(db.scalars(statement).all())


def get_dataset(
    db: Session,
    current_user: User,
    dataset_id: int,
) -> UploadedDataset | None:
    dataset = db.get(UploadedDataset, dataset_id)
    if dataset is None:
        return None
    if current_user.role != "admin" and dataset.user_id != current_user.id:
        return None
    return dataset


def save_uploaded_dataset(
    db: Session,
    current_user: User,
    filename: str,
    source_file,
) -> UploadedDataset:
    if not filename.lower().endswith(".csv"):
        raise DatasetValidationError("Only CSV dataset uploads are supported.")

    dataset_token = uuid4().hex
    dataset_dir = UPLOAD_ROOT / f"user_{current_user.id}" / dataset_token
    factory_root = dataset_dir / "factories"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(filename).name or "dataset.csv"
    stored_path = dataset_dir / safe_filename
    with stored_path.open("wb") as handle:
        shutil.copyfileobj(source_file, handle)

    processed_path = dataset_dir / "processed.csv"
    try:
        clean = prepare_dataset_files(stored_path, processed_path, factory_root)
        status = "READY"
        error_message = None
        rows = int(clean.shape[0])
        columns = int(clean.shape[1])
    except Exception as error:
        status = "FAILED"
        error_message = str(error)[:1000]
        rows = 0
        columns = 0

    dataset = UploadedDataset(
        user_id=current_user.id,
        original_filename=safe_filename,
        stored_path=str(stored_path),
        processed_path=str(processed_path),
        factory_root=str(factory_root),
        rows=rows,
        columns=columns,
        status=status,
        error_message=error_message,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    if status == "FAILED":
        raise DatasetValidationError(error_message or "Dataset validation failed.")
    return dataset


def prepare_dataset_files(
    stored_path: Path,
    processed_path: Path,
    factory_root: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(stored_path)
    validate_uploaded_frame(raw)
    raw = raw.copy()
    for mode in FAILURE_MODES:
        if mode not in raw.columns:
            raw[mode] = 0

    clean, _scaler = preparation.clean_and_standardize(raw)
    missing_features = [column for column in (ID_COL, TARGET, *FEATURES) if column not in clean.columns]
    if missing_features:
        raise DatasetValidationError(f"Dataset cannot be converted for experiments; missing {missing_features}.")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_path, index=False)

    partitions = {
        "iid": preparation.stratified_iid_split(clean, n_clients=5, seed=42),
        "moderate_non_iid": preparation.moderate_non_iid_split(clean, n_clients=5, seed=42),
        "highly_non_iid": preparation.highly_non_iid_split(clean, n_clients=5, seed=42),
    }
    preparation.export_partitions(partitions, factory_root)
    return clean


def validate_uploaded_frame(frame: pd.DataFrame) -> None:
    required = {ID_COL, TARGET, *RAW_FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetValidationError(
            "Dataset must contain AI4I columns: " + ", ".join(missing)
        )
    if frame.empty:
        raise DatasetValidationError("Dataset is empty.")
    if frame[ID_COL].duplicated().any():
        raise DatasetValidationError("Dataset UDI values must be unique.")
    numeric_columns = [TARGET, *RAW_FEATURE_COLUMNS[1:]]
    for column in numeric_columns:
        pd.to_numeric(frame[column], errors="raise")
    invalid_target = sorted(set(frame[TARGET].astype(int)) - {0, 1})
    if invalid_target:
        raise DatasetValidationError("Machine failure must contain only 0/1 labels.")
    invalid_type = sorted(set(frame["Type"].astype(str)) - {"L", "M", "H"})
    if invalid_type:
        raise DatasetValidationError("Type must contain only L, M, or H values.")

