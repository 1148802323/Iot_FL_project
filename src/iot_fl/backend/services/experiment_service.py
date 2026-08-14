from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from iot_fl.algorithms import run_experiment
from iot_fl.backend.models import Experiment, User
from iot_fl.backend.schemas import ExperimentCreate
from iot_fl.backend.services.dataset_service import get_dataset


PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_experiment(
    db: Session,
    current_user: User,
    payload: ExperimentCreate,
) -> Experiment:
    if payload.dataset_id is not None:
        dataset = get_dataset(db, current_user, payload.dataset_id)
        if dataset is None:
            raise ValueError("Dataset not found.")
        if dataset.status != "READY":
            raise ValueError("Dataset is not ready for experiments.")

    experiment = Experiment(
        user_id=current_user.id,
        dataset_id=payload.dataset_id,
        algorithm=payload.algorithm,
        distribution=payload.distribution,
        rounds=payload.rounds,
        local_epochs=payload.local_epochs,
        learning_rate=payload.learning_rate,
        parameters=payload.parameters,
        status=PENDING,
        convergence_history=[],
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def list_experiments(db: Session, current_user: User) -> list[Experiment]:
    statement = select(Experiment).order_by(Experiment.created_at.desc(), Experiment.id.desc())
    if current_user.role != "admin":
        statement = statement.where(Experiment.user_id == current_user.id)
    return list(db.scalars(statement).all())


def get_experiment(
    db: Session,
    current_user: User,
    experiment_id: int,
) -> Experiment | None:
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        return None
    if current_user.role != "admin" and experiment.user_id != current_user.id:
        return None
    return experiment


def run_managed_experiment(db: Session, experiment: Experiment) -> Experiment:
    if experiment.status == RUNNING:
        return experiment

    experiment.status = RUNNING
    experiment.started_at = utc_now()
    experiment.finished_at = None
    experiment.error_message = None
    clear_result_fields(experiment)
    db.commit()
    db.refresh(experiment)

    try:
        config = {
            "rounds": experiment.rounds,
            "local_epochs": experiment.local_epochs,
            "learning_rate": experiment.learning_rate,

        }

        config.update(experiment.parameters or {})

        if experiment.dataset is not None:
            config["data_path"] = experiment.dataset.processed_path
            config["factory_root"] = experiment.dataset.factory_root
        result = run_experiment(
            algorithm=experiment.algorithm,
            distribution=experiment.distribution,
            config=config,
        )
    except Exception as error:
        experiment.status = FAILED
        experiment.error_message = str(error)[:1000]
        experiment.finished_at = utc_now()
        db.commit()
        db.refresh(experiment)
        return experiment

    store_result(experiment, result)
    experiment.status = COMPLETED
    experiment.finished_at = utc_now()
    db.commit()
    db.refresh(experiment)
    return experiment


def clear_result_fields(experiment: Experiment) -> None:
    experiment.accuracy = None
    experiment.precision = None
    experiment.recall = None
    experiment.f1_score = None
    experiment.communication_cost = None
    experiment.training_time = None
    experiment.convergence_history = []


def store_result(experiment: Experiment, result: dict[str, Any]) -> None:
    experiment.accuracy = optional_float(result.get("accuracy"))
    experiment.precision = optional_float(result.get("precision"))
    experiment.recall = optional_float(result.get("recall"))
    experiment.f1_score = optional_float(result.get("f1"))
    experiment.communication_cost = optional_float(result.get("communication_cost"))
    experiment.training_time = optional_float(result.get("training_time"))
    history = result.get("convergence_history")
    experiment.convergence_history = history if isinstance(history, list) else []


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
