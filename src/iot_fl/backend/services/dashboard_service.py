from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from iot_fl.backend.config import PROJECT_ROOT
from iot_fl.backend.models import Client, Experiment, Factory, User
from iot_fl.backend.schemas import (
    AdminDashboardResponse,
    ClientDashboardResponse,
    ClientStatistics,
    ExperimentComparisonResponse,
    ExperimentComparisonRow,
    ExperimentConvergenceSeries,
    ExperimentResponse,
    RecentModelPerformance,
)
from iot_fl.config import TARGET


class DashboardAccessError(ValueError):
    """Raised when a user cannot access the requested dashboard."""


class ExperimentComparisonError(ValueError):
    """Raised when selected experiments cannot be compared."""


def build_client_dashboard(
    db: Session,
    current_user: User,
) -> ClientDashboardResponse:
    if current_user.factory_id is None:
        raise DashboardAccessError("Client dashboard requires an assigned factory.")

    factory = db.get(Factory, current_user.factory_id)
    if factory is None:
        raise DashboardAccessError("Assigned factory was not found.")

    clients = list(
        db.scalars(
            select(Client)
            .where(Client.factory_id == current_user.factory_id)
            .order_by(Client.distribution_type, Client.id)
        ).all()
    )
    client_stats = [_statistics_for_client(client) for client in clients]
    failure_modes = Counter[str]()
    for stats in client_stats:
        failure_modes.update(stats.failure_modes)

    total_samples = sum(stats.total_rows for stats in client_stats)
    failure_samples = sum(stats.failure_count for stats in client_stats)
    recent_experiments = list(
        db.scalars(
            select(Experiment)
            .where(Experiment.user_id == current_user.id)
            .order_by(Experiment.created_at.desc(), Experiment.id.desc())
            .limit(8)
        ).all()
    )
    recent_completed = [
        experiment
        for experiment in recent_experiments
        if experiment.status == "COMPLETED"
    ]

    return ClientDashboardResponse(
        factory_id=factory.id,
        factory_name=factory.name,
        total_samples=total_samples,
        failure_samples=failure_samples,
        failure_rate=failure_samples / total_samples if total_samples else 0.0,
        distribution_types=sorted({stats.distribution_type for stats in client_stats}),
        dominant_failure_mode=_dominant_failure_mode(failure_modes),
        failure_modes=dict(failure_modes),
        clients=client_stats,
        recent_experiments=[
            ExperimentResponse.model_validate(experiment)
            for experiment in recent_experiments
        ],
        recent_model_performance=[
            RecentModelPerformance(
                experiment_id=experiment.id,
                algorithm=experiment.algorithm,
                distribution=experiment.distribution,
                accuracy=experiment.accuracy,
                precision=experiment.precision,
                recall=experiment.recall,
                f1_score=experiment.f1_score,
                communication_cost=experiment.communication_cost,
                training_time=experiment.training_time,
                finished_at=experiment.finished_at,
            )
            for experiment in recent_completed[:5]
        ],
    )


def build_admin_dashboard(db: Session) -> AdminDashboardResponse:
    recent_experiments = list(
        db.scalars(
            select(Experiment)
            .order_by(Experiment.created_at.desc(), Experiment.id.desc())
            .limit(10)
        ).all()
    )
    return AdminDashboardResponse(
        registered_users=_count(db, User.id),
        factory_clients=_count(db, Client.id),
        factory_ids=list(db.scalars(select(Factory.id).order_by(Factory.id)).all()),
        experiment_count=_count(db, Experiment.id),
        algorithm_usage=_group_counts(db, Experiment.algorithm),
        status_counts=_group_counts(db, Experiment.status),
        recent_experiments=[
            ExperimentResponse.model_validate(experiment)
            for experiment in recent_experiments
        ],
    )


def compare_completed_experiments(
    db: Session,
    current_user: User,
    experiment_ids: list[int],
) -> ExperimentComparisonResponse:
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ExperimentComparisonError("Experiment IDs must be unique.")

    experiments = []
    for experiment_id in experiment_ids:
        experiment = db.get(Experiment, experiment_id)
        if experiment is None:
            raise LookupError(f"Experiment #{experiment_id} was not found.")
        if current_user.role != "admin" and experiment.user_id != current_user.id:
            raise LookupError(f"Experiment #{experiment_id} was not found.")
        if experiment.status != "COMPLETED":
            raise ExperimentComparisonError(
                f"Experiment #{experiment_id} is not completed."
            )
        experiments.append(experiment)

    return ExperimentComparisonResponse(
        experiments=[
            ExperimentComparisonRow(
                id=experiment.id,
                algorithm=experiment.algorithm,
                distribution=experiment.distribution,
                rounds=experiment.rounds,
                local_epochs=experiment.local_epochs,
                learning_rate=experiment.learning_rate,
                accuracy=experiment.accuracy,
                precision=experiment.precision,
                recall=experiment.recall,
                f1_score=experiment.f1_score,
                communication_cost=experiment.communication_cost,
                training_time=experiment.training_time,
                created_at=experiment.created_at,
            )
            for experiment in experiments
        ],
        convergence=[
            ExperimentConvergenceSeries(
                experiment_id=experiment.id,
                algorithm=experiment.algorithm,
                distribution=experiment.distribution,
                history=experiment.convergence_history or [],
            )
            for experiment in experiments
        ],
    )


def _count(db: Session, column: object) -> int:
    return int(db.scalar(select(func.count(column))) or 0)


def _group_counts(db: Session, column: object) -> dict[str, int]:
    rows = db.execute(select(column, func.count()).group_by(column)).all()
    return {
        str(key): int(count)
        for key, count in rows
        if key is not None
    }


def _statistics_for_client(client: Client) -> ClientStatistics:
    dataset_path = _dataset_absolute_path(client)
    total_rows = client.train_rows + client.validation_rows
    failure_count = client.failure_count
    failure_ratio = client.failure_ratio
    failure_modes: dict[str, int] = {}

    if dataset_path.exists():
        with dataset_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            total_rows = 0
            failure_count = 0
            for row in reader:
                total_rows += 1
                failure_count += int(row.get(TARGET, "0") or 0)
                failure_mode = row.get("failure_mode")
                if failure_mode:
                    failure_modes[failure_mode] = failure_modes.get(failure_mode, 0) + 1
        failure_ratio = failure_count / total_rows if total_rows else 0.0

    return ClientStatistics(
        id=client.id,
        factory_id=client.factory_id,
        name=client.name,
        distribution_type=client.distribution_type,
        total_rows=total_rows,
        train_rows=client.train_rows,
        validation_rows=client.validation_rows,
        failure_count=failure_count,
        failure_ratio=failure_ratio,
        failure_modes=failure_modes,
    )


def _dataset_absolute_path(client: Client) -> Path:
    dataset_path = Path(client.dataset_path)
    if dataset_path.is_absolute():
        return dataset_path
    return PROJECT_ROOT / dataset_path


def _dominant_failure_mode(failure_modes: Counter[str]) -> str | None:
    if not failure_modes:
        return None
    return failure_modes.most_common(1)[0][0]
