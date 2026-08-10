from __future__ import annotations

import csv
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from iot_fl.algorithms.registry import ALGORITHM_REGISTRY
from iot_fl.backend.config import PROJECT_ROOT, settings
from iot_fl.backend.database import SessionLocal, get_db, init_db
from iot_fl.backend.dependencies import get_current_user, require_admin
from iot_fl.backend.models import Client, Experiment, Factory, User
from iot_fl.backend.schemas import (
    AlgorithmRead,
    ClientExperimentRead,
    ClientRead,
    ClientStatistics,
    ExperimentCreate,
    ExperimentResponse,
    ExperimentResult,
    HealthResponse,
    MessageResponse,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserRead,
)
from iot_fl.backend.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from iot_fl.backend.seed import ensure_factory_clients
from iot_fl.backend.services.experiment_service import (
    create_experiment,
    get_experiment,
    list_experiments,
    run_managed_experiment,
)
from iot_fl.config import TARGET


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    del app_instance
    init_db()
    with SessionLocal() as db:
        ensure_factory_clients(db)
    yield


app = FastAPI(
    title="IoT FL Predictive Maintenance API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/algorithms", response_model=list[AlgorithmRead])
def list_algorithm_options(current_user: User = Depends(get_current_user)) -> list[AlgorithmRead]:
    del current_user
    return [
        AlgorithmRead(
            name=name,
            display_name=adapter.display_name,
            implementation_file=adapter.implementation_file,
        )
        for name, adapter in sorted(ALGORITHM_REGISTRY.items())
    ]


@app.post(
    "/api/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: UserCreate,
    db: Session = Depends(get_db),
) -> TokenResponse:
    existing_user = db.scalar(
        select(User).where(
            or_(
                User.username == payload.username,
                User.email == payload.email,
            )
        )
    )
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists",
        )

    if payload.role == "client":
        if payload.factory_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client users must be bound to a factory_id",
            )
        factory = db.get(Factory, payload.factory_id)
        if factory is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="factory_id does not exist",
            )
    elif payload.factory_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin users must not be bound to a factory_id",
        )

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        factory_id=payload.factory_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(
        subject=user.username,
        role=user.role,
    )
    return TokenResponse(
        access_token=token,
        user=UserRead.model_validate(user),
    )


@app.post("/api/auth/login", response_model=TokenResponse)
def login(
    payload: UserLogin,
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = db.scalar(select(User).where(User.username == payload.username))
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    token = create_access_token(
        subject=user.username,
        role=user.role,
    )
    return TokenResponse(
        access_token=token,
        user=UserRead.model_validate(user),
    )


@app.get("/api/auth/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current_user)


@app.post("/api/auth/logout", response_model=MessageResponse)
def logout(current_user: User = Depends(get_current_user)) -> MessageResponse:
    del current_user
    return MessageResponse(message="Logged out")


def _client_visible_to_user(client: Client, user: User) -> bool:
    return user.role == "admin" or user.factory_id == client.factory_id


def _get_authorized_client(
    client_id: int,
    db: Session,
    current_user: User,
) -> Client:
    client = db.get(Client, client_id)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found",
        )
    if not _client_visible_to_user(client, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client belongs to another factory",
        )
    return client


def _serialize_client(client: Client, *, include_dataset_path: bool) -> ClientRead:
    payload = ClientRead.model_validate(client)
    if not include_dataset_path:
        payload.dataset_path = None
    return payload


def _dataset_absolute_path(client: Client) -> Path:
    dataset_path = Path(client.dataset_path)
    if dataset_path.is_absolute():
        return dataset_path
    return PROJECT_ROOT / dataset_path


def _statistics_for_client(client: Client) -> ClientStatistics:
    dataset_path = _dataset_absolute_path(client)
    total_rows = client.train_rows + client.validation_rows
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
                    failure_modes[failure_mode] = (
                        failure_modes.get(failure_mode, 0) + 1
                    )
        failure_ratio = failure_count / total_rows if total_rows else 0.0
    else:
        failure_count = client.failure_count
        failure_ratio = client.failure_ratio

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


def _load_experiment_rows(distribution_type: str) -> list[ClientExperimentRead]:
    experiment_files = [
        PROJECT_ROOT / "reports" / "fedavg_baseline_results.csv",
        PROJECT_ROOT / "reports" / "failure_aware_fedavg_results.csv",
    ]
    experiments: list[ClientExperimentRead] = []

    for path in experiment_files:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("strategy") != distribution_type:
                    continue
                experiments.append(
                    ClientExperimentRead(
                        strategy=row["strategy"],
                        method=row["method"],
                        rounds=int(row["rounds"]),
                        local_epochs=int(row["local_epochs"]),
                        threshold=float(row["threshold"]),
                        accuracy=float(row["accuracy"]),
                        precision=float(row["precision"]),
                        recall=float(row["recall"]),
                        f1=float(row["f1"]),
                    )
                )

    return experiments


def _get_visible_experiment(
    experiment_id: int,
    db: Session,
    current_user: User,
) -> Experiment:
    experiment = get_experiment(db, current_user, experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    return experiment


@app.get("/api/experiments", response_model=list[ExperimentResponse])
def list_user_experiments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ExperimentResponse]:
    return [
        ExperimentResponse.model_validate(experiment)
        for experiment in list_experiments(db, current_user)
    ]


@app.post(
    "/api/experiments",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_user_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ExperimentResponse:
    experiment = create_experiment(db, current_user, payload)
    return ExperimentResponse.model_validate(experiment)


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
def get_user_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ExperimentResponse:
    experiment = _get_visible_experiment(experiment_id, db, current_user)
    return ExperimentResponse.model_validate(experiment)


@app.post("/api/experiments/{experiment_id}/run", response_model=ExperimentResponse)
def run_user_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ExperimentResponse:
    experiment = _get_visible_experiment(experiment_id, db, current_user)
    if experiment.status == "RUNNING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Experiment is already running",
        )
    experiment = run_managed_experiment(db, experiment)
    return ExperimentResponse.model_validate(experiment)


@app.get("/api/experiments/{experiment_id}/results", response_model=ExperimentResult)
def get_user_experiment_results(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ExperimentResult:
    experiment = _get_visible_experiment(experiment_id, db, current_user)
    return ExperimentResult.model_validate(experiment)


@app.get(
    "/api/clients",
    response_model=list[ClientRead],
    response_model_exclude_none=True,
)
def list_clients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ClientRead]:
    statement = select(Client).order_by(Client.distribution_type, Client.factory_id)
    if current_user.role != "admin":
        statement = statement.where(Client.factory_id == current_user.factory_id)

    clients = db.scalars(statement).all()
    return [
        _serialize_client(
            client,
            include_dataset_path=current_user.role == "admin",
        )
        for client in clients
    ]


@app.get(
    "/api/clients/{client_id}",
    response_model=ClientRead,
    response_model_exclude_none=True,
)
def get_client(
    client_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClientRead:
    client = _get_authorized_client(client_id, db, current_user)
    return _serialize_client(
        client,
        include_dataset_path=current_user.role == "admin",
    )


@app.get(
    "/api/clients/{client_id}/statistics",
    response_model=ClientStatistics,
)
def get_client_statistics(
    client_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClientStatistics:
    client = _get_authorized_client(client_id, db, current_user)
    return _statistics_for_client(client)


@app.get(
    "/api/clients/{client_id}/experiments",
    response_model=list[ClientExperimentRead],
)
def get_client_experiments(
    client_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ClientExperimentRead]:
    client = _get_authorized_client(client_id, db, current_user)
    return _load_experiment_rows(client.distribution_type)


@app.get("/api/admin/users", response_model=list[UserRead])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
) -> list[UserRead]:
    del current_user
    users = db.scalars(select(User).order_by(User.id)).all()
    return [UserRead.model_validate(user) for user in users]
