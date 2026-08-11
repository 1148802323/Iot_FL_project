from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from iot_fl.algorithms import list_algorithms, list_distributions


Role = Literal["admin", "client"]
ExperimentStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]


class FactoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None = None
    created_at: datetime


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=8, max_length=128)
    role: Role = "client"
    factory_id: int | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@"):
            raise ValueError("email must be valid")
        return normalized


class UserLogin(BaseModel):
    username: str
    password: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    role: Role
    factory_id: int | None
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class HealthResponse(BaseModel):
    status: str


class MessageResponse(BaseModel):
    message: str


class ClientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    factory_id: int
    name: str
    distribution_type: str
    dataset_path: str | None = None
    train_rows: int
    validation_rows: int
    failure_count: int
    failure_ratio: float
    last_active_at: datetime
    status: str


class ClientStatistics(BaseModel):
    id: int
    factory_id: int
    name: str
    distribution_type: str
    total_rows: int
    train_rows: int
    validation_rows: int
    failure_count: int
    failure_ratio: float
    failure_modes: dict[str, int] = Field(default_factory=dict)


class ClientExperimentRead(BaseModel):
    strategy: str
    method: str
    rounds: int
    local_epochs: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class AlgorithmRead(BaseModel):
    name: str
    display_name: str
    implementation_file: str


class ExperimentCreate(BaseModel):
    algorithm: str
    distribution: str
    rounds: int = Field(gt=0)
    local_epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    dataset_id: int | None = Field(default=None, gt=0)

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in list_algorithms():
            raise ValueError(f"Unknown algorithm '{value}'")
        return normalized

    @field_validator("distribution")
    @classmethod
    def validate_distribution(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in list_distributions():
            raise ValueError(f"Unknown distribution '{value}'")
        return normalized


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    dataset_id: int | None = None
    algorithm: str
    distribution: str
    rounds: int
    local_epochs: int
    learning_rate: float
    status: ExperimentStatus
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    communication_cost: float | None = None
    training_time: float | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExperimentResult(ExperimentResponse):
    convergence_history: list[dict[str, object]] = Field(default_factory=list)


class RecentModelPerformance(BaseModel):
    experiment_id: int
    algorithm: str
    distribution: str
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    communication_cost: float | None = None
    training_time: float | None = None
    finished_at: datetime | None = None


class ClientDashboardResponse(BaseModel):
    role: Literal["client"] = "client"
    factory_id: int
    factory_name: str
    total_samples: int
    failure_samples: int
    failure_rate: float
    distribution_types: list[str]
    dominant_failure_mode: str | None = None
    failure_modes: dict[str, int] = Field(default_factory=dict)
    clients: list[ClientStatistics] = Field(default_factory=list)
    recent_experiments: list[ExperimentResponse] = Field(default_factory=list)
    recent_model_performance: list[RecentModelPerformance] = Field(default_factory=list)


class AdminDashboardResponse(BaseModel):
    role: Literal["admin"] = "admin"
    registered_users: int
    factory_clients: int
    factory_ids: list[int]
    experiment_count: int
    algorithm_usage: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    recent_experiments: list[ExperimentResponse] = Field(default_factory=list)


class ExperimentCompareRequest(BaseModel):
    experiment_ids: list[int] = Field(min_length=1, max_length=12)


class ExperimentComparisonRow(BaseModel):
    id: int
    algorithm: str
    distribution: str
    rounds: int
    local_epochs: int
    learning_rate: float
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    communication_cost: float | None = None
    training_time: float | None = None
    created_at: datetime


class ExperimentConvergenceSeries(BaseModel):
    experiment_id: int
    algorithm: str
    distribution: str
    history: list[dict[str, object]] = Field(default_factory=list)


class ExperimentComparisonResponse(BaseModel):
    experiments: list[ExperimentComparisonRow]
    convergence: list[ExperimentConvergenceSeries] = Field(default_factory=list)


class UploadedDatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    original_filename: str
    rows: int
    columns: int
    status: str
    error_message: str | None = None
    created_at: datetime
