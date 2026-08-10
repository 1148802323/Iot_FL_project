from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Role = Literal["admin", "client"]


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
