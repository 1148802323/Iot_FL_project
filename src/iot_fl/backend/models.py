from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from iot_fl.backend.database import Base


class Factory(Base):
    __tablename__ = "factories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    users: Mapped[list["User"]] = relationship(back_populates="factory")
    clients: Mapped[list["Client"]] = relationship(back_populates="factory")


class Client(Base):
    __tablename__ = "clients"
    __table_args__ = (
        UniqueConstraint(
            "factory_id",
            "distribution_type",
            name="uq_clients_factory_distribution",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    factory_id: Mapped[int] = mapped_column(
        ForeignKey("factories.id"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    distribution_type: Mapped[str] = mapped_column(
        String(50),
        index=True,
        nullable=False,
    )
    dataset_path: Mapped[str] = mapped_column(String(500), nullable=False)
    train_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    validation_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)

    factory: Mapped[Factory] = relationship(back_populates="clients")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="client", nullable=False)
    factory_id: Mapped[int | None] = mapped_column(
        ForeignKey("factories.id"),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    factory: Mapped[Factory | None] = relationship(back_populates="users")
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="user")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
        nullable=False,
    )
    algorithm: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    distribution: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    local_epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    communication_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    convergence_history: Mapped[list[dict[str, object]] | None] = mapped_column(
        JSON,
        default=list,
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="experiments")
