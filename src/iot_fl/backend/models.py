from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
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
