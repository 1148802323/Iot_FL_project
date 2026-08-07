from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from iot_fl.backend.config import settings
from iot_fl.backend.database import SessionLocal, get_db, init_db
from iot_fl.backend.dependencies import get_current_user, require_admin
from iot_fl.backend.models import Factory, User
from iot_fl.backend.schemas import (
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
from iot_fl.backend.seed import ensure_default_factories


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    del app_instance
    init_db()
    with SessionLocal() as db:
        ensure_default_factories(db)
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


@app.get("/api/admin/users", response_model=list[UserRead])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
) -> list[UserRead]:
    del current_user
    users = db.scalars(select(User).order_by(User.id)).all()
    return [UserRead.model_validate(user) for user in users]
