from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("JWT_SECRET", "test-secret-for-auth-tests")

from iot_fl.backend.database import Base, get_db
from iot_fl.backend.main import app
from iot_fl.backend.models import Client as FactoryClient
from iot_fl.backend.models import Factory, User
from iot_fl.backend.seed import ensure_factory_clients


@pytest.fixture()
def client(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'test_auth.db'}"
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    Base.metadata.create_all(bind=engine)

    factory_one_path = tmp_path / "factory_01.csv"
    factory_one_path.write_text(
        "\n".join(
            [
                "UDI,Machine failure,failure_mode",
                "1,0,None",
                "2,1,HDF",
                "3,0,None",
                "4,1,PWF",
            ]
        ),
        encoding="utf-8",
    )
    factory_two_path = tmp_path / "factory_02.csv"
    factory_two_path.write_text(
        "\n".join(
            [
                "UDI,Machine failure,failure_mode",
                "5,0,None",
                "6,0,None",
            ]
        ),
        encoding="utf-8",
    )

    with TestingSessionLocal() as db:
        db.add_all(
            [
                Factory(id=1, name="factory_01", description="Test factory 1"),
                Factory(id=2, name="factory_02", description="Test factory 2"),
            ]
        )
        db.add_all(
            [
                FactoryClient(
                    id=1,
                    factory_id=1,
                    name="factory_01_iid",
                    distribution_type="iid",
                    dataset_path=str(factory_one_path),
                    train_rows=3,
                    validation_rows=1,
                    failure_count=2,
                    failure_ratio=0.5,
                    status="active",
                ),
                FactoryClient(
                    id=2,
                    factory_id=2,
                    name="factory_02_iid",
                    distribution_type="iid",
                    dataset_path=str(factory_two_path),
                    train_rows=2,
                    validation_rows=0,
                    failure_count=0,
                    failure_ratio=0.0,
                    status="active",
                ),
            ]
        )
        db.commit()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def register_user(
    client: TestClient,
    *,
    username: str,
    email: str,
    role: str,
    factory_id: int | None = None,
    password: str = "strong-password",
):
    payload = {
        "username": username,
        "email": email,
        "password": password,
        "role": role,
        "factory_id": factory_id,
    }
    return client.post("/api/auth/register", json=payload)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_check(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_hashes_password_and_login_me_logout(client: TestClient) -> None:
    response = register_user(
        client,
        username="factory_user",
        email="factory@example.com",
        role="client",
        factory_id=1,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["role"] == "client"
    assert body["user"]["factory_id"] == 1

    with next(app.dependency_overrides[get_db]()) as db:
        user = db.scalar(select(User).where(User.username == "factory_user"))
        assert user is not None
        assert user.hashed_password != "strong-password"
        assert user.hashed_password.startswith("$2")

    login = client.post(
        "/api/auth/login",
        json={"username": "factory_user", "password": "strong-password"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = client.get("/api/auth/me", headers=bearer(token))
    assert me.status_code == 200
    assert me.json()["username"] == "factory_user"

    logout = client.post("/api/auth/logout", headers=bearer(token))
    assert logout.status_code == 200
    assert logout.json() == {"message": "Logged out"}


def test_login_rejects_wrong_password(client: TestClient) -> None:
    register_user(
        client,
        username="wrong_password_user",
        email="wrong@example.com",
        role="client",
        factory_id=1,
    )

    response = client.post(
        "/api/auth/login",
        json={"username": "wrong_password_user", "password": "bad-password"},
    )
    assert response.status_code == 401


def test_jwt_required_for_me(client: TestClient) -> None:
    response = client.get("/api/auth/me")
    assert response.status_code == 401

    response = client.get("/api/auth/me", headers=bearer("not-a-real-token"))
    assert response.status_code == 401


def test_inactive_user_cannot_login_or_use_existing_token(client: TestClient) -> None:
    response = register_user(
        client,
        username="inactive_user",
        email="inactive@example.com",
        role="client",
        factory_id=1,
    )
    token = response.json()["access_token"]

    with next(app.dependency_overrides[get_db]()) as db:
        user = db.scalar(select(User).where(User.username == "inactive_user"))
        assert user is not None
        user.is_active = False
        db.commit()

    login = client.post(
        "/api/auth/login",
        json={"username": "inactive_user", "password": "strong-password"},
    )
    assert login.status_code == 403

    me = client.get("/api/auth/me", headers=bearer(token))
    assert me.status_code == 403


def test_role_isolation_for_admin_only_api(client: TestClient) -> None:
    admin_response = register_user(
        client,
        username="admin_user",
        email="admin@example.com",
        role="admin",
    )
    client_response = register_user(
        client,
        username="client_user",
        email="client@example.com",
        role="client",
        factory_id=1,
    )

    admin_token = admin_response.json()["access_token"]
    client_token = client_response.json()["access_token"]

    admin_users = client.get("/api/admin/users", headers=bearer(admin_token))
    assert admin_users.status_code == 200
    assert {user["username"] for user in admin_users.json()} == {
        "admin_user",
        "client_user",
    }

    client_users = client.get("/api/admin/users", headers=bearer(client_token))
    assert client_users.status_code == 403


def test_client_must_bind_factory_and_admin_must_not(client: TestClient) -> None:
    missing_factory = register_user(
        client,
        username="client_without_factory",
        email="nofactory@example.com",
        role="client",
    )
    assert missing_factory.status_code == 400

    admin_with_factory = register_user(
        client,
        username="admin_with_factory",
        email="adminfactory@example.com",
        role="admin",
        factory_id=1,
    )
    assert admin_with_factory.status_code == 400


def test_clients_api_hides_dataset_path_for_client_users(
    client: TestClient,
) -> None:
    admin_response = register_user(
        client,
        username="client_api_admin",
        email="clientapiadmin@example.com",
        role="admin",
    )
    client_response = register_user(
        client,
        username="factory_one_client",
        email="factoryone@example.com",
        role="client",
        factory_id=1,
    )

    admin_clients = client.get(
        "/api/clients",
        headers=bearer(admin_response.json()["access_token"]),
    )
    assert admin_clients.status_code == 200
    assert {item["id"] for item in admin_clients.json()} == {1, 2}
    assert all("dataset_path" in item for item in admin_clients.json())

    factory_clients = client.get(
        "/api/clients",
        headers=bearer(client_response.json()["access_token"]),
    )
    assert factory_clients.status_code == 200
    assert [item["id"] for item in factory_clients.json()] == [1]
    assert "dataset_path" not in factory_clients.json()[0]


def test_client_cannot_access_another_factory_client(
    client: TestClient,
) -> None:
    client_response = register_user(
        client,
        username="factory_one_limited_client",
        email="factoryonelimited@example.com",
        role="client",
        factory_id=1,
    )

    response = client.get(
        "/api/clients/2",
        headers=bearer(client_response.json()["access_token"]),
    )

    assert response.status_code == 403


def test_client_statistics_match_csv(
    client: TestClient,
) -> None:
    client_response = register_user(
        client,
        username="stats_client",
        email="stats@example.com",
        role="client",
        factory_id=1,
    )

    response = client.get(
        "/api/clients/1/statistics",
        headers=bearer(client_response.json()["access_token"]),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == 4
    assert body["failure_count"] == 2
    assert body["failure_ratio"] == 0.5
    assert body["failure_modes"] == {"None": 2, "HDF": 1, "PWF": 1}


def test_unknown_client_id_returns_404(client: TestClient) -> None:
    admin_response = register_user(
        client,
        username="missing_client_admin",
        email="missingclientadmin@example.com",
        role="admin",
    )

    response = client.get(
        "/api/clients/999",
        headers=bearer(admin_response.json()["access_token"]),
    )

    assert response.status_code == 404


def test_factory_client_seed_is_idempotent(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    processed_root = data_root / "processed"
    iid_root = data_root / "factories" / "iid"
    processed_root.mkdir(parents=True)
    iid_root.mkdir(parents=True)

    processed_root.joinpath("ai4i_clean_standardized.csv").write_text(
        "\n".join(
            [
                "UDI,Machine failure",
                "1,0",
                "2,0",
                "3,0",
                "4,0",
                "5,0",
                "6,1",
                "7,1",
                "8,1",
                "9,1",
                "10,1",
            ]
        ),
        encoding="utf-8",
    )
    iid_root.joinpath("factory_01.csv").write_text(
        "\n".join(
            [
                "UDI,Machine failure,failure_mode",
                "1,0,None",
                "2,0,None",
                "3,0,None",
                "4,0,None",
                "5,0,None",
                "6,1,HDF",
                "7,1,HDF",
                "8,1,PWF",
                "9,1,PWF",
                "10,1,OSF",
            ]
        ),
        encoding="utf-8",
    )

    engine = create_engine(
        f"sqlite:///{tmp_path / 'seed.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    with TestingSessionLocal() as db:
        ensure_factory_clients(db, project_root=tmp_path)
        ensure_factory_clients(db, project_root=tmp_path)

        clients = db.scalars(select(FactoryClient)).all()
        assert len(clients) == 1
        assert clients[0].failure_count == 5
        assert clients[0].failure_ratio == 0.5
        assert clients[0].train_rows == 6
        assert clients[0].validation_rows == 2
