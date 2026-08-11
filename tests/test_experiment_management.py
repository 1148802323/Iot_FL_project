from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


os.environ.setdefault("JWT_SECRET", "test-secret-for-experiment-tests")

from iot_fl.backend.database import Base, get_db
from iot_fl.backend.main import app
from iot_fl.backend.models import Factory


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database_url = f"sqlite:///{tmp_path / 'test_experiments.db'}"
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
    testing_session = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        "iot_fl.backend.services.dataset_service.UPLOAD_ROOT",
        tmp_path / "uploads",
    )

    with testing_session() as db:
        db.add_all(
            [
                Factory(id=1, name="factory_01", description="Test factory 1"),
                Factory(id=2, name="factory_02", description="Test factory 2"),
            ]
        )
        db.commit()

    def override_get_db():
        db = testing_session()
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
    role: str = "client",
    factory_id: int | None = 1,
) -> str:
    payload = {
        "username": username,
        "email": f"{username}@example.com",
        "password": "strong-password",
        "role": role,
        "factory_id": factory_id if role == "client" else None,
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    return response.json()["access_token"]


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def experiment_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "algorithm": "fedavg",
        "distribution": "iid",
        "rounds": 5,
        "local_epochs": 1,
        "learning_rate": 0.01,
    }
    payload.update(overrides)
    return payload


def create_experiment(client: TestClient, token: str, **overrides: object) -> dict[str, object]:
    response = client.post(
        "/api/experiments",
        json=experiment_payload(**overrides),
        headers=bearer(token),
    )
    assert response.status_code == 201
    return response.json()


def sample_dataset_csv() -> str:
    rows = [
        "UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF",
    ]
    for index in range(1, 31):
        failure = 1 if index % 10 == 0 else 0
        rows.append(
            ",".join(
                [
                    str(index),
                    f"M{index:05d}",
                    ["L", "M", "H"][index % 3],
                    str(298.0 + index * 0.1),
                    str(308.0 + index * 0.1),
                    str(1300 + index * 7),
                    str(35.0 + index * 0.4),
                    str(40 + index),
                    str(failure),
                    "1" if failure and index % 2 == 0 else "0",
                    "1" if failure and index % 2 == 1 else "0",
                    "0",
                    "0",
                    "0",
                ]
            )
        )
    return "\n".join(rows)


def upload_dataset(client: TestClient, token: str) -> dict[str, object]:
    response = client.post(
        "/api/datasets",
        files={"file": ("uploaded.csv", sample_dataset_csv(), "text/csv")},
        headers=bearer(token),
    )
    assert response.status_code == 201
    return response.json()


def test_algorithms_endpoint_lists_registry(client: TestClient) -> None:
    token = register_user(client, username="algorithm_user")

    unauthorized = client.get("/api/algorithms")
    assert unauthorized.status_code == 401

    response = client.get("/api/algorithms", headers=bearer(token))
    assert response.status_code == 200
    names = {item["name"] for item in response.json()}
    assert names == {
        "fedavg",
        "failure_aware_v1",
        "failure_aware_v2",
        "dynamic_failure_aware",
    }


def test_create_and_retrieve_experiment(client: TestClient) -> None:
    token = register_user(client, username="owner_user")
    created = create_experiment(client, token, distribution="highly_non_iid")

    assert created["status"] == "PENDING"
    assert created["user_id"] > 0
    assert created["algorithm"] == "fedavg"
    assert created["distribution"] == "highly_non_iid"

    listing = client.get("/api/experiments", headers=bearer(token))
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()] == [created["id"]]

    detail = client.get(f"/api/experiments/{created['id']}", headers=bearer(token))
    assert detail.status_code == 200
    assert detail.json()["id"] == created["id"]


def test_upload_dataset_and_create_experiment_with_dataset(client: TestClient) -> None:
    token = register_user(client, username="dataset_owner")
    dataset = upload_dataset(client, token)

    assert dataset["status"] == "READY"
    assert dataset["rows"] == 30
    assert dataset["columns"] > 10

    listing = client.get("/api/datasets", headers=bearer(token))
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()] == [dataset["id"]]

    created = create_experiment(client, token, dataset_id=dataset["id"])
    assert created["dataset_id"] == dataset["id"]


def test_upload_dataset_rejects_invalid_csv(client: TestClient) -> None:
    token = register_user(client, username="bad_dataset_owner")

    response = client.post(
        "/api/datasets",
        files={"file": ("bad.csv", "UDI,Machine failure\n1,0\n", "text/csv")},
        headers=bearer(token),
    )

    assert response.status_code == 400
    assert "Dataset must contain AI4I columns" in response.text


def test_create_rejects_invalid_algorithm_and_distribution(client: TestClient) -> None:
    token = register_user(client, username="validation_user")

    bad_algorithm = client.post(
        "/api/experiments",
        json=experiment_payload(algorithm="missing"),
        headers=bearer(token),
    )
    assert bad_algorithm.status_code == 422
    assert "Unknown algorithm" in bad_algorithm.text

    bad_distribution = client.post(
        "/api/experiments",
        json=experiment_payload(distribution="bad_split"),
        headers=bearer(token),
    )
    assert bad_distribution.status_code == 422
    assert "Unknown distribution" in bad_distribution.text

    bad_rounds = client.post(
        "/api/experiments",
        json=experiment_payload(rounds=0),
        headers=bearer(token),
    )
    assert bad_rounds.status_code == 422


def test_experiment_ownership_and_admin_visibility(client: TestClient) -> None:
    owner_token = register_user(client, username="owner")
    other_token = register_user(client, username="other", factory_id=2)
    admin_token = register_user(client, username="admin", role="admin", factory_id=None)
    created = create_experiment(client, owner_token)

    other_detail = client.get(f"/api/experiments/{created['id']}", headers=bearer(other_token))
    assert other_detail.status_code == 404

    other_listing = client.get("/api/experiments", headers=bearer(other_token))
    assert other_listing.status_code == 200
    assert other_listing.json() == []

    admin_detail = client.get(f"/api/experiments/{created['id']}", headers=bearer(admin_token))
    assert admin_detail.status_code == 200
    assert admin_detail.json()["id"] == created["id"]


def test_dataset_ownership(client: TestClient) -> None:
    owner_token = register_user(client, username="dataset_visible_owner")
    other_token = register_user(client, username="dataset_visible_other", factory_id=2)
    admin_token = register_user(client, username="dataset_visible_admin", role="admin", factory_id=None)
    dataset = upload_dataset(client, owner_token)

    other_detail = client.get(f"/api/datasets/{dataset['id']}", headers=bearer(other_token))
    assert other_detail.status_code == 404

    other_create = client.post(
        "/api/experiments",
        json=experiment_payload(dataset_id=dataset["id"]),
        headers=bearer(other_token),
    )
    assert other_create.status_code == 400
    assert "Dataset not found" in other_create.text

    admin_detail = client.get(f"/api/datasets/{dataset['id']}", headers=bearer(admin_token))
    assert admin_detail.status_code == 200


def test_successful_experiment_run_persists_results(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_user(client, username="runner")
    created = create_experiment(client, token, rounds=2)

    def fake_run_experiment(
        algorithm: str,
        distribution: str,
        config: dict[str, object],
    ) -> dict[str, object]:
        assert algorithm == "fedavg"
        assert distribution == "iid"
        assert config["rounds"] == 2
        return {
            "algorithm": algorithm,
            "distribution": distribution,
            "accuracy": 0.91,
            "precision": 0.82,
            "recall": 0.73,
            "f1": 0.77,
            "communication_cost": 10,
            "training_time": 1.25,
            "rounds": 2,
            "convergence_history": [{"round": 1, "val_f1": 0.7}],
        }

    monkeypatch.setattr(
        "iot_fl.backend.services.experiment_service.run_experiment",
        fake_run_experiment,
    )

    run_response = client.post(f"/api/experiments/{created['id']}/run", headers=bearer(token))
    assert run_response.status_code == 200
    body = run_response.json()
    assert body["status"] == "COMPLETED"
    assert body["accuracy"] == 0.91
    assert body["f1_score"] == 0.77
    assert body["finished_at"] is not None

    results = client.get(f"/api/experiments/{created['id']}/results", headers=bearer(token))
    assert results.status_code == 200
    assert results.json()["convergence_history"] == [{"round": 1, "val_f1": 0.7}]


def test_dataset_backed_experiment_passes_dataset_config(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_user(client, username="dataset_runner")
    dataset = upload_dataset(client, token)
    created = create_experiment(client, token, dataset_id=dataset["id"])

    def fake_run_experiment(
        algorithm: str,
        distribution: str,
        config: dict[str, object],
    ) -> dict[str, object]:
        del algorithm, distribution
        assert str(config["data_path"]).endswith("processed.csv")
        assert str(config["factory_root"]).endswith("factories")
        return {
            "algorithm": "fedavg",
            "distribution": "iid",
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "communication_cost": 5,
            "training_time": 0.5,
            "rounds": 5,
            "convergence_history": [],
        }

    monkeypatch.setattr(
        "iot_fl.backend.services.experiment_service.run_experiment",
        fake_run_experiment,
    )

    run_response = client.post(f"/api/experiments/{created['id']}/run", headers=bearer(token))
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "COMPLETED"


def test_failed_experiment_run_is_recorded(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_user(client, username="failure_runner")
    created = create_experiment(client, token)

    def fake_run_experiment(
        algorithm: str,
        distribution: str,
        config: dict[str, object],
    ) -> dict[str, object]:
        del algorithm, distribution, config
        raise RuntimeError("training failed")

    monkeypatch.setattr(
        "iot_fl.backend.services.experiment_service.run_experiment",
        fake_run_experiment,
    )

    run_response = client.post(f"/api/experiments/{created['id']}/run", headers=bearer(token))
    assert run_response.status_code == 200
    body = run_response.json()
    assert body["status"] == "FAILED"
    assert body["error_message"] == "training failed"
    assert body["accuracy"] is None
