from __future__ import annotations

from sqlalchemy import select

from sqlalchemy.orm import Session

from iot_fl.backend.database import SessionLocal, init_db
from iot_fl.backend.models import Factory


DEFAULT_FACTORIES = [
    ("factory_01", "Simulated factory client 1"),
    ("factory_02", "Simulated factory client 2"),
    ("factory_03", "Simulated factory client 3"),
    ("factory_04", "Simulated factory client 4"),
    ("factory_05", "Simulated factory client 5"),
]


def ensure_default_factories(db: Session) -> None:
    for name, description in DEFAULT_FACTORIES:
        exists = db.scalar(select(Factory).where(Factory.name == name))
        if exists is None:
            db.add(Factory(name=name, description=description))
    db.commit()


def seed_factories() -> None:
    init_db()
    with SessionLocal() as db:
        ensure_default_factories(db)


if __name__ == "__main__":
    seed_factories()
    print("Seeded default factories")
