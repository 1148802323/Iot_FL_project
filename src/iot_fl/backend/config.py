from __future__ import annotations

import os
import secrets
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Settings:
    def __init__(self) -> None:
        load_dotenv()
        self.jwt_secret = os.getenv("JWT_SECRET", "")
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_urlsafe(32)

        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120")
        )
        self.database_url = os.getenv(
            "DATABASE_URL",
            f"sqlite:///{PROJECT_ROOT / 'iot_fl_app.db'}",
        )
        self.cors_origins = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]


settings = Settings()

