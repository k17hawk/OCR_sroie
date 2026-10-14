from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEVALABEL_", env_file=".env", extra="ignore"
    )

    db_url: str = "sqlite:///./devalabel.db"
    storage_backend: str = "local"  # "local" | "s3"
    storage_root: Path = Path("./data")
    s3_endpoint: str | None = None
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"
    jwt_secret: str = Field(default="change-me-in-prod")
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 8 * 60
    max_upload_mb: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
