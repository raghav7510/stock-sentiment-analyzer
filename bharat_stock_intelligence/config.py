"""Centralised application configuration."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Bharat Stock Intelligence"
    environment: str = "development"
    news_api_key: str | None = None
    finbert_model: str = "ProsusAI/finbert"
    database_url: str = "postgresql+psycopg://bsi:bsi@localhost:5432/bsi"
    request_timeout_seconds: float = 12.0
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
