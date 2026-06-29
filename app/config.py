"""Application configuration.

All secrets and tunables are loaded from environment variables (or an `.env`
file) via Pydantic ``BaseSettings`` so that nothing sensitive is ever hard
coded into the codebase.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -----------------------------------------------------
    app_name: str = Field(default="Moldova AVM")
    app_env: str = Field(default="development")
    api_prefix: str = Field(default="/api")

    # --- CORS ------------------------------------------------------------
    # Comma separated list, or "*" for all origins.
    allowed_origins: str = Field(default="*")

    # --- Supabase --------------------------------------------------------
    supabase_url: str = Field(default="")
    supabase_key: str = Field(default="")
    supabase_listings_table: str = Field(default="listings")

    # --- Scraper ---------------------------------------------------------
    scraper_user_agent: str = Field(
        default="MoldovaAVM/1.0 (+https://example.com)"
    )
    scraper_request_timeout: int = Field(default=20)

    @property
    def cors_origins(self) -> list[str]:
        """Parse ``allowed_origins`` into a list usable by CORSMiddleware."""
        raw = (self.allowed_origins or "").strip()
        if raw == "" or raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def supabase_configured(self) -> bool:
        """True when both URL and key are present."""
        return bool(self.supabase_url) and bool(self.supabase_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance."""
    return Settings()
