"""Centralized configuration management for the Female Wellness & Lifestyle MCP Server.

Uses Pydantic BaseSettings to read and validate environment variables from .env
or the host execution environment.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from female_wellness.core.constants import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_CHROMA_PERSIST_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)
from female_wellness.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Application settings with strict validation and safe defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Anthropic Claude Configuration
    ANTHROPIC_API_KEY: SecretStr | None = Field(
        default=None,
        description="Optional API key for server-side Claude generation. Not required if MCP host handles LLM reasoning.",
    )
    ANTHROPIC_MODEL: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL,
        description="Anthropic Claude model identifier to use when server-side generation is requested.",
    )

    # ChromaDB Vector Store Configuration
    CHROMA_PERSIST_DIRECTORY: Path = Field(
        default=Path(DEFAULT_CHROMA_PERSIST_DIR),
        description="Directory path where ChromaDB vector data and sqlite files are persisted.",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default=DEFAULT_COLLECTION_NAME,
        description="Collection name for female wellness knowledge embeddings.",
    )

    # Embeddings Configuration
    EMBEDDING_MODEL: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Hugging Face model identifier or local path for Sentence Transformers embedding generation.",
    )

    # RAG Chunking Configuration
    CHUNK_SIZE: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        gt=50,
        le=4000,
        description="Maximum character or token length of each document chunk.",
    )
    CHUNK_OVERLAP: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        ge=0,
        le=1000,
        description="Character or token overlap between consecutive chunks to maintain semantic continuity.",
    )

    # RAG Retrieval Configuration
    TOP_K: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=50,
        description="Default number of top relevant chunks to retrieve.",
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Cosine similarity cutoff threshold (1 - cosine_distance). Chunks below this score are filtered.",
    )

    # Logging Configuration
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application logging verbosity.",
    )

    @field_validator("CHROMA_PERSIST_DIRECTORY", mode="after")
    @classmethod
    def validate_persist_directory(cls, v: Path) -> Path:
        """Resolve path and ensure directory parent exists or can be created."""
        try:
            resolved = v.resolve()
            return resolved
        except Exception as exc:
            raise ConfigurationError(
                f"Invalid CHROMA_PERSIST_DIRECTORY path '{v}': {exc}"
            ) from exc

    @model_validator(mode="after")
    def validate_chunk_parameters(self) -> "Settings":
        """Ensure chunk overlap is strictly smaller than chunk size."""
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ConfigurationError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be strictly less than CHUNK_SIZE ({self.CHUNK_SIZE})."
            )
        return self

    @property
    def has_anthropic_key(self) -> bool:
        """Check whether an Anthropic API key is configured."""
        return (
            self.ANTHROPIC_API_KEY is not None
            and bool(self.ANTHROPIC_API_KEY.get_secret_value().strip())
        )

    def get_anthropic_api_key_str(self) -> str:
        """Safely retrieve the raw Anthropic API key string or raise an actionable error."""
        if not self.has_anthropic_key or self.ANTHROPIC_API_KEY is None:
            raise ConfigurationError(
                "Anthropic API key is not configured. Please set ANTHROPIC_API_KEY in your .env file "
                "or environment variables to enable server-side Claude generation."
            )
        return self.ANTHROPIC_API_KEY.get_secret_value()


# Cached singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Retrieve or initialize the cached application Settings singleton.

    Returns:
        Settings: Validated application configuration.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset cached settings singleton (primarily used for unit testing)."""
    global _settings
    _settings = None

