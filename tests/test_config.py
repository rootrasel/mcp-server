"""Unit tests for configuration management, logging, constants, and exceptions."""

import logging

import pytest
from pydantic import ValidationError

from female_wellness.config import Settings, reset_settings
from female_wellness.core.constants import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    PRIMARY_LIFE_STAGES,
    LifeStageId,
)
from female_wellness.core.exceptions import (
    ConfigurationError,
    FemaleWellnessError,
    LifeStageNotFoundError,
)
from female_wellness.core.logging_config import SensitiveDataFilter, setup_logging


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure clean environment state for each test."""
    reset_settings()
    yield
    reset_settings()


def test_default_settings():
    """Verify that default settings instantiate with expected values."""
    settings = Settings(_env_file=None)  # Ignore any local .env file

    assert settings.ANTHROPIC_API_KEY is None
    assert settings.ANTHROPIC_MODEL == DEFAULT_ANTHROPIC_MODEL
    assert str(settings.CHROMA_PERSIST_DIRECTORY).endswith("chroma_db")
    assert settings.CHROMA_COLLECTION_NAME == DEFAULT_COLLECTION_NAME
    assert settings.EMBEDDING_MODEL == DEFAULT_EMBEDDING_MODEL
    assert settings.CHUNK_SIZE == DEFAULT_CHUNK_SIZE
    assert settings.CHUNK_OVERLAP == DEFAULT_CHUNK_OVERLAP
    assert settings.TOP_K == DEFAULT_TOP_K
    assert settings.SIMILARITY_THRESHOLD == DEFAULT_SIMILARITY_THRESHOLD
    assert settings.LOG_LEVEL == "INFO"
    assert settings.has_anthropic_key is False


def test_custom_settings_override(monkeypatch):
    """Verify that environment variables cleanly override defaults."""
    monkeypatch.setenv("TOP_K", "12")
    monkeypatch.setenv("CHUNK_SIZE", "600")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.45")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-testkey1234567890abcdef")

    settings = Settings(_env_file=None)
    assert settings.TOP_K == 12
    assert settings.CHUNK_SIZE == 600
    assert settings.CHUNK_OVERLAP == 100
    assert settings.SIMILARITY_THRESHOLD == 0.45
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.has_anthropic_key is True
    assert settings.get_anthropic_api_key_str() == "sk-ant-api03-testkey1234567890abcdef"


def test_invalid_chunk_overlap():
    """Verify validation fails when overlap is greater than or equal to chunk size."""
    with pytest.raises((ValidationError, ConfigurationError)):
        Settings(_env_file=None, CHUNK_SIZE=200, CHUNK_OVERLAP=250)


def test_api_key_redaction():
    """Verify API keys are masked in string/repr output of Settings."""
    settings = Settings(
        _env_file=None,
        ANTHROPIC_API_KEY="sk-ant-api03-testkey1234567890abcdef",
    )
    repr_str = repr(settings)
    assert "sk-ant-api03" not in repr_str
    assert "**********" in repr_str or "SecretStr" in repr_str


def test_missing_api_key_raises_actionable_error():
    """Verify informative error when requesting API key while not set."""
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigurationError) as exc_info:
        settings.get_anthropic_api_key_str()
    assert "Please set ANTHROPIC_API_KEY" in str(exc_info.value)


def test_logging_routes_to_stderr(monkeypatch):
    """Verify setup_logging attaches stream handler strictly to stderr."""
    logger = setup_logging(log_level="DEBUG")
    handlers = logger.handlers
    assert len(handlers) > 0
    # Every stream handler must point to sys.stderr
    import sys

    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            assert h.stream == sys.stderr


def test_sensitive_data_logging_filter():
    """Verify that the logging filter scrubs Anthropic API keys."""
    filtr = SensitiveDataFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Connecting using key sk-ant-api03-abcdef123456789012345678 to Claude",
        args=(),
        exc_info=None,
    )
    filtr.filter(record)
    assert "sk-ant-api03-abcdef" not in record.msg
    assert "sk-ant-***REDACTED***" in record.msg


def test_exceptions_hierarchy():
    """Verify custom exception details and dictionary conversion."""
    err = LifeStageNotFoundError(
        "Invalid life stage",
        details={"provided": "invalid_stage", "allowed": ["beauty_radiance"]},
    )
    assert isinstance(err, FemaleWellnessError)
    err_dict = err.to_dict()
    assert err_dict["error_type"] == "LifeStageNotFoundError"
    assert err_dict["message"] == "Invalid life stage"
    assert err_dict["details"]["provided"] == "invalid_stage"


def test_primary_life_stages_count():
    """Verify that exactly six primary journeys are defined."""
    assert len(PRIMARY_LIFE_STAGES) == 6
    assert LifeStageId.BEAUTY_RADIANCE in PRIMARY_LIFE_STAGES
    assert LifeStageId.CYCLE_FERTILITY in PRIMARY_LIFE_STAGES
    assert LifeStageId.ATHLETE_PERFORMANCE in PRIMARY_LIFE_STAGES
    assert LifeStageId.PREGNANCY_POSTPARTUM in PRIMARY_LIFE_STAGES
    assert LifeStageId.PERIMENOPAUSE in PRIMARY_LIFE_STAGES
    assert LifeStageId.LIFELONG_THRIVING in PRIMARY_LIFE_STAGES
