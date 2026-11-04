"""Logging configuration for the Female Wellness & Lifestyle MCP Server.

CRITICAL ARCHITECTURAL REQUIREMENT:
When operating as an MCP STDIO server, stdout is reserved EXCLUSIVELY for
JSON-RPC protocol frames. Any write to stdout corrupts the transport and
causes MCP clients (such as Claude Desktop) to disconnect.

All logging is therefore strictly routed to sys.stderr with sensitive
credential masking.
"""

import logging
import re
import sys
from typing import ClassVar

# Regex pattern matching Anthropic API keys (e.g., sk-ant-api03-...)
_API_KEY_PATTERN = re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}")


class SensitiveDataFilter(logging.Filter):
    """Logging filter that scrubs API keys and sensitive tokens from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _API_KEY_PATTERN.sub("sk-ant-***REDACTED***", record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (_API_KEY_PATTERN.sub("sk-ant-***REDACTED***", v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    _API_KEY_PATTERN.sub("sk-ant-***REDACTED***", a) if isinstance(a, str) else a
                    for a in record.args
                )
        return True


class StderrFormatter(logging.Formatter):
    """Clean, structured formatter for STDIO-safe server logging."""

    _FORMAT: ClassVar[str] = (
        "[%(asctime)s] [%(levelname)-7s] [%(name)s:%(lineno)d]: %(message)s"
    )
    _DATEFMT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FORMAT, datefmt=self._DATEFMT)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure the root logger for the application.

    Ensures logs are written ONLY to sys.stderr, never sys.stdout.
    Applies sensitive data redaction to all handlers.

    Args:
        log_level: Desired log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        The configured root logger for the female_wellness package.
    """
    normalized_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger("female_wellness")
    root_logger.setLevel(normalized_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in root_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(normalized_level)
        handler.setFormatter(StderrFormatter())
        handler.addFilter(SensitiveDataFilter())
        root_logger.addHandler(handler)

    # Prevent propagation to third-party loggers that might write to stdout
    root_logger.propagate = False

    # Also silence noisy third-party loggers (e.g. Chroma, Transformers, HTTPX)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child logger under the female_wellness namespace.

    Args:
        name: Name of the child module (e.g. 'rag.retriever').

    Returns:
        A logging.Logger instance inheriting stderr output and filters.
    """
    if name.startswith("female_wellness"):
        return logging.getLogger(name)
    return logging.getLogger(f"female_wellness.{name}")

