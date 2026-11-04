"""Domain exception hierarchy for the Female Wellness & Lifestyle MCP Server.

Provides specific, categorized exceptions to enable robust error handling,
clear client-facing error messages, and controlled MCP protocol responses.
"""


class FemaleWellnessError(Exception):
    """Base exception for all domain errors within the platform."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        """Serialize error for MCP client responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(FemaleWellnessError):
    """Raised when environment variables or configuration values are missing or invalid."""


class LifeStageNotFoundError(FemaleWellnessError):
    """Raised when an unsupported or invalid life-stage identifier is supplied."""


class InvalidCategoryError(FemaleWellnessError):
    """Raised when a requested category is not recognized for a given life stage."""


class VectorStoreError(FemaleWellnessError):
    """Raised when vector database operations fail."""


class DocumentIngestionError(FemaleWellnessError):
    """Raised when parsing, extracting, or chunking a document fails."""


class RetrievalError(FemaleWellnessError):
    """Raised when retrieval queries or similarity searches encounter an error."""


class SafetyViolationError(FemaleWellnessError):
    """Raised when an operation breaches female health safety or diagnostic rules."""


class RedFlagEscalationError(FemaleWellnessError):
    """Raised when urgent medical symptoms are detected that require immediate clinical care."""


class AnthropicServiceError(FemaleWellnessError):
    """Raised when Anthropic Claude API interactions fail (timeouts, rate limits, API errors)."""


class AnthropicAPIKeyMissingError(ConfigurationError):
    """Raised when an operation requires the Claude API but no key is configured."""

