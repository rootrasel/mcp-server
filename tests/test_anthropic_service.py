"""Unit tests for AnthropicService with mocked Claude API interactions."""

from unittest.mock import MagicMock

import anthropic
import pytest

from female_wellness.config import Settings
from female_wellness.core.exceptions import (
    AnthropicAPIKeyMissingError,
    AnthropicServiceError,
)
from female_wellness.schemas.documents import ChunkMetadata, RetrievedChunk
from female_wellness.schemas.wellness import RetrievedContext, SourceCitation
from female_wellness.services.anthropic_service import AnthropicService


@pytest.fixture
def sample_context() -> RetrievedContext:
    """Fixture providing populated evidence context."""
    meta = ChunkMetadata(
        document_id="doc_skin",
        source="skin_barrier_guide.md",
        life_stage="beauty_radiance",
        category="skin_health",
        chunk_index=0,
    )
    chunk = RetrievedChunk(
        chunk_id="doc_skin_c0",
        text="Essential fatty acids and ceramides support epidermal barrier hydration.",
        metadata=meta,
        distance=0.15,
        similarity_score=0.85,
    )
    citation = SourceCitation(
        source="skin_barrier_guide.md",
        page_number=1,
        life_stage="beauty_radiance",
        category="skin_health",
        chunk_id="doc_skin_c0",
        similarity_score=0.85,
        excerpt="Essential fatty acids and ceramides...",
    )
    return RetrievedContext(
        query="How to support skin barrier hydration?",
        life_stage="beauty_radiance",
        chunks=[chunk],
        citations=[citation],
        formatted_context="### Evidence: Ceramides support hydration.",
        has_sufficient_evidence=True,
    )


def test_missing_api_key_raises_error():
    """Verify AnthropicAPIKeyMissingError is raised if no API key is configured."""
    settings = Settings(
        _env_file=None,
        ANTHROPIC_API_KEY=None,
    )
    service = AnthropicService(settings=settings)
    with pytest.raises(AnthropicAPIKeyMissingError) as exc_info:
        _ = service.client
    assert "Anthropic API key is required" in str(exc_info.value)


def test_generate_answer_insufficient_evidence():
    """Verify service returns a safe disclaimer without calling Claude if evidence is missing."""
    mock_client = MagicMock()
    service = AnthropicService(client=mock_client)

    empty_context = RetrievedContext(
        query="Unresearched topic",
        life_stage="cycle_fertility",
        chunks=[],
        citations=[],
        formatted_context="",
        has_sufficient_evidence=False,
    )

    response = service.generate_answer(
        question="Unresearched topic?",
        context=empty_context,
        life_stage="cycle_fertility",
    )

    # API should NOT have been invoked
    mock_client.messages.create.assert_not_called()
    assert response.evidence_grounded is False
    assert "does not currently contain sufficient verified information" in response.answer


def test_generate_answer_grounded_success(sample_context):
    """Verify successful evidence synthesis with mock Claude response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Based on skin_barrier_guide.md, ceramides restore the lipid barrier."
    mock_response.content = [mock_content]
    mock_client.messages.create.return_value = mock_response

    service = AnthropicService(client=mock_client)
    response = service.generate_answer(
        question="How do ceramides work?",
        context=sample_context,
        life_stage="beauty_radiance",
    )

    mock_client.messages.create.assert_called_once()
    assert response.evidence_grounded is True
    assert "Based on skin_barrier_guide.md" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].source == "skin_barrier_guide.md"


def test_rate_limit_error_handling(sample_context):
    """Verify RateLimitError is caught and wrapped into AnthropicServiceError."""
    mock_client = MagicMock()
    rate_err = anthropic.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={},
    )
    mock_client.messages.create.side_effect = rate_err

    service = AnthropicService(client=mock_client)
    with pytest.raises(AnthropicServiceError) as exc_info:
        service.generate_answer(
            question="Rate test",
            context=sample_context,
            life_stage="beauty_radiance",
        )
    assert "rate limit exceeded" in str(exc_info.value).lower()


def test_timeout_error_handling(sample_context):
    """Verify APITimeoutError is caught and converted to AnthropicServiceError."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = anthropic.APITimeoutError(request=MagicMock())

    service = AnthropicService(client=mock_client)
    with pytest.raises(AnthropicServiceError) as exc_info:
        service.generate_answer(
            question="Timeout test",
            context=sample_context,
            life_stage="beauty_radiance",
        )
    assert "timed out" in str(exc_info.value).lower()


def test_summarize_context_success(sample_context):
    """Verify summarize_context calls messages API and returns summary."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Summary: Ceramides support epidermal hydration."
    mock_response.content = [mock_content]
    mock_client.messages.create.return_value = mock_response

    service = AnthropicService(client=mock_client)
    summary = service.summarize_context(sample_context)

    assert "Ceramides support epidermal hydration" in summary


def test_generate_recommendations_success(sample_context):
    """Verify generate_recommendations builds structured RecommendationResponse."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "1. Maintain adequate hydration with electrolytes.\n2. Supplement omega-3 fatty acids."
    mock_response.content = [mock_content]
    mock_client.messages.create.return_value = mock_response

    service = AnthropicService(client=mock_client)
    rec_response = service.generate_recommendations(
        life_stage="beauty_radiance",
        category="skin_health",
        context=sample_context,
    )

    assert len(rec_response.recommendations) == 1
    assert "hydration" in rec_response.recommendations[0].description
    assert len(rec_response.citations) == 1
