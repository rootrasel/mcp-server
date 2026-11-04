"""Anthropic Claude integration service for server-side generation and reasoning."""

from typing import Any

import anthropic

from female_wellness.config import Settings, get_settings
from female_wellness.core.exceptions import (
    AnthropicAPIKeyMissingError,
    AnthropicServiceError,
)
from female_wellness.core.logging_config import get_logger
from female_wellness.mcp.prompts.wellness_prompts import (
    SYSTEM_WELLNESS_PROMPT,
    build_recommendations_prompt,
    build_summarize_prompt,
    build_wellness_answer_prompt,
)
from female_wellness.schemas.recommendations import (
    AnswerResponse,
    RecommendationItem,
    RecommendationResponse,
)
from female_wellness.schemas.wellness import RetrievedContext

logger = get_logger("services.anthropic")


class AnthropicService:
    """Manages LLM-powered reasoning and evidence synthesis using the Anthropic Claude API."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: anthropic.Anthropic | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the Anthropic service.

        Args:
            settings: Application settings containing model and API credentials.
            client: Pre-configured Anthropic client instance (optional, useful for testing).
            timeout_seconds: HTTP request timeout in seconds.
        """
        self.settings = settings or get_settings()
        self.model = self.settings.ANTHROPIC_MODEL
        self.timeout_seconds = timeout_seconds
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy initializer for the Anthropic client."""
        if self._client is None:
            if not self.settings.has_anthropic_key:
                raise AnthropicAPIKeyMissingError(
                    "Anthropic API key is required for server-side Claude generation. "
                    "Please set ANTHROPIC_API_KEY in your .env file or environment."
                )
            try:
                self._client = anthropic.Anthropic(
                    api_key=self.settings.get_anthropic_api_key_str(),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                raise AnthropicServiceError(
                    f"Failed to initialize Anthropic client: {exc}"
                ) from exc
        return self._client

    def _format_user_context(self, user_context: dict[str, Any] | None) -> str:
        """Safely format consented user profile parameters into text."""
        if not user_context:
            return ""
        lines: list[str] = []
        for k, v in user_context.items():
            clean_k = k.replace("_", " ").title()
            lines.append(f"- {clean_k}: {v}")
        return "\n".join(lines)

    def generate_answer(
        self,
        question: str,
        context: RetrievedContext,
        life_stage: str,
        user_context: dict[str, Any] | None = None,
    ) -> AnswerResponse:
        """Generate an evidence-grounded answer to a female wellness inquiry.

        Args:
            question: User question string.
            context: Retrieved evidence context from the RAG pipeline.
            life_stage: Selected life-stage journey.
            user_context: Optional consented user profile parameters.

        Returns:
            AnswerResponse with synthesized answer, citations, and evidence grounding flag.
        """
        # Grounding Rule: If no relevant evidence was found, do NOT hallucinate medical claims
        if not context.has_sufficient_evidence:
            logger.info("Insufficient evidence retrieved for query='%s'. Returning safe notice.", question)
            return AnswerResponse(
                question=question,
                answer=(
                    "The knowledge base does not currently contain sufficient verified information on this topic "
                    f"within the '{life_stage}' journey to provide grounded wellness guidance. "
                    "For personalized health questions or specific clinical symptoms, please consult a licensed "
                    "healthcare professional."
                ),
                life_stage=life_stage,
                citations=[],
                evidence_grounded=False,
            )

        user_context_str = self._format_user_context(user_context)
        prompt_text = build_wellness_answer_prompt(
            question=question,
            life_stage=life_stage,
            formatted_evidence=context.formatted_context,
            user_context_str=user_context_str,
        )

        try:
            logger.debug("Dispatching request to Claude model '%s'...", self.model)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.2,  # Low temperature for strict factual consistency
                system=SYSTEM_WELLNESS_PROMPT,
                messages=[{"role": "user", "content": prompt_text}],
            )
            raw_text = response.content[0].text if response.content else ""
            return AnswerResponse(
                question=question,
                answer=raw_text.strip(),
                life_stage=life_stage,
                citations=context.citations,
                evidence_grounded=True,
            )
        except anthropic.RateLimitError as exc:
            raise AnthropicServiceError(
                "Anthropic Claude API rate limit exceeded. Please wait a moment before retrying.",
                details={"model": self.model, "error": str(exc)},
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise AnthropicServiceError(
                f"Anthropic Claude API request timed out after {self.timeout_seconds} seconds.",
                details={"model": self.model, "timeout": self.timeout_seconds},
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AnthropicServiceError(
                f"Failed to connect to Anthropic Claude API: {exc}",
                details={"model": self.model},
            ) from exc
        except anthropic.APIStatusError as exc:
            raise AnthropicServiceError(
                f"Anthropic Claude API returned error status {exc.status_code}: {exc.message}",
                details={"status_code": exc.status_code, "error": str(exc)},
            ) from exc
        except Exception as exc:
            raise AnthropicServiceError(
                f"Unexpected error communicating with Claude API: {exc}",
                details={"model": self.model, "error": str(exc)},
            ) from exc

    def summarize_context(self, context: RetrievedContext, max_tokens: int = 400) -> str:
        """Generate an executive summary of retrieved wellness evidence.

        Args:
            context: Retrieved evidence context.
            max_tokens: Approximate target summary length.

        Returns:
            Summarized string.
        """
        if not context.has_sufficient_evidence:
            return "No verified evidence available to summarize."

        prompt_text = build_summarize_prompt(
            formatted_evidence=context.formatted_context,
            max_tokens=max_tokens,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens + 100,
                temperature=0.2,
                system=SYSTEM_WELLNESS_PROMPT,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return response.content[0].text.strip() if response.content else ""
        except Exception as exc:
            raise AnthropicServiceError(f"Summarization failed: {exc}") from exc

    def generate_recommendations(
        self,
        life_stage: str,
        category: str | None,
        context: RetrievedContext,
        user_context: dict[str, Any] | None = None,
    ) -> RecommendationResponse:
        """Generate structured lifestyle and wellness recommendations grounded in evidence.

        Args:
            life_stage: Target wellness journey.
            category: Domain category or None.
            context: Retrieved evidence context.
            user_context: Optional consented profile context.

        Returns:
            RecommendationResponse with recommendation items and citations.
        """
        if not context.has_sufficient_evidence:
            return RecommendationResponse(
                life_stage=life_stage,
                category=category,
                recommendations=[],
                citations=[],
            )

        user_context_str = self._format_user_context(user_context)
        prompt_text = build_recommendations_prompt(
            life_stage=life_stage,
            category=category,
            formatted_evidence=context.formatted_context,
            user_context_str=user_context_str,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.2,
                system=SYSTEM_WELLNESS_PROMPT,
                messages=[{"role": "user", "content": prompt_text}],
            )
            raw_text = response.content[0].text.strip() if response.content else ""

            # Extract distinct recommendation items or format top item
            sources = [c.source for c in context.citations]
            recommendation_items = [
                RecommendationItem(
                    title=f"Evidence-Based Guidance for {life_stage.replace('_', ' ').title()}",
                    description=raw_text,
                    category=category or "general_wellness",
                    life_stage=life_stage,
                    evidence_sources=list(set(sources)),
                )
            ]

            return RecommendationResponse(
                life_stage=life_stage,
                category=category,
                recommendations=recommendation_items,
                citations=context.citations,
            )
        except Exception as exc:
            raise AnthropicServiceError(
                f"Recommendation generation failed: {exc}"
            ) from exc
