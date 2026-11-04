"""MCP Tools for asking wellness questions and retrieving grounded recommendations."""

from typing import Any

from female_wellness.config import get_settings
from female_wellness.core.exceptions import (
    AnthropicAPIKeyMissingError,
    FemaleWellnessError,
)
from female_wellness.core.logging_config import get_logger
from female_wellness.rag.pipeline import RAGPipeline
from female_wellness.schemas.recommendations import (
    AnswerResponse,
    RecommendationItem,
    RecommendationResponse,
)
from female_wellness.services.anthropic_service import AnthropicService
from female_wellness.services.life_stage_service import LifeStageService
from female_wellness.services.safety_service import SafetyService

logger = get_logger("mcp.tools.recommendations")


def register_recommendation_tools(
    mcp: Any,
    pipeline: RAGPipeline | None = None,
    anthropic_service: AnthropicService | None = None,
    life_stage_service: LifeStageService | None = None,
    safety_service: SafetyService | None = None,
) -> None:
    """Register question-answering and recommendation tools on the MCP server.

    Args:
        mcp: FastMCP or MCPServer instance.
        pipeline: Injected RAGPipeline.
        anthropic_service: Injected AnthropicService.
        life_stage_service: Injected LifeStageService.
        safety_service: Injected SafetyService.
    """
    rag = pipeline or RAGPipeline()
    life_stages = life_stage_service or LifeStageService()
    claude = anthropic_service or AnthropicService()
    safety = safety_service or SafetyService()
    settings = get_settings()

    @mcp.tool(
        name="ask_wellness_question",
        description=(
            "Retrieve verified female wellness evidence and generate an evidence-grounded, "
            "contextual answer for a specific life-stage journey. Returns answer, citations, "
            "and health disclaimer."
        ),
    )
    def ask_wellness_question(
        question: str,
        life_stage: str,
        user_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer a female health inquiry grounded in verified knowledge."""
        logger.info("Tool 'ask_wellness_question' invoked for stage='%s'", life_stage)
        try:
            # 1. Validate inputs
            cleaned_question = question.strip()
            if not cleaned_question:
                return {"error": "InvalidInput", "message": "Question cannot be empty."}

            canonical_stage = life_stages.validate_life_stage(life_stage)

            # 2. Medical Safety & Red-Flag Escalation Check
            safety_eval = safety.evaluate_query(cleaned_question, life_stage=canonical_stage)
            if not safety_eval.is_safe:
                logger.warning(
                    "Safety trigger intercepted query. Condition: %s", safety_eval.flagged_condition
                )
                escalation_resp = AnswerResponse(
                    question=cleaned_question,
                    answer=safety_eval.escalation_message or "Medical safety limit triggered.",
                    life_stage=canonical_stage,
                    citations=[],
                    evidence_grounded=False,
                    disclaimer=(
                        "URGENT CLINICAL NOTICE: Please seek prompt medical evaluation from a licensed healthcare provider."
                        if safety_eval.is_emergency
                        else "Notice: Wellness education cannot prescribe or diagnose medical conditions."
                    ),
                )
                return escalation_resp.model_dump()

            # 2. Retrieve relevant context
            context = rag.retrieve_context(
                query=cleaned_question,
                life_stage=canonical_stage,
                top_k=settings.TOP_K,
                min_similarity=settings.SIMILARITY_THRESHOLD,
            )

            # 3. If no sufficient evidence found
            if not context.has_sufficient_evidence:
                no_evidence_resp = AnswerResponse(
                    question=cleaned_question,
                    answer=(
                        f"The knowledge base does not currently contain sufficient verified evidence "
                        f"in the '{canonical_stage}' journey to answer this inquiry responsibly. "
                        "We prioritize clinical safety and do not fabricate health information. "
                        "Please consult a licensed healthcare professional for medical advice."
                    ),
                    life_stage=canonical_stage,
                    citations=[],
                    evidence_grounded=False,
                )
                return no_evidence_resp.model_dump()

            # 4. Generate answer using server-side Claude if configured
            if settings.has_anthropic_key:
                try:
                    resp = claude.generate_answer(
                        question=cleaned_question,
                        context=context,
                        life_stage=canonical_stage,
                        user_context=user_context,
                    )
                    return resp.model_dump()
                except AnthropicAPIKeyMissingError:
                    pass  # Fallback to direct evidence assembly below

            # 5. Host-side Reasoning Fallback (if no server-side Claude API key configured)
            # Returns formatted evidence directly so Claude Desktop can reason over it!
            assembled_answer = (
                f"### Verified Evidence for '{canonical_stage.replace('_', ' ').title()}' Journey\n\n"
                f"{context.formatted_context}\n\n"
                "*Note: Guidance derived directly from indexed knowledge base documents.*"
            )
            fallback_resp = AnswerResponse(
                question=cleaned_question,
                answer=assembled_answer,
                life_stage=canonical_stage,
                citations=context.citations,
                evidence_grounded=True,
            )
            return fallback_resp.model_dump()

        except FemaleWellnessError as exc:
            return exc.to_dict()
        except Exception as exc:
            logger.error("Error in ask_wellness_question: %s", exc)
            return {"error": "AnswerGenerationFailed", "message": str(exc)}

    @mcp.tool(
        name="get_wellness_recommendations",
        description=(
            "Retrieve evidence-based lifestyle, nutrition, and wellness recommendations "
            "relevant to the selected journey. Grounded in retrieved evidence without prescribing "
            "treatments or clinical diagnoses."
        ),
    )
    def get_wellness_recommendations(
        life_stage: str,
        category: str | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve grounded wellness recommendations for a chosen life stage."""
        logger.info("Tool 'get_wellness_recommendations' stage='%s' cat='%s'", life_stage, category)
        try:
            canonical_stage = life_stages.validate_life_stage(life_stage)

            # Query for general lifestyle guidance in this journey
            query_topic = f"{canonical_stage} {category or ''} lifestyle recovery nutrition wellness"
            context = rag.retrieve_context(
                query=query_topic,
                life_stage=canonical_stage,
                category=category,
                top_k=settings.TOP_K,
                min_similarity=settings.SIMILARITY_THRESHOLD,
            )

            if settings.has_anthropic_key and context.has_sufficient_evidence:
                try:
                    resp = claude.generate_recommendations(
                        life_stage=canonical_stage,
                        category=category,
                        context=context,
                        user_context=user_context,
                    )
                    return resp.model_dump()
                except Exception as exc:
                    logger.warning("Claude generation failed in recommendations, using evidence fallback: %s", exc)

            # Direct evidence-grounded recommendation assembly
            items: list[RecommendationItem] = []
            for chunk in context.chunks:
                items.append(
                    RecommendationItem(
                        title=f"{chunk.metadata.category.replace('_', ' ').title()} Guidance",
                        description=chunk.text,
                        category=chunk.metadata.category,
                        life_stage=canonical_stage,
                        evidence_sources=[chunk.metadata.source],
                    )
                )

            rec_resp = RecommendationResponse(
                life_stage=canonical_stage,
                category=category,
                recommendations=items,
                citations=context.citations,
            )
            return rec_resp.model_dump()

        except FemaleWellnessError as exc:
            return exc.to_dict()
        except Exception as exc:
            logger.error("Error in get_wellness_recommendations: %s", exc)
            return {"error": "RecommendationFailed", "message": str(exc)}

