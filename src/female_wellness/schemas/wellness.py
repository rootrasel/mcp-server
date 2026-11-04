"""Domain schemas for wellness queries, citations, context, life stages, and user profiles."""

from typing import Any

from pydantic import BaseModel, Field

from female_wellness.schemas.documents import RetrievedChunk


class SourceCitation(BaseModel):
    """Identifiable citation pointing to retrieved source evidence."""

    source: str = Field(..., description="Document source filename (e.g., 'skin_barrier_guide.pdf').")
    page_number: int | None = Field(
        default=None,
        description="1-based page number where evidence was located.",
    )
    life_stage: str = Field(..., description="Target wellness journey.")
    category: str = Field(..., description="Domain category of the cited evidence.")
    chunk_id: str = Field(..., description="Identifier of the specific evidence chunk.")
    similarity_score: float = Field(..., description="Relevance similarity score (0.0 to 1.0).")
    excerpt: str = Field(..., description="Brief snippet or leading excerpt from the source chunk.")


class RetrievedContext(BaseModel):
    """Assembled evidence context ready for answer generation or MCP tool output."""

    query: str = Field(..., description="Original user query.")
    life_stage: str | None = Field(default=None, description="Applied life stage filter.")
    category: str | None = Field(default=None, description="Applied category filter.")
    chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Raw retrieved chunks meeting similarity threshold.",
    )
    citations: list[SourceCitation] = Field(
        default_factory=list,
        description="Deduplicated, structured source citations.",
    )
    formatted_context: str = Field(
        default="",
        description="Structured markdown text ready for LLM consumption.",
    )
    has_sufficient_evidence: bool = Field(
        default=False,
        description="True if one or more chunks met the minimum similarity threshold.",
    )


class UserContext(BaseModel):
    """Optional contextual profile parameters provided by the user."""

    current_cycle_day: int | None = Field(
        default=None,
        ge=1,
        le=60,
        description="Current day of menstrual cycle (if applicable and consented).",
    )
    pregnancy_week: int | None = Field(
        default=None,
        ge=1,
        le=45,
        description="Current gestational week (if applicable and consented).",
    )
    primary_goals: list[str] = Field(
        default_factory=list,
        description="User wellness goals (e.g. ['improve_sleep', 'cycle_awareness']).",
    )
    lifestyle_preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="Dietary or exercise preferences (e.g., {'diet': 'vegetarian'}).",
    )


class LifeStageSummary(BaseModel):
    """Summary overview of a female wellness journey."""

    id: str = Field(..., description="Canonical life stage identifier.")
    display_name: str = Field(..., description="User-facing display name.")
    description: str = Field(..., description="Concise journey description.")


class LifeStageDetail(BaseModel):
    """Complete specification of a female wellness life stage journey."""

    id: str = Field(..., description="Canonical life stage identifier.")
    display_name: str = Field(..., description="User-facing display name.")
    description: str = Field(..., description="Detailed journey description.")
    supported_categories: list[str] = Field(
        default_factory=list,
        description="Supported wellness domains for this journey.",
    )
    features: list[str] = Field(
        default_factory=list,
        description="Core features and capabilities provided in this journey.",
    )
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Search parameters and default retrieval tags.",
    )


class LifeStageContext(BaseModel):
    """Combined context containing validated life stage details and consented user attributes."""

    life_stage: LifeStageDetail
    user_context: dict[str, Any] = Field(default_factory=dict)
    context_summary: str = Field(..., description="Human-readable summary of the active context.")
