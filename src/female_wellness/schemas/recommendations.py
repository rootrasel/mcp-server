"""Schemas for wellness answers, evidence grounding, and personalized recommendations."""

from pydantic import BaseModel, Field

from female_wellness.schemas.wellness import SourceCitation


class RecommendationItem(BaseModel):
    """An individual lifestyle, nutrition, or wellness recommendation."""

    title: str = Field(..., description="Actionable summary title.")
    description: str = Field(..., description="Detailed lifestyle or wellness guidance.")
    category: str = Field(..., description="Wellness domain category (e.g., 'nutrition', 'recovery').")
    life_stage: str = Field(..., description="Target wellness journey.")
    evidence_sources: list[str] = Field(
        default_factory=list,
        description="Identifiable source documents supporting this recommendation.",
    )


class RecommendationResponse(BaseModel):
    """Structured response containing wellness recommendations and clinical disclaimer."""

    life_stage: str = Field(..., description="Target wellness journey.")
    category: str | None = Field(default=None, description="Applied category filter.")
    recommendations: list[RecommendationItem] = Field(
        default_factory=list,
        description="List of evidence-grounded recommendations.",
    )
    citations: list[SourceCitation] = Field(
        default_factory=list,
        description="Source citations backing the recommendations.",
    )
    disclaimer: str = Field(
        default=(
            "This information is for educational and wellness lifestyle purposes only. "
            "It does not constitute medical advice, diagnosis, or prescription."
        ),
        description="Standard female health safety disclaimer.",
    )


class AnswerResponse(BaseModel):
    """Structured answer to a female wellness query with citations and evidence flags."""

    question: str = Field(..., description="User's original question.")
    answer: str = Field(..., description="Synthesized, evidence-grounded response.")
    life_stage: str = Field(..., description="Selected wellness journey.")
    citations: list[SourceCitation] = Field(
        default_factory=list,
        description="Source citations referenced in the response.",
    )
    evidence_grounded: bool = Field(
        default=True,
        description="True if the answer was derived from verified retrieved evidence.",
    )
    disclaimer: str = Field(
        default=(
            "This guidance is educational and focuses on lifestyle and wellness. "
            "Consult a licensed healthcare provider for personal clinical evaluation."
        ),
        description="Standard female health safety disclaimer.",
    )
