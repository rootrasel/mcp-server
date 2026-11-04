"""Core constants for the Female Wellness & Lifestyle MCP Server.

Defines supported life-stage identifiers, wellness categories, default
model configurations, and system-wide boundaries.
"""

from enum import StrEnum
from typing import Final


class LifeStageId(StrEnum):
    """The six primary female wellness life-stage journeys supported by the platform."""

    BEAUTY_RADIANCE = "beauty_radiance"
    CYCLE_FERTILITY = "cycle_fertility"
    ATHLETE_PERFORMANCE = "athlete_performance"
    PREGNANCY_POSTPARTUM = "pregnancy_postpartum"
    PERIMENOPAUSE = "perimenopause"
    LIFELONG_THRIVING = "lifelong_thriving"
    GENERAL = "general"


# Supported life stages excluding the cross-stage "general" fallback
PRIMARY_LIFE_STAGES: Final[tuple[LifeStageId, ...]] = (
    LifeStageId.BEAUTY_RADIANCE,
    LifeStageId.CYCLE_FERTILITY,
    LifeStageId.ATHLETE_PERFORMANCE,
    LifeStageId.PREGNANCY_POSTPARTUM,
    LifeStageId.PERIMENOPAUSE,
    LifeStageId.LIFELONG_THRIVING,
)

# Human-readable display titles
LIFE_STAGE_DISPLAY_NAMES: Final[dict[LifeStageId, str]] = {
    LifeStageId.BEAUTY_RADIANCE: "Beauty & Radiance",
    LifeStageId.CYCLE_FERTILITY: "Cycle & Fertility",
    LifeStageId.ATHLETE_PERFORMANCE: "Athlete Performance",
    LifeStageId.PREGNANCY_POSTPARTUM: "Pregnancy & Postpartum",
    LifeStageId.PERIMENOPAUSE: "Perimenopause",
    LifeStageId.LIFELONG_THRIVING: "Lifelong Thriving",
    LifeStageId.GENERAL: "General Female Wellness",
}

# Categories associated with each life-stage journey
LIFE_STAGE_CATEGORIES: Final[dict[LifeStageId, tuple[str, ...]]] = {
    LifeStageId.BEAUTY_RADIANCE: (
        "skin_health",
        "hydration",
        "nutrition",
        "hair_health",
        "sleep_quality",
        "gut_skin_axis",
    ),
    LifeStageId.CYCLE_FERTILITY: (
        "menstrual_cycle",
        "ovulation_awareness",
        "fertility_window",
        "hormonal_patterns",
        "cycle_forecasting",
        "luteal_follicular_health",
    ),
    LifeStageId.ATHLETE_PERFORMANCE: (
        "training_load",
        "recovery_analysis",
        "hrv_tracking",
        "cycle_based_training",
        "fatigue_monitoring",
        "female_sports_nutrition",
    ),
    LifeStageId.PREGNANCY_POSTPARTUM: (
        "pregnancy_weeks",
        "symptom_education",
        "maternal_nutrition",
        "sleep_ergonomics",
        "maternal_wellness",
        "postpartum_recovery",
    ),
    LifeStageId.PERIMENOPAUSE: (
        "hot_flash_education",
        "sleep_disturbances",
        "mood_changes",
        "hormonal_transition",
        "bone_density",
        "lifestyle_adjustments",
    ),
    LifeStageId.LIFELONG_THRIVING: (
        "healthy_aging",
        "cognitive_wellness",
        "vitality_scoring",
        "mobility_joint_health",
        "preventive_health",
        "longevity_lifestyle",
    ),
    LifeStageId.GENERAL: (
        "general_nutrition",
        "stress_management",
        "circadian_rhythm",
        "preventive_guidance",
    ),
}

# Standard defaults
DEFAULT_CHROMA_PERSIST_DIR: Final[str] = "./chroma_db"
DEFAULT_COLLECTION_NAME: Final[str] = "female_wellness_knowledge"
DEFAULT_EMBEDDING_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"
DEFAULT_ANTHROPIC_MODEL: Final[str] = "claude-3-5-sonnet-20241022"

# RAG Defaults
DEFAULT_CHUNK_SIZE: Final[int] = 800
DEFAULT_CHUNK_OVERLAP: Final[int] = 150
DEFAULT_TOP_K: Final[int] = 5
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.35

# MCP Resource URI schemes
RESOURCE_SCHEME: Final[str] = "wellness"

