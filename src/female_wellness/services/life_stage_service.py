"""Life Stage Context Engine for the six female wellness journeys."""

from typing import Any, Final

from female_wellness.core.constants import (
    LIFE_STAGE_CATEGORIES,
    LIFE_STAGE_DISPLAY_NAMES,
    PRIMARY_LIFE_STAGES,
    LifeStageId,
)
from female_wellness.core.exceptions import LifeStageNotFoundError
from female_wellness.core.logging_config import get_logger
from female_wellness.schemas.wellness import (
    LifeStageContext,
    LifeStageDetail,
    LifeStageSummary,
)

logger = get_logger("services.life_stage")


# Comprehensive journey definitions based on platform product specifications
_JOURNEY_SPECS: Final[dict[str, dict[str, Any]]] = {
    LifeStageId.BEAUTY_RADIANCE.value: {
        "description": (
            "Connect inner health with visible outcomes. Focuses on gut-skin axis health, "
            "epidermal barrier nutrition, cellular hydration, hair follicle vitality, "
            "restorative sleep quality, and hormonal breakout balance."
        ),
        "features": [
            "Skin barrier health tracking",
            "Hydration and electrolyte monitoring",
            "Nutritional guidance for collagen synthesis",
            "Hair follicle and scalp wellness",
            "Sleep quality and nocturnal repair optimization",
            "Skin-related hormonal balance recommendations",
        ],
        "retrieval_metadata": {
            "primary_keywords": ["skin", "hydration", "collagen", "hair", "barrier", "sleep"],
            "default_top_k": 5,
        },
    },
    LifeStageId.CYCLE_FERTILITY.value: {
        "description": (
            "Build hormonal baseline and fertility awareness. Focuses on menstrual cycle phases, "
            "ovulation tracking, fertile window education, natural progesterone support, and "
            "cycle regularity forecasting."
        ),
        "features": [
            "Menstrual cycle tracking (Menstrual, Follicular, Ovulatory, Luteal)",
            "Ovulation awareness and thermal shift confirmation",
            "Fertility window education and cervical fluid monitoring",
            "Hormonal pattern analysis (estrogen and progesterone dynamics)",
            "Cycle regularity forecasting and luteal phase support",
            "Menstrual health education",
        ],
        "retrieval_metadata": {
            "primary_keywords": ["cycle", "ovulation", "progesterone", "estrogen", "fertility", "luteal"],
            "default_top_k": 5,
        },
    },
    LifeStageId.ATHLETE_PERFORMANCE.value: {
        "description": (
            "Optimize training and athletic recovery around female physiology and hormonal rhythms. "
            "Focuses on phase-based periodization, Heart Rate Variability (HRV), and RED-s prevention."
        ),
        "features": [
            "Training load monitoring across follicular and luteal phases",
            "Recovery analysis and autonomic nervous system tracking",
            "HRV tracking and resting heart rate trends",
            "Cycle-based training recommendations",
            "Fatigue monitoring and overtraining prevention",
            "Sports nutrition tailored to female physiology",
        ],
        "retrieval_metadata": {
            "primary_keywords": ["training", "hrv", "recovery", "periodization", "red-s", "endurance"],
            "default_top_k": 5,
        },
    },
    LifeStageId.PREGNANCY_POSTPARTUM.value: {
        "description": (
            "Support maternal wellness throughout pregnancy and Fourth Trimester postpartum recovery. "
            "Focuses on trimester-specific nutrition, sleep ergonomics, and gentle tissue healing."
        ),
        "features": [
            "Pregnancy week tracking and gestational milestones",
            "Symptom education (nausea, fatigue, postural changes)",
            "Maternal nutrition education (choline, folate, iron, DHA)",
            "Sleep ergonomics and musculoskeletal comfort",
            "Maternal wellness and emotional resilience",
            "Fourth trimester postpartum pelvic floor and tissue recovery",
        ],
        "retrieval_metadata": {
            "primary_keywords": ["pregnancy", "postpartum", "trimester", "maternal", "pelvic", "lactation"],
            "default_top_k": 5,
        },
    },
    LifeStageId.PERIMENOPAUSE.value: {
        "description": (
            "Navigate hormonal midlife transitions with clarity, evidence-based thermoregulation, "
            "sleep restoration, and bone mineral density preservation."
        ),
        "features": [
            "Hot flash and night sweat education (hypothalamic thermoregulation)",
            "Sleep disturbance monitoring and circadian support",
            "Mood changes and neurosteroid fluctuation guidance",
            "Hormonal transition education (estrogen fluctuations)",
            "Bone mineral density and resistance training guidance",
            "Cardiovascular and metabolic lifestyle adjustments",
        ],
        "retrieval_metadata": {
            "primary_keywords": ["perimenopause", "hot flash", "vasomotor", "bone density", "estrogen"],
            "default_top_k": 5,
        },
    },
    LifeStageId.LIFELONG_THRIVING.value: {
        "description": (
            "Support long-term health optimization, healthy aging, functional mobility, and "
            "cognitive vitality. Focuses on sarcopenia prevention and preventive longevity."
        ),
        "features": [
            "Healthy aging education and longevity lifestyle strategies",
            "Cognitive wellness and neuroprotection",
            "Vitality scoring and energy optimization",
            "Mobility, joint health, and sarcopenia prevention",
            "Preventive health education and cardiovascular resilience",
            "Long-term metabolic lifestyle optimization",
        ],
        "retrieval_metadata": {
            "primary_keywords": ["longevity", "aging", "sarcopenia", "cognitive", "mobility", "vitality"],
            "default_top_k": 5,
        },
    },
    LifeStageId.GENERAL.value: {
        "description": (
            "Foundational female health principles applying across all life stages, including "
            "circadian entrainment, nervous system regulation, and micronutrient architecture."
        ),
        "features": [
            "Circadian rhythm optimization",
            "Stress neurobiology and vagal regulation",
            "Foundational micronutrient guidance (magnesium, omega-3, vitamin D)",
            "General preventive lifestyle education",
        ],
        "retrieval_metadata": {
            "primary_keywords": ["wellness", "nutrition", "circadian", "stress", "sleep"],
            "default_top_k": 5,
        },
    },
}


class LifeStageService:
    """Service managing life stage validation, discovery, and context resolution."""

    @classmethod
    def validate_life_stage(cls, life_stage: str) -> str:
        """Validate and normalize a user-supplied life-stage identifier.

        Args:
            life_stage: Life-stage identifier string.

        Returns:
            Canonical normalized life-stage identifier.

        Raises:
            LifeStageNotFoundError: If identifier does not match any supported journey.
        """
        if not life_stage:
            raise LifeStageNotFoundError(
                "Life stage identifier cannot be empty.",
                details={"allowed_stages": [s.value for s in PRIMARY_LIFE_STAGES]},
            )

        normalized = life_stage.strip().lower().replace(" ", "_").replace("-", "_")

        # Allow supported primary stages and cross-stage 'general'
        valid_stages = {s.value for s in PRIMARY_LIFE_STAGES} | {LifeStageId.GENERAL.value}
        if normalized not in valid_stages:
            raise LifeStageNotFoundError(
                f"Unsupported life stage: '{life_stage}'. "
                f"Must be one of: {', '.join(sorted(valid_stages))}",
                details={
                    "provided": life_stage,
                    "normalized": normalized,
                    "allowed_stages": sorted(valid_stages),
                },
            )

        return normalized

    def get_available_life_stages(self) -> list[LifeStageSummary]:
        """Return high-level summary of all six primary supported wellness journeys.

        Returns:
            List of LifeStageSummary instances.
        """
        summaries: list[LifeStageSummary] = []
        for stage in PRIMARY_LIFE_STAGES:
            stage_id = stage.value
            display_name = LIFE_STAGE_DISPLAY_NAMES.get(stage, stage_id.title())
            spec = _JOURNEY_SPECS.get(stage_id, {})
            desc = spec.get("description", "")
            summaries.append(
                LifeStageSummary(
                    id=stage_id,
                    display_name=display_name,
                    description=desc,
                )
            )
        return summaries

    def get_life_stage_details(self, life_stage: str) -> LifeStageDetail:
        """Retrieve full operational specification and features for a given journey.

        Args:
            life_stage: Target life stage identifier.

        Returns:
            LifeStageDetail with description, categories, features, and search metadata.

        Raises:
            LifeStageNotFoundError: If identifier is invalid.
        """
        canonical_stage = self.validate_life_stage(life_stage)
        enum_stage = LifeStageId(canonical_stage)
        display_name = LIFE_STAGE_DISPLAY_NAMES.get(enum_stage, canonical_stage.title())

        spec = _JOURNEY_SPECS.get(canonical_stage, {})
        categories = list(LIFE_STAGE_CATEGORIES.get(enum_stage, ()))

        return LifeStageDetail(
            id=canonical_stage,
            display_name=display_name,
            description=spec.get("description", ""),
            supported_categories=categories,
            features=spec.get("features", []),
            retrieval_metadata=spec.get("retrieval_metadata", {}),
        )

    def get_life_stage_context(
        self,
        life_stage: str,
        user_context: dict[str, Any] | None = None,
    ) -> LifeStageContext:
        """Assemble complete context combining validated journey details and consented user context.

        CRITICAL HEALTH SAFETY & PRIVACY RULE:
        Never infer pregnancy, fertility status, or perimenopause automatically.
        Only user-provided, consented context attributes are incorporated.

        Args:
            life_stage: Target life stage identifier.
            user_context: Optional user profile attributes.

        Returns:
            LifeStageContext model ready for downstream reasoning.
        """
        detail = self.get_life_stage_details(life_stage)
        sanitized_context = user_context.copy() if user_context else {}

        # Build clean descriptive summary
        context_lines: list[str] = [
            f"Active Journey: {detail.display_name} ({detail.id})",
            f"Focus: {detail.description}",
        ]

        if sanitized_context:
            context_lines.append("User Provided Profile Signals:")
            for k, v in sanitized_context.items():
                label = k.replace("_", " ").title()
                context_lines.append(f"  • {label}: {v}")

        summary = "\n".join(context_lines)

        return LifeStageContext(
            life_stage=detail,
            user_context=sanitized_context,
            context_summary=summary,
        )

