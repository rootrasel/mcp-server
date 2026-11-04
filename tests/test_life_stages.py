"""Unit tests for LifeStageService: validation, discovery, and context assembly."""

import pytest

from female_wellness.core.constants import PRIMARY_LIFE_STAGES, LifeStageId
from female_wellness.core.exceptions import LifeStageNotFoundError
from female_wellness.services.life_stage_service import LifeStageService


@pytest.fixture
def service() -> LifeStageService:
    return LifeStageService()


def test_validate_all_primary_life_stages(service: LifeStageService):
    """Verify that all six canonical life stages validate successfully."""
    for stage in PRIMARY_LIFE_STAGES:
        validated = service.validate_life_stage(stage.value)
        assert validated == stage.value


def test_validate_life_stage_normalization(service: LifeStageService):
    """Verify normalization of spaces, hyphens, and mixed casing."""
    assert service.validate_life_stage("BEAUTY_RADIANCE") == "beauty_radiance"
    assert service.validate_life_stage("beauty-radiance") == "beauty_radiance"
    assert service.validate_life_stage("Athlete Performance") == "athlete_performance"
    assert service.validate_life_stage("  pregnancy_postpartum  ") == "pregnancy_postpartum"


def test_validate_invalid_life_stage_raises_error(service: LifeStageService):
    """Verify that unsupported life stage names raise LifeStageNotFoundError."""
    with pytest.raises(LifeStageNotFoundError) as exc_info:
        service.validate_life_stage("unsupported_journey_xyz")

    assert "Unsupported life stage" in str(exc_info.value)
    assert "allowed_stages" in exc_info.value.details


def test_validate_empty_life_stage_raises_error(service: LifeStageService):
    """Verify that empty life stage strings raise LifeStageNotFoundError."""
    with pytest.raises(LifeStageNotFoundError) as exc_info:
        service.validate_life_stage("")
    assert "cannot be empty" in str(exc_info.value)


def test_get_available_life_stages_count_and_content(service: LifeStageService):
    """Verify available journeys list returns all six stages with summaries."""
    stages = service.get_available_life_stages()
    assert len(stages) == 6

    stage_ids = [s.id for s in stages]
    assert LifeStageId.BEAUTY_RADIANCE.value in stage_ids
    assert LifeStageId.CYCLE_FERTILITY.value in stage_ids
    assert LifeStageId.ATHLETE_PERFORMANCE.value in stage_ids
    assert LifeStageId.PREGNANCY_POSTPARTUM.value in stage_ids
    assert LifeStageId.PERIMENOPAUSE.value in stage_ids
    assert LifeStageId.LIFELONG_THRIVING.value in stage_ids

    for s in stages:
        assert len(s.display_name) > 0
        assert len(s.description) > 20


def test_get_life_stage_details(service: LifeStageService):
    """Verify details for a specific life stage include categories and features."""
    details = service.get_life_stage_details("cycle_fertility")

    assert details.id == "cycle_fertility"
    assert details.display_name == "Cycle & Fertility"
    assert "menstrual_cycle" in details.supported_categories
    assert "ovulation_awareness" in details.supported_categories
    assert len(details.features) >= 5
    assert "default_top_k" in details.retrieval_metadata


def test_get_life_stage_context_with_user_attributes(service: LifeStageService):
    """Verify life stage context assembles cleanly and preserves user attributes."""
    user_ctx = {
        "current_cycle_day": 14,
        "primary_goals": ["ovulation_tracking"],
    }
    context = service.get_life_stage_context("cycle_fertility", user_context=user_ctx)

    assert context.life_stage.id == "cycle_fertility"
    assert context.user_context["current_cycle_day"] == 14
    assert "Current Cycle Day: 14" in context.context_summary


def test_get_life_stage_context_privacy_no_inference(service: LifeStageService):
    """Verify that when user context is empty, no sensitive parameters are inferred."""
    context = service.get_life_stage_context("beauty_radiance")
    assert context.user_context == {}
    assert "User Provided Profile Signals" not in context.context_summary

