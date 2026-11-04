"""Unit tests for the SafetyService: medical safety rules, red flags, and escalations."""

import pytest

from female_wellness.server import create_mcp_server
from female_wellness.services.safety_service import SafetyService


@pytest.fixture
def safety() -> SafetyService:
    return SafetyService()


def test_preeclampsia_red_flag_detection(safety: SafetyService):
    """Verify preeclampsia symptoms trigger immediate emergency alert in pregnancy."""
    query = "I am in week 32 of pregnancy and have a severe headache with blurry vision and spots."
    eval_result = safety.evaluate_query(query, life_stage="pregnancy_postpartum")

    assert eval_result.is_safe is False
    assert eval_result.is_emergency is True
    assert eval_result.flagged_condition == "Suspected Preeclampsia"
    assert "CRITICAL MEDICAL ALERT" in eval_result.escalation_message


def test_hemorrhage_red_flag_detection(safety: SafetyService):
    """Verify heavy bleeding soaking pads triggers emergency alert."""
    query = "I have heavy bleeding and am soaking more than one pad an hour with sharp pelvic pain."
    eval_result = safety.evaluate_query(query, life_stage="cycle_fertility")

    assert eval_result.is_safe is False
    assert eval_result.is_emergency is True
    assert "CRITICAL MEDICAL ALERT" in eval_result.escalation_message


def test_dvt_calf_swelling_red_flag_detection(safety: SafetyService):
    """Verify unilateral calf pain/swelling triggers DVT warning across all stages."""
    query = "I have sudden unilateral calf swelling, redness, and tender pain."
    eval_result = safety.evaluate_query(query, life_stage="athlete_performance")

    assert eval_result.is_safe is False
    assert eval_result.is_emergency is True
    assert eval_result.flagged_condition == "Suspected Deep Vein Thrombosis"
    assert "URGENT MEDICAL ALERT" in eval_result.escalation_message


def test_cardiac_emergency_detection(safety: SafetyService):
    """Verify crushing chest pain triggers immediate emergency escalation."""
    query = "Experiencing crushing chest pain radiating to my left arm."
    eval_result = safety.evaluate_query(query, life_stage="lifelong_thriving")

    assert eval_result.is_safe is False
    assert eval_result.is_emergency is True
    assert "EMERGENCY ALERT" in eval_result.escalation_message


def test_postpartum_sepsis_infection_detection(safety: SafetyService):
    """Verify postpartum fever with foul discharge triggers infection alert."""
    query = "I have high fever and chills with foul lochia discharge 1 week after birth."
    eval_result = safety.evaluate_query(query, life_stage="pregnancy_postpartum")

    assert eval_result.is_safe is False
    assert eval_result.is_emergency is True
    assert "URGENT MEDICAL ALERT" in eval_result.escalation_message


def test_prescription_dosage_inquiry_intercepted(safety: SafetyService):
    """Verify inquiries asking for medication dosages are cleanly blocked."""
    query = "What dose of metformin should I take for regular cycles?"
    eval_result = safety.evaluate_query(query, life_stage="cycle_fertility")

    assert eval_result.is_safe is False
    assert eval_result.is_emergency is False
    assert eval_result.flagged_condition == "PrescriptionDosageInquiry"
    assert "cannot prescribe or adjust medication dosages" in eval_result.escalation_message


def test_medication_discontinuation_intercepted(safety: SafetyService):
    """Verify advice on stopping prescription medication is blocked."""
    query = "Should I stop taking my levothyroxine during the follicular phase?"
    eval_result = safety.evaluate_query(query, life_stage="cycle_fertility")

    assert eval_result.is_safe is False
    assert eval_result.is_emergency is False
    assert eval_result.flagged_condition == "PrescriptionDosageInquiry"


def test_diagnostic_request_redirection(safety: SafetyService):
    """Verify requests to diagnose disease are redirected to clinical care."""
    query = "Do I have endometriosis or PCOS based on my painful periods?"
    eval_result = safety.evaluate_query(query, life_stage="cycle_fertility")

    assert eval_result.is_safe is False
    assert eval_result.is_emergency is False
    assert eval_result.flagged_condition == "DiagnosticRequest"
    assert "require clinical evaluation" in eval_result.escalation_message


def test_normal_wellness_query_allowed(safety: SafetyService):
    """Verify standard educational wellness inquiries pass safety validation."""
    query = "What dietary foods and micronutrients naturally support follicular phase estrogen?"
    eval_result = safety.evaluate_query(query, life_stage="cycle_fertility")

    assert eval_result.is_safe is True
    assert eval_result.is_emergency is False
    assert eval_result.flagged_condition is None


def test_mcp_tool_ask_question_red_flag_escalation():
    """Verify ask_wellness_question MCP tool intercepts emergencies before RAG/LLM invocation."""
    server = create_mcp_server()
    tool = server._tool_manager._tools["ask_wellness_question"]

    # Incur acute preeclampsia red flag
    response = tool.fn(
        question="Severe headache with blurry vision and sudden facial swelling in week 36",
        life_stage="pregnancy_postpartum",
    )

    assert response["evidence_grounded"] is False
    assert "CRITICAL MEDICAL ALERT" in response["answer"]
    assert "URGENT CLINICAL NOTICE" in response["disclaimer"]
    assert len(response["citations"]) == 0


def test_sanitize_output_appends_disclaimer(safety: SafetyService):
    """Verify sanitize_output guarantees presence of non-medical disclaimer."""
    raw = "Hydration with electrolytes supports cellular recovery."
    sanitized = safety.sanitize_output(raw)
    assert "Disclaimer: This information is for wellness and educational lifestyle purposes only" in sanitized

