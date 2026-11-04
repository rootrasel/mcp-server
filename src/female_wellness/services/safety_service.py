"""Female health safety, red-flag escalation, and clinical boundary engine.

Implements deterministic rule-based safety validation independent of LLM generation
to guarantee patient safety and ethical female wellness guidance.
"""

import re
from typing import ClassVar, NamedTuple

from female_wellness.core.constants import LifeStageId
from female_wellness.core.logging_config import get_logger

logger = get_logger("services.safety")


class RedFlagRule(NamedTuple):
    """Definition of an urgent medical red-flag condition."""

    condition_name: str
    patterns: list[re.Pattern]
    escalation_message: str
    relevant_stages: tuple[str, ...] = ()  # Empty means applies to all


class SafetyEvaluation(NamedTuple):
    """Result of safety validation check."""

    is_safe: bool
    is_emergency: bool
    flagged_condition: str | None
    escalation_message: str | None
    recommended_action: str | None


class SafetyService:
    """Deterministic medical safety, red-flag escalation, and ethical boundary engine."""

    # 1. Red-Flag Emergency Patterns
    _RED_FLAG_RULES: ClassVar[list[RedFlagRule]] = [
        # Gestational Preeclampsia Warning Signs
        RedFlagRule(
            condition_name="Suspected Preeclampsia",
            patterns=[
                re.compile(r"\b(?:severe|intense|pounding)\s+headache\b.*\b(?:vision|blurry|spots|aura|flashing)\b", re.IGNORECASE),
                re.compile(r"\b(?:vision|blurry|spots|aura|flashing)\b.*\b(?:severe|intense|pounding)\s+headache\b", re.IGNORECASE),
                re.compile(r"\b(?:sudden|rapid|severe)\s+swelling\b.*\b(?:face|hands|eyes)\b", re.IGNORECASE),
                re.compile(r"\bright\s+upper\s+quadrant\s+pain\b.*\bpregnant\b", re.IGNORECASE),
            ],
            escalation_message=(
                "CRITICAL MEDICAL ALERT: The symptoms described (severe headache with visual disturbances "
                "or sudden facial swelling) are potential warning signs of preeclampsia, a serious hypertensive "
                "condition of pregnancy. Please contact your obstetrician, midwife, or visit an emergency room "
                "IMMEDIATELY for blood pressure and urinary protein assessment."
            ),
            relevant_stages=(LifeStageId.PREGNANCY_POSTPARTUM.value,),
        ),
        # Obstetric Hemorrhage / Ectopic Warning Signs
        RedFlagRule(
            condition_name="Obstetric Hemorrhage or Ectopic Risk",
            patterns=[
                re.compile(r"\bheavy\s+bleeding\b.*\bsoak(?:ing)?\s+(?:more\s+than\s+)?(?:one|1|two|2)\s+pad", re.IGNORECASE),
                re.compile(r"\bsoak(?:ing)?\s+(?:more\s+than\s+)?(?:one|1|two|2)\s+pad\b.*\bhour", re.IGNORECASE),
                re.compile(r"\b(?:severe|sharp|knife-like)\s+(?:one-sided|unilateral)\s+pelvic\s+pain\b", re.IGNORECASE),
                re.compile(r"\bshoulder\s+tip\s+pain\b.*\b(?:pelvic|pregnant|bleeding)\b", re.IGNORECASE),
            ],
            escalation_message=(
                "CRITICAL MEDICAL ALERT: Abnormally heavy bleeding (soaking through pads within an hour) "
                "or severe unilateral pelvic pain with shoulder pain may indicate an acute obstetric complication "
                "such as an ectopic pregnancy or hemorrhage. Please seek IMMEDIATE emergency medical evaluation."
            ),
            relevant_stages=(LifeStageId.PREGNANCY_POSTPARTUM.value, LifeStageId.CYCLE_FERTILITY.value),
        ),
        # Deep Vein Thrombosis (DVT) Warning Signs
        RedFlagRule(
            condition_name="Suspected Deep Vein Thrombosis",
            patterns=[
                re.compile(r"\b(?:one|single|unilateral|left|right)\s+calf\s+(?:pain|swelling|tenderness|redness)\b", re.IGNORECASE),
                re.compile(r"\bcalf\s+(?:swelling|tenderness|warmth)\b.*\b(?:painful|red)\b", re.IGNORECASE),
            ],
            escalation_message=(
                "URGENT MEDICAL ALERT: Unilateral calf swelling, heat, redness, or pain can be an indicator "
                "of Deep Vein Thrombosis (blood clot), especially during pregnancy, postpartum, or hormonal transitions. "
                "Do not massage the area. Please seek urgent clinical assessment or ultrasound imaging right away."
            ),
        ),
        # Cardiovascular Warning Signs
        RedFlagRule(
            condition_name="Acute Cardiovascular Distress",
            patterns=[
                re.compile(r"\b(?:crushing|severe|squeezing)\s+chest\s+pain\b", re.IGNORECASE),
                re.compile(r"\bchest\s+pain\b.*\b(?:radiating|left\s+arm|jaw|neck)\b", re.IGNORECASE),
                re.compile(r"\bsevere\s+(?:shortness\s+of\s+breath|dyspnea)\b.*\brest\b", re.IGNORECASE),
            ],
            escalation_message=(
                "EMERGENCY ALERT: Severe crushing chest pain, pain radiating to the jaw or arm, and acute shortness "
                "of breath require IMMEDIATE emergency evaluation. Please call emergency services (e.g. 911 / 999 / 112) "
                "or proceed to the nearest emergency room immediately."
            ),
        ),
        # Acute Severe Postpartum Infection
        RedFlagRule(
            condition_name="Acute Postpartum Sepsis Risk",
            patterns=[
                re.compile(r"\b(?:high\s+fever|chills)\b.*\b(?:foul|stink|smelly)\s+(?:lochia|discharge|bleeding)\b", re.IGNORECASE),
                re.compile(r"\b(?:foul|stink|smelly)\s+(?:lochia|discharge)\b.*\b(?:fever|chills)\b", re.IGNORECASE),
            ],
            escalation_message=(
                "URGENT MEDICAL ALERT: Fever accompanied by foul-smelling postpartum discharge or acute abdominal "
                "tenderness may signify a postpartum uterine infection (endometritis). Please contact your maternity "
                "care provider or seek urgent medical attention."
            ),
            relevant_stages=(LifeStageId.PREGNANCY_POSTPARTUM.value,),
        ),
    ]

    # 2. Medication Dosage & Prescriptive Patterns
    _PRESCRIPTION_PATTERNS: ClassVar[list[re.Pattern]] = [
        re.compile(r"\b(?:what\s+dose|dosage|how\s+much\s+(?:mg|milligrams?))\s+of\s+[a-zA-Z]+\s+should\s+i\s+take\b", re.IGNORECASE),
        re.compile(r"\b(?:should\s+i|can\s+i)\s+(?:stop|discontinue|increase|decrease|change\s+the\s+dose\s+of)\s+(?:taking\s+)?(?:my\s+)?(?:medication|prescription|pills?|levothyroxine|metformin|hrt|antidepressant)\b", re.IGNORECASE),
        re.compile(r"\bprescribe\s+(?:me\s+)?(?:something|medication|drugs?)\b", re.IGNORECASE),
    ]

    # 3. Diagnostic Confirmation Patterns
    _DIAGNOSIS_PATTERNS: ClassVar[list[re.Pattern]] = [
        re.compile(r"\bdo\s+i\s+have\s+(?:endometriosis|pcos|cancer|adenomyosis|infertility|fibroids)\b", re.IGNORECASE),
        re.compile(r"\bdiagnose\s+(?:my|me\s+with)\b", re.IGNORECASE),
    ]

    def evaluate_query(
        self,
        query: str,
        life_stage: str | None = None,
    ) -> SafetyEvaluation:
        """Evaluate a user query for medical emergencies, diagnostic requests, and prescription boundaries.

        Args:
            query: User plaintext question.
            life_stage: Selected wellness journey.

        Returns:
            SafetyEvaluation with safety flags and escalation guidance.
        """
        text = query.strip()
        if not text:
            return SafetyEvaluation(
                is_safe=False,
                is_emergency=False,
                flagged_condition="EmptyQuery",
                escalation_message="Query cannot be empty.",
                recommended_action="Please provide a health or wellness topic.",
            )

        # 1. Red-Flag Emergency Screening
        for rule in self._RED_FLAG_RULES:
            if rule.relevant_stages and life_stage and life_stage not in rule.relevant_stages:
                continue
            for pattern in rule.patterns:
                if pattern.search(text):
                    logger.warning("Red flag triggered: '%s' in query", rule.condition_name)
                    return SafetyEvaluation(
                        is_safe=False,
                        is_emergency=True,
                        flagged_condition=rule.condition_name,
                        escalation_message=rule.escalation_message,
                        recommended_action="Seek emergency or urgent clinical medical evaluation immediately.",
                    )

        # 2. Prescription / Dosage Modification Screening
        for pattern in self._PRESCRIPTION_PATTERNS:
            if pattern.search(text):
                logger.info("Prescription dosage inquiry blocked by safety rules.")
                return SafetyEvaluation(
                    is_safe=False,
                    is_emergency=False,
                    flagged_condition="PrescriptionDosageInquiry",
                    escalation_message=(
                        "Prescription medication dosages, adjustments, or discontinuations require individualized "
                        "clinical oversight. As an educational wellness platform, we cannot prescribe or adjust "
                        "medication dosages. Please consult your prescribing physician or pharmacist directly."
                    ),
                    recommended_action="Consult a licensed physician or pharmacist regarding medications.",
                )

        # 3. Clinical Self-Diagnosis Screening
        for pattern in self._DIAGNOSIS_PATTERNS:
            if pattern.search(text):
                logger.info("Medical diagnosis inquiry intercepted by safety rules.")
                return SafetyEvaluation(
                    is_safe=False,
                    is_emergency=False,
                    flagged_condition="DiagnosticRequest",
                    escalation_message=(
                        "Medical conditions such as PCOS, endometriosis, or fibroids require clinical evaluation, "
                        "laboratory biomarker testing, and diagnostic imaging (such as pelvic ultrasound). "
                        "We can provide educational information on these topics, but cannot provide a diagnosis."
                    ),
                    recommended_action="Schedule an in-person diagnostic evaluation with a healthcare provider.",
                )

        return SafetyEvaluation(
            is_safe=True,
            is_emergency=False,
            flagged_condition=None,
            escalation_message=None,
            recommended_action=None,
        )

    def sanitize_output(self, generated_text: str) -> str:
        """Post-process generated text to enforce non-prescriptive disclaimers and tone.

        Args:
            generated_text: Raw output string.

        Returns:
            Sanitized text with safety disclaimers.
        """
        if not generated_text:
            return ""

        disclaimer = (
            "\n\n---\n*Disclaimer: This information is for wellness and educational lifestyle purposes only. "
            "It does not constitute medical diagnosis or prescription. Always consult a licensed healthcare provider "
            "for individualized medical advice.*"
        )

        if "disclaimer" not in generated_text.lower():
            return generated_text.strip() + disclaimer

        return generated_text.strip()
