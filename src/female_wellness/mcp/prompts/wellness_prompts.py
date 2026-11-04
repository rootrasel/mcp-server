"""Prompt templates for Claude reasoning, grounded generation, and MCP prompts."""

from typing import Any, Final

# Core system instructions applied to all server-side Claude generations
SYSTEM_WELLNESS_PROMPT: Final[str] = """\
You are an expert Female Wellness & Lifestyle AI specialist. Your role is to provide compassionate, scientifically accurate, evidence-grounded wellness and lifestyle guidance tailored to women across six life stages:
1. Beauty & Radiance (gut-skin axis, cellular hydration, hair, restorative sleep)
2. Cycle & Fertility (menstrual cycle tracking, ovulation, fertility awareness, hormonal rhythm)
3. Athlete Performance (female physiology, training periodization, HRV, recovery, RED-s prevention)
4. Pregnancy & Postpartum (maternal nutrition, trimester wellness, Fourth Trimester recovery)
5. Perimenopause (vasomotor symptom education, thermoregulation, bone density, sleep)
6. Lifelong Thriving (healthy aging, sarcopenia prevention, cognitive vitality, longevity)

CRITICAL OPERATIONAL & SAFETY RULES:
1. Grounded in Evidence: You must ground your response strictly in the provided RETRIEVED EVIDENCE. Do not invent clinical studies, dosage figures, or health claims not supported by the context.
2. Educational, Not Diagnostic: This platform provides lifestyle, nutrition, and wellness education. Never claim to diagnose a medical disease, never prescribe prescription medications, and never advise changing prescription dosages.
3. Physiological Nuance: Do not assume all women have regular 28-day cycles, nor assume all users are fertile, sexually active, or pregnant. Speak with biological precision and physiological empathy.
4. Source Attribution: Explicitly reference the provided sources (e.g., "[Source: filename, Page X]") when citing facts.
5. Uncertainty & Limitations: If the retrieved evidence does not contain sufficient information to answer the question responsibly, state clearly that the knowledge base lacks sufficient verified evidence on that specific topic. Do not speculate on medical matters.
"""


def build_wellness_answer_prompt(
    question: str,
    life_stage: str,
    formatted_evidence: str,
    user_context_str: str = "",
) -> str:
    """Build user message prompt for grounded question answering."""
    user_context_block = (
        f"\nUSER CONTEXT (Consented profile signals):\n{user_context_str}\n"
        if user_context_str.strip()
        else ""
    )

    return f"""\
SELECTED LIFE STAGE: {life_stage}
{user_context_block}
RETRIEVED EVIDENCE:
{formatted_evidence if formatted_evidence.strip() else "[No verified evidence found in knowledge base]"}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer the user's question clearly, warmly, and thoroughly.
- Directly cite the evidence where appropriate.
- Conclude with an educational reminder that this guidance supports lifestyle and wellness and is not a substitute for clinical healthcare.
"""


def build_recommendations_prompt(
    life_stage: str,
    category: str | None,
    formatted_evidence: str,
    user_context_str: str = "",
) -> str:
    """Build user message prompt for generating evidence-grounded lifestyle recommendations."""
    cat_str = f" | Category: {category}" if category else ""
    user_context_block = (
        f"\nUSER CONTEXT:\n{user_context_str}\n" if user_context_str.strip() else ""
    )

    return f"""\
TARGET JOURNEY: {life_stage}{cat_str}
{user_context_block}
RETRIEVED EVIDENCE:
{formatted_evidence}

INSTRUCTIONS:
Provide 3 to 5 actionable, evidence-grounded lifestyle and wellness recommendations for the '{life_stage}' journey.
Format each recommendation with:
- Title
- Detailed guidance (nutrition, circadian rhythm, training, or recovery)
- Grounding source citation
- Relevant wellness category
"""


def build_summarize_prompt(formatted_evidence: str, max_tokens: int = 400) -> str:
    """Build user message prompt for summarizing retrieved knowledge."""
    return f"""\
Please synthesize the following female wellness evidence into an executive summary of key takeaways (approx. {max_tokens} tokens):

RETRIEVED EVIDENCE:
{formatted_evidence}

Focus on key physiological mechanisms, practical lifestyle implications, and relevant life stages.
"""


def register_wellness_prompts(mcp: Any) -> None:
    """Register reusable prompt templates on the MCP server instance.

    Args:
        mcp: FastMCP or MCPServer instance.
    """

    @mcp.prompt(name="wellness_question_prompt")
    def wellness_question_prompt(question: str, life_stage: str) -> str:
        """Prompt template guiding Claude to answer female wellness questions with citations."""
        return (
            f"You are answering a female wellness question within the '{life_stage}' journey.\n"
            f"Question: {question}\n\n"
            "Requirements:\n"
            "1. Ground all recommendations strictly in retrieved evidence.\n"
            "2. Distinguish wellness education from clinical medical advice.\n"
            "3. Cite source documents with page numbers where provided.\n"
            "4. Acknowledge physiological variation across individual cycles."
        )

    @mcp.prompt(name="life_stage_recommendation_prompt")
    def life_stage_recommendation_prompt(life_stage: str, category: str = "general") -> str:
        """Prompt template for generating phase-appropriate lifestyle recommendations."""
        return (
            f"Generate evidence-based lifestyle recommendations for the '{life_stage}' journey "
            f"focusing on the '{category}' domain.\n"
            "Emphasize practical nutrition, restorative sleep, and cycle-aware exercise guidance."
        )

    @mcp.prompt(name="evidence_based_wellness_prompt")
    def evidence_based_wellness_prompt(topic: str, life_stage: str) -> str:
        """Prompt template directing Claude to synthesize scientific wellness evidence."""
        return (
            f"Analyze the following female wellness topic: '{topic}' for a user in the '{life_stage}' life stage.\n"
            "Explain biological mechanisms, summarize relevant studies from the knowledge base, "
            "and emphasize evidence-based lifestyle adaptations."
        )
