"""MCP Resources providing static and dynamic female wellness knowledge context."""

import json
from typing import Any

from female_wellness.core.constants import RESOURCE_SCHEME
from female_wellness.core.logging_config import get_logger
from female_wellness.rag.vector_store import BaseVectorStore, ChromaVectorStore
from female_wellness.services.life_stage_service import LifeStageService

logger = get_logger("mcp.resources.knowledge")

SAFETY_GUIDELINES_TEXT = """\
# Female Wellness & Health Platform: Safety & Ethics Guidelines

## 1. Educational Scope & Non-Diagnostic Boundary
- This platform provides evidence-grounded wellness, lifestyle, and nutrition education.
- The system does NOT provide medical diagnoses, clinical treatment plans, or prescription medications.
- Guidance must never be presented as a substitute for individualized clinical consultation.

## 2. Hormonal & Cycle Individuality
- Avoid making universal generalizations regarding female biology.
- Do not assume all women have regular 28-day menstrual cycles.
- Do not assume users are fertile, pregnant, sexually active, or menstruating without explicit consent.

## 3. Red-Flag Symptoms & Emergency Care
If a user mentions acute, potentially life-threatening or urgent symptoms, the system halts general advice
and immediately directs the user to seek emergency clinical care.

Red-flag examples:
- Pregnancy: Sudden severe headache with visual changes, sudden facial swelling, heavy vaginal bleeding.
- General: Crushing chest pain, severe sudden shortness of breath, acute severe pelvic/abdominal pain.
- Postpartum: Heavy bleeding soaking >1 pad/hour, fever >38°C, unilateral leg swelling.
"""


def register_knowledge_resources(
    mcp: Any,
    life_stage_service: LifeStageService | None = None,
    vector_store: BaseVectorStore | None = None,
) -> None:
    """Register MCP Resources under the 'wellness://' URI scheme.

    Args:
        mcp: FastMCP or MCPServer instance.
        life_stage_service: Injected life stage service.
        vector_store: Injected vector store.
    """
    life_stages = life_stage_service or LifeStageService()
    store = vector_store or ChromaVectorStore()

    @mcp.resource(f"{RESOURCE_SCHEME}://life-stages")
    def get_life_stages_resource() -> str:
        """Resource listing all supported life-stage journeys and specifications."""
        logger.debug("Resource 'wellness://life-stages' accessed.")
        stages = life_stages.get_available_life_stages()
        detailed = [life_stages.get_life_stage_details(s.id).model_dump() for s in stages]
        return json.dumps(
            {
                "platform": "Female Wellness & Lifestyle AI Platform",
                "total_journeys": len(detailed),
                "journeys": detailed,
            },
            indent=2,
        )

    @mcp.resource(f"{RESOURCE_SCHEME}://knowledge-base")
    def get_knowledge_base_resource() -> str:
        """Resource reporting collection statistics, indexed documents, and categories."""
        logger.debug("Resource 'wellness://knowledge-base' accessed.")
        stats = store.get_collection_stats()
        return json.dumps(stats.model_dump(), indent=2)

    @mcp.resource(f"{RESOURCE_SCHEME}://safety-guidelines")
    def get_safety_guidelines_resource() -> str:
        """Resource outlining female health safety rules, disclaimers, and red flags."""
        logger.debug("Resource 'wellness://safety-guidelines' accessed.")
        return SAFETY_GUIDELINES_TEXT

