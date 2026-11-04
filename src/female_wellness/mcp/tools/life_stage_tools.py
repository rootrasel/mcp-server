"""MCP Tools for discovering and exploring female wellness life-stage journeys."""

from typing import Any

from female_wellness.core.exceptions import LifeStageNotFoundError
from female_wellness.core.logging_config import get_logger
from female_wellness.services.life_stage_service import LifeStageService

logger = get_logger("mcp.tools.life_stage")


def register_life_stage_tools(mcp: Any, life_stage_service: LifeStageService | None = None) -> None:
    """Register life-stage tools on the MCP server instance.

    Args:
        mcp: FastMCP or MCPServer instance.
        life_stage_service: Injected life stage service or None for default.
    """
    service = life_stage_service or LifeStageService()

    @mcp.tool(
        name="get_available_life_stages",
        description=(
            "Return all six primary supported female wellness journeys: "
            "Beauty & Radiance, Cycle & Fertility, Athlete Performance, "
            "Pregnancy & Postpartum, Perimenopause, and Lifelong Thriving."
        ),
    )
    def get_available_life_stages() -> list[dict[str, str]]:
        """Return the list of all available female wellness journeys."""
        logger.debug("Tool 'get_available_life_stages' invoked.")
        stages = service.get_available_life_stages()
        return [s.model_dump() for s in stages]

    @mcp.tool(
        name="get_life_stage_details",
        description=(
            "Return comprehensive details, supported wellness categories, and available "
            "features for a specific female wellness journey."
        ),
    )
    def get_life_stage_details(life_stage: str) -> dict[str, Any]:
        """Return full details for a chosen life stage.

        Args:
            life_stage: Journey identifier (e.g., 'beauty_radiance', 'cycle_fertility').
        """
        logger.debug("Tool 'get_life_stage_details' invoked for stage='%s'", life_stage)
        try:
            detail = service.get_life_stage_details(life_stage)
            return detail.model_dump()
        except LifeStageNotFoundError as exc:
            return exc.to_dict()

