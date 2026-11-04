"""Unit and integration tests for MCP server tools, resources, and prompts."""

import json

import pytest

from female_wellness.server import create_mcp_server


@pytest.fixture(scope="module")
def mcp_server():
    """Initialize MCP server instance for testing."""
    return create_mcp_server()


def test_server_initialization(mcp_server):
    """Verify server initializes with proper metadata."""
    assert mcp_server.name == "Female Wellness AI Platform"
    assert mcp_server.version == "0.1.0"


def test_registered_tools_count_and_names(mcp_server):
    """Verify all seven required MCP tools are registered."""
    tool_manager = getattr(mcp_server, "_tool_manager", None)
    assert tool_manager is not None

    registered_tools = tool_manager._tools
    expected_tools = {
        "get_available_life_stages",
        "get_life_stage_details",
        "search_wellness_knowledge",
        "ask_wellness_question",
        "get_wellness_recommendations",
        "get_knowledge_sources",
        "get_knowledge_base_stats",
    }
    assert expected_tools.issubset(set(registered_tools.keys()))


def test_tool_get_available_life_stages(mcp_server):
    """Verify get_available_life_stages returns all six primary journeys."""
    tool = mcp_server._tool_manager._tools["get_available_life_stages"]
    result = tool.fn()

    assert isinstance(result, list)
    assert len(result) == 6
    ids = [item["id"] for item in result]
    assert "beauty_radiance" in ids
    assert "cycle_fertility" in ids
    assert "athlete_performance" in ids
    assert "pregnancy_postpartum" in ids
    assert "perimenopause" in ids
    assert "lifelong_thriving" in ids


def test_tool_get_life_stage_details(mcp_server):
    """Verify get_life_stage_details returns valid schema for a stage."""
    tool = mcp_server._tool_manager._tools["get_life_stage_details"]
    result = tool.fn(life_stage="perimenopause")

    assert result["id"] == "perimenopause"
    assert result["display_name"] == "Perimenopause"
    assert "hot_flash_education" in result["supported_categories"]
    assert len(result["features"]) >= 4


def test_tool_get_knowledge_base_stats(mcp_server):
    """Verify get_knowledge_base_stats returns indexed collection metrics."""
    tool = mcp_server._tool_manager._tools["get_knowledge_base_stats"]
    stats = tool.fn()

    assert stats["collection_name"] == "female_wellness_knowledge"
    assert stats["total_chunks"] >= 20
    assert stats["total_documents"] >= 7
    assert "pregnancy_postpartum" in stats["available_life_stages"]


def test_tool_search_wellness_knowledge(mcp_server):
    """Verify search_wellness_knowledge returns relevant results with metadata."""
    tool = mcp_server._tool_manager._tools["search_wellness_knowledge"]
    response = tool.fn(
        query="cellular hydration and skin barrier lipids",
        life_stage="beauty_radiance",
        top_k=2,
    )

    assert response["total_results"] >= 1
    top_result = response["results"][0]
    assert top_result["life_stage"] in ("beauty_radiance", "general")
    assert "similarity_score" in top_result
    assert top_result["similarity_score"] > 0.35


def test_tool_ask_wellness_question(mcp_server):
    """Verify ask_wellness_question returns evidence-grounded answer with citations."""
    tool = mcp_server._tool_manager._tools["ask_wellness_question"]
    response = tool.fn(
        question="What nutrients support Fourth Trimester pelvic recovery?",
        life_stage="pregnancy_postpartum",
    )

    assert response["evidence_grounded"] is True
    assert response["life_stage"] == "pregnancy_postpartum"
    assert len(response["citations"]) >= 1
    assert "maternal_nutrition_recovery.md" in response["citations"][0]["source"]
    assert "disclaimer" in response


def test_tool_get_wellness_recommendations(mcp_server):
    """Verify get_wellness_recommendations returns structured items."""
    tool = mcp_server._tool_manager._tools["get_wellness_recommendations"]
    response = tool.fn(
        life_stage="athlete_performance",
        category="recovery_analysis",
    )

    assert response["life_stage"] == "athlete_performance"
    assert len(response["recommendations"]) >= 1
    assert len(response["citations"]) >= 1


def test_registered_resources(mcp_server):
    """Verify all three wellness:// resources are registered and readable."""
    resource_manager = getattr(mcp_server, "_resource_manager", None)
    assert resource_manager is not None

    resources = resource_manager._resources
    assert "wellness://life-stages" in resources
    assert "wellness://knowledge-base" in resources
    assert "wellness://safety-guidelines" in resources

    # Test reading safety guidelines
    safety_fn = resources["wellness://safety-guidelines"].fn
    safety_content = safety_fn()
    assert "Safety & Ethics Guidelines" in safety_content
    assert "Red-Flag Symptoms" in safety_content

    # Test reading life stages resource
    stages_fn = resources["wellness://life-stages"].fn
    stages_json = json.loads(stages_fn())
    assert stages_json["total_journeys"] == 6


def test_registered_prompts(mcp_server):
    """Verify all three prompt templates are registered."""
    prompt_manager = getattr(mcp_server, "_prompt_manager", None)
    assert prompt_manager is not None

    prompts = prompt_manager._prompts
    assert "wellness_question_prompt" in prompts
    assert "life_stage_recommendation_prompt" in prompts
    assert "evidence_based_wellness_prompt" in prompts

    # Test prompt generation
    p_fn = prompts["wellness_question_prompt"].fn
    prompt_str = p_fn(question="What is the luteal phase?", life_stage="cycle_fertility")
    assert "cycle_fertility" in prompt_str
    assert "What is the luteal phase?" in prompt_str
