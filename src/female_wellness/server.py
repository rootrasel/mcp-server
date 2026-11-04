"""Female Wellness & Lifestyle Model Context Protocol (MCP) Server.

Entry point for starting the FastMCP STDIO server.
Exposes 7 MCP tools, 3 resources, and 3 prompt templates to MCP clients
(such as Claude Desktop).
"""

import sys

from female_wellness.config import get_settings
from female_wellness.core.logging_config import get_logger, setup_logging
from female_wellness.mcp.prompts.wellness_prompts import register_wellness_prompts
from female_wellness.mcp.resources.knowledge_resources import register_knowledge_resources
from female_wellness.mcp.tools.knowledge_tools import register_knowledge_tools
from female_wellness.mcp.tools.life_stage_tools import register_life_stage_tools
from female_wellness.mcp.tools.recommendation_tools import register_recommendation_tools
from female_wellness.rag.pipeline import RAGPipeline
from female_wellness.services.anthropic_service import AnthropicService
from female_wellness.services.life_stage_service import LifeStageService
from female_wellness.services.safety_service import SafetyService

# FastMCP / MCPServer backward and forward compatibility
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]

logger = get_logger("server")


def create_mcp_server() -> FastMCP:
    """Create and configure the FastMCP server instance.

    Initializes services, registers all tools, resources, and prompt templates.

    Returns:
        Configured FastMCP / MCPServer instance.
    """
    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL)

    logger.info("Initializing Female Wellness & Lifestyle MCP Server...")

    # Initialize FastMCP server
    server = FastMCP(
        name="Female Wellness AI Platform",
        version="0.1.0",
        log_level=settings.LOG_LEVEL,
    )

    # Initialize domain services
    life_stage_service = LifeStageService()
    rag_pipeline = RAGPipeline()
    anthropic_service = AnthropicService(settings=settings)
    safety_service = SafetyService()

    # 1. Register Tools (7 tools)
    register_life_stage_tools(server, life_stage_service=life_stage_service)
    register_knowledge_tools(server, pipeline=rag_pipeline, vector_store=rag_pipeline.vector_store)
    register_recommendation_tools(
        server,
        pipeline=rag_pipeline,
        anthropic_service=anthropic_service,
        life_stage_service=life_stage_service,
        safety_service=safety_service,
    )

    # 2. Register Resources (3 resources)
    register_knowledge_resources(
        server,
        life_stage_service=life_stage_service,
        vector_store=rag_pipeline.vector_store,
    )

    # 3. Register Prompts (3 prompts)
    register_wellness_prompts(server)

    logger.info("Female Wellness MCP Server initialized with tools, resources, and prompts.")
    return server


def main() -> None:
    """Execute the MCP server using STDIO transport.

    CRITICAL: stdout is reserved strictly for JSON-RPC MCP framing.
    All logging is directed to stderr.
    """
    try:
        server = create_mcp_server()
        # Default STDIO transport for Claude Desktop
        server.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server terminated by user.")
        sys.exit(0)
    except Exception as exc:
        logger.critical("Fatal server startup failure: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
