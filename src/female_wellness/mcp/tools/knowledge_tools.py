"""MCP Tools for searching the knowledge base and inspecting collection statistics."""

from typing import Any

from female_wellness.core.exceptions import FemaleWellnessError
from female_wellness.core.logging_config import get_logger
from female_wellness.rag.pipeline import RAGPipeline
from female_wellness.rag.vector_store import BaseVectorStore, ChromaVectorStore

logger = get_logger("mcp.tools.knowledge")


def register_knowledge_tools(
    mcp: Any,
    pipeline: RAGPipeline | None = None,
    vector_store: BaseVectorStore | None = None,
) -> None:
    """Register knowledge search and inspection tools on the MCP server.

    Args:
        mcp: FastMCP or MCPServer instance.
        pipeline: Injected RAGPipeline or None for default.
        vector_store: Injected vector store or None for default.
    """
    rag = pipeline or RAGPipeline()
    store = vector_store or rag.vector_store

    @mcp.tool(
        name="search_wellness_knowledge",
        description=(
            "Search the female wellness knowledge base using semantic similarity and "
            "optional life stage and category filters. Returns relevant text chunks, "
            "source documents, page numbers, and similarity relevance scores."
        ),
    )
    def search_wellness_knowledge(
        query: str,
        life_stage: str | None = None,
        category: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Execute semantic search across indexed female health knowledge."""
        logger.debug(
            "Tool 'search_wellness_knowledge' query='%s' stage=%s cat=%s k=%d",
            query,
            life_stage,
            category,
            top_k,
        )
        try:
            chunks = rag.retriever.retrieve(
                query=query,
                life_stage=life_stage,
                category=category,
                top_k=top_k,
            )

            results: list[dict[str, Any]] = []
            for c in chunks:
                results.append({
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source": c.metadata.source,
                    "page_number": c.metadata.page_number,
                    "life_stage": c.metadata.life_stage,
                    "category": c.metadata.category,
                    "similarity_score": c.similarity_score,
                })

            return {
                "query": query,
                "life_stage": life_stage,
                "category": category,
                "total_results": len(results),
                "results": results,
            }
        except FemaleWellnessError as exc:
            return exc.to_dict()
        except Exception as exc:
            logger.error("Error in search_wellness_knowledge: %s", exc)
            return {"error": "SearchFailed", "message": str(exc)}

    @mcp.tool(
        name="get_knowledge_sources",
        description=(
            "List source documents and reference materials available in the knowledge base, "
            "optionally filtered by life stage or category."
        ),
    )
    def get_knowledge_sources(
        life_stage: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """List distinct reference documents and their coverage."""
        logger.debug("Tool 'get_knowledge_sources' invoked.")
        try:
            stats = store.get_collection_stats()
            # Retrieve sample metadata records to build source catalog
            if isinstance(store, ChromaVectorStore):
                where_filter: dict[str, Any] | None = None
                if life_stage and category:
                    where_filter = {"$and": [{"life_stage": life_stage}, {"category": category}]}
                elif life_stage:
                    where_filter = {"life_stage": life_stage}
                elif category:
                    where_filter = {"category": category}

                kwargs: dict[str, Any] = {"include": ["metadatas"]}
                if where_filter:
                    kwargs["where"] = where_filter

                records = store._collection.get(**kwargs)
                metas = records.get("metadatas", [])

                sources: dict[str, dict[str, Any]] = {}
                for m in metas:
                    src = m.get("source", "Unknown")
                    if src not in sources:
                        sources[src] = {
                            "source": src,
                            "life_stage": m.get("life_stage", "general"),
                            "category": m.get("category", "general"),
                            "total_chunks": m.get("total_chunks", 1),
                        }

                return {
                    "total_sources": len(sources),
                    "life_stage_filter": life_stage,
                    "category_filter": category,
                    "sources": list(sources.values()),
                }

            return {
                "total_documents": stats.total_documents,
                "available_life_stages": stats.available_life_stages,
            }
        except Exception as exc:
            logger.error("Error in get_knowledge_sources: %s", exc)
            return {"error": "FailedToRetrieveSources", "message": str(exc)}

    @mcp.tool(
        name="get_knowledge_base_stats",
        description=(
            "Return aggregate statistics about the knowledge base: total indexed chunks, "
            "document count, collection name, available life stages, and domain categories."
        ),
    )
    def get_knowledge_base_stats() -> dict[str, Any]:
        """Return knowledge base collection metrics."""
        logger.debug("Tool 'get_knowledge_base_stats' invoked.")
        try:
            stats = store.get_collection_stats()
            return stats.model_dump()
        except Exception as exc:
            logger.error("Error in get_knowledge_base_stats: %s", exc)
            return {"error": "StatsFailed", "message": str(exc)}

