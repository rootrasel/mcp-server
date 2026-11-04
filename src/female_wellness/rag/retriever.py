"""Retriever component for semantic search across female wellness knowledge."""

from female_wellness.config import get_settings
from female_wellness.core.exceptions import RetrievalError
from female_wellness.core.logging_config import get_logger
from female_wellness.rag.embeddings import EmbeddingEngine
from female_wellness.rag.vector_store import BaseVectorStore, ChromaVectorStore
from female_wellness.schemas.documents import RetrievedChunk

logger = get_logger("rag.retriever")


class WellnessRetriever:
    """Orchestrates query embedding, similarity search, and threshold filtering."""

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        embedding_engine: EmbeddingEngine | None = None,
    ) -> None:
        """Initialize retriever with injected or default components.

        Args:
            vector_store: Vector store interface (defaults to ChromaVectorStore).
            embedding_engine: Dense embedding generator (defaults to EmbeddingEngine).
        """
        settings = get_settings()
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            collection_name=settings.CHROMA_COLLECTION_NAME,
        )
        self.embedding_engine = embedding_engine or EmbeddingEngine(
            model_name=settings.EMBEDDING_MODEL
        )
        self.default_top_k = settings.TOP_K
        self.default_similarity_threshold = settings.SIMILARITY_THRESHOLD

    def retrieve(
        self,
        query: str,
        life_stage: str | None = None,
        category: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        include_general: bool = True,
    ) -> list[RetrievedChunk]:
        """Perform semantic retrieval for a user question or search topic.

        Args:
            query: Plaintext search query.
            life_stage: Target wellness journey identifier or None.
            category: Domain category filter or None.
            top_k: Maximum chunks to retrieve (defaults to settings.TOP_K).
            min_similarity: Cutoff threshold (defaults to settings.SIMILARITY_THRESHOLD).
            include_general: Whether to include cross-stage general documents.

        Returns:
            List of RetrievedChunk models sorted by descending similarity score.

        Raises:
            RetrievalError: If query is blank or underlying vector search fails.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise RetrievalError("Search query cannot be empty.")

        k = top_k if top_k is not None and top_k > 0 else self.default_top_k
        threshold = (
            min_similarity
            if min_similarity is not None and 0.0 <= min_similarity <= 1.0
            else self.default_similarity_threshold
        )

        logger.debug(
            "Retrieving for query='%s' (stage=%s, category=%s, top_k=%d, threshold=%.2f)",
            cleaned_query,
            life_stage,
            category,
            k,
            threshold,
        )

        # 1. Generate query embedding
        try:
            query_vector = self.embedding_engine.embed_text(cleaned_query)
        except Exception as exc:
            raise RetrievalError(f"Failed to generate query embedding: {exc}") from exc

        # 2. Execute similarity search in vector store
        try:
            chunks = self.vector_store.search_documents(
                query_embedding=query_vector,
                top_k=k,
                life_stage=life_stage,
                category=category,
                min_similarity=threshold,
                include_general=include_general,
            )
            logger.info(
                "Retrieved %d chunks for query='%s' (threshold=%.2f)",
                len(chunks),
                cleaned_query,
                threshold,
            )
            return chunks
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

