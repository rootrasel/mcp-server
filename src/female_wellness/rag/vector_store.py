"""Vector database abstraction and ChromaDB implementation.

Provides a modular interface allowing ChromaDB to be replaced by PostgreSQL pgvector
or another vector engine without restructuring RAG pipelines.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from female_wellness.core.constants import DEFAULT_CHROMA_PERSIST_DIR, DEFAULT_COLLECTION_NAME
from female_wellness.core.exceptions import VectorStoreError
from female_wellness.core.logging_config import get_logger
from female_wellness.schemas.documents import (
    ChunkMetadata,
    CollectionStats,
    DocumentChunk,
    RetrievedChunk,
)

logger = get_logger("rag.vector_store")


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage and retrieval."""

    @abstractmethod
    def create_collection(self) -> None:
        """Create or ensure existence of the vector collection."""

    @abstractmethod
    def add_documents(self, chunks: list[DocumentChunk]) -> int:
        """Add or upsert document chunks with vector embeddings.

        Args:
            chunks: List of DocumentChunk instances with populated embeddings.

        Returns:
            Number of successfully indexed chunks.
        """

    @abstractmethod
    def search_documents(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        life_stage: str | None = None,
        category: str | None = None,
        min_similarity: float = 0.0,
        include_general: bool = True,
    ) -> list[RetrievedChunk]:
        """Perform semantic similarity search with metadata filtering.

        Args:
            query_embedding: Dense vector for query.
            top_k: Maximum number of chunks to return.
            life_stage: Target life stage filter.
            category: Wellness category filter.
            min_similarity: Minimum cosine similarity score threshold (0.0 to 1.0).
            include_general: Whether to include cross-stage 'general' content.

        Returns:
            List of RetrievedChunk instances sorted by similarity.
        """

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a given document ID.

        Args:
            document_id: Target document identifier.

        Returns:
            Number of chunks removed.
        """

    @abstractmethod
    def clear_collection(self) -> None:
        """Delete and recreate the collection, wiping all indexed vectors."""

    @abstractmethod
    def get_collection_stats(self) -> CollectionStats:
        """Return aggregate statistics about the vector store."""

    @abstractmethod
    def get_document_by_id(self, document_id: str) -> list[DocumentChunk]:
        """Retrieve all indexed chunks belonging to a document ID."""


class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB implementation using HNSW cosine index."""

    def __init__(
        self,
        persist_directory: Path | str = DEFAULT_CHROMA_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize ChromaDB client.

        Args:
            persist_directory: Local filesystem path where ChromaDB sqlite and vectors reside.
            collection_name: Name of the vector collection.
        """
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._get_or_create_collection()
            logger.info(
                "Initialized ChromaVectorStore (dir='%s', collection='%s')",
                self.persist_directory,
                self.collection_name,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize ChromaDB persistent client: {exc}",
                details={"dir": str(self.persist_directory), "collection": self.collection_name},
            ) from exc

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get or create collection with explicit cosine distance space."""
        # Using cosine distance metric in Chroma HNSW configuration
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def create_collection(self) -> None:
        """Ensure collection exists."""
        self._collection = self._get_or_create_collection()

    def add_documents(self, chunks: list[DocumentChunk]) -> int:
        """Upsert document chunks into ChromaDB.

        Idempotent: Uses chunk_id as primary key to prevent duplicate indexing.

        Args:
            chunks: List of DocumentChunk items.

        Returns:
            Count of indexed chunks.
        """
        if not chunks:
            return 0

        # Validate that embeddings exist
        missing_embed = [c.chunk_id for c in chunks if not c.embedding]
        if missing_embed:
            raise VectorStoreError(
                f"Cannot add chunks to vector store: {len(missing_embed)} chunks lack embeddings.",
                details={"sample_missing_ids": missing_embed[:5]},
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata.to_chroma_metadata() for c in chunks]
        embeddings = [c.embedding for c in chunks]  # type: ignore[misc]

        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.debug(
                "Upserted %d chunks into collection '%s'", len(chunks), self.collection_name
            )
            return len(chunks)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert chunks into ChromaDB collection '{self.collection_name}': {exc}",
                details={"chunk_count": len(chunks), "error": str(exc)},
            ) from exc

    def _build_where_filter(
        self,
        life_stage: str | None,
        category: str | None,
        include_general: bool,
    ) -> dict[str, Any] | None:
        """Construct ChromaDB metadata filter dictionary."""
        filters: list[dict[str, Any]] = []

        if life_stage:
            if include_general and life_stage != "general":
                filters.append(
                    {
                        "$or": [
                            {"life_stage": {"$eq": life_stage}},
                            {"life_stage": {"$eq": "general"}},
                        ]
                    }
                )
            else:
                filters.append({"life_stage": {"$eq": life_stage}})

        if category:
            if include_general and category != "general":
                filters.append(
                    {
                        "$or": [
                            {"category": {"$eq": category}},
                            {"category": {"$eq": "general"}},
                        ]
                    }
                )
            else:
                filters.append({"category": {"$eq": category}})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def search_documents(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        life_stage: str | None = None,
        category: str | None = None,
        min_similarity: float = 0.0,
        include_general: bool = True,
    ) -> list[RetrievedChunk]:
        """Perform similarity search with cosine distance metric and metadata filtering.

        Cosine Distance Metric Explanation:
            ChromaDB HNSW with 'cosine' space calculates cosine distance:
            distance = 1.0 - cosine_similarity
            Therefore:
            cosine_similarity = 1.0 - distance

            When query and documents are normalized vectors, distance is in [0, 2].
            We bound similarity_score in [0.0, 1.0].
        """
        where_filter = self._build_where_filter(life_stage, category, include_general)

        total_count = self._collection.count()
        if total_count == 0:
            logger.debug("Vector store is empty, returning empty search results.")
            return []

        effective_k = min(top_k, total_count)

        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": effective_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kwargs["where"] = where_filter

            results = self._collection.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB query failed: {exc}",
                details={"where": where_filter, "error": str(exc)},
            ) from exc

        retrieved: list[RetrievedChunk] = []
        ids_batch = results.get("ids", [[]])[0]
        docs_batch = results.get("documents", [[]])[0]
        metas_batch = results.get("metadatas", [[]])[0]
        distances_batch = results.get("distances", [[]])[0]

        for chunk_id, text, meta_dict, distance in zip(
            ids_batch, docs_batch, metas_batch, distances_batch, strict=False
        ):
            # ChromaDB cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            raw_dist = float(distance)
            sim_score = max(0.0, min(1.0, 1.0 - raw_dist))

            if sim_score < min_similarity:
                continue

            metadata = ChunkMetadata.from_chroma_metadata(meta_dict)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    distance=raw_dist,
                    similarity_score=round(sim_score, 4),
                )
            )

        # Sort descending by similarity score
        retrieved.sort(key=lambda r: r.similarity_score, reverse=True)
        return retrieved

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document ID."""
        try:
            # Query existing to get count
            existing = self._collection.get(
                where={"document_id": document_id},
                include=[],
            )
            count = len(existing.get("ids", []))
            if count > 0:
                self._collection.delete(where={"document_id": document_id})
                logger.info("Deleted %d chunks for document '%s'", count, document_id)
            return count
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete document '{document_id}' from ChromaDB: {exc}",
                details={"document_id": document_id, "error": str(exc)},
            ) from exc

    def clear_collection(self) -> None:
        """Reset and recreate the collection."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._get_or_create_collection()
            logger.info("Cleared and recreated collection '%s'", self.collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to clear collection '{self.collection_name}': {exc}",
                details={"collection": self.collection_name, "error": str(exc)},
            ) from exc

    def get_collection_stats(self) -> CollectionStats:
        """Compute aggregate collection statistics."""
        try:
            count = self._collection.count()
            if count == 0:
                return CollectionStats(
                    collection_name=self.collection_name,
                    total_chunks=0,
                    total_documents=0,
                    available_life_stages=[],
                    available_categories=[],
                )

            all_records = self._collection.get(include=["metadatas"])
            metadatas = all_records.get("metadatas", [])

            doc_ids = set()
            life_stages = set()
            categories = set()

            for m in metadatas:
                if not m:
                    continue
                if "document_id" in m:
                    doc_ids.add(str(m["document_id"]))
                if "life_stage" in m:
                    life_stages.add(str(m["life_stage"]))
                if "category" in m:
                    categories.add(str(m["category"]))

            return CollectionStats(
                collection_name=self.collection_name,
                total_chunks=count,
                total_documents=len(doc_ids),
                available_life_stages=sorted(life_stages),
                available_categories=sorted(categories),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to retrieve stats for collection '{self.collection_name}': {exc}",
                details={"collection": self.collection_name, "error": str(exc)},
            ) from exc

    def get_document_by_id(self, document_id: str) -> list[DocumentChunk]:
        """Fetch all chunks for a document ID."""
        try:
            records = self._collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
            chunks: list[DocumentChunk] = []
            ids = records.get("ids", [])
            docs = records.get("documents", [])
            metas = records.get("metadatas", [])

            for cid, doc, meta in zip(ids, docs, metas, strict=False):
                chunk_meta = ChunkMetadata.from_chroma_metadata(meta)
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        text=doc,
                        metadata=chunk_meta,
                    )
                )

            chunks.sort(key=lambda c: c.metadata.chunk_index)
            return chunks
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to get chunks for document '{document_id}': {exc}",
                details={"document_id": document_id, "error": str(exc)},
            ) from exc

