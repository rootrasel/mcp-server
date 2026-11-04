"""End-to-end RAG pipeline coordinating document ingestion, indexing, and context assembly."""

from pathlib import Path
from typing import Any

from female_wellness.config import get_settings
from female_wellness.core.exceptions import DocumentIngestionError
from female_wellness.core.logging_config import get_logger
from female_wellness.rag.chunking import SentenceAwareChunker
from female_wellness.rag.document_loader import DocumentLoader
from female_wellness.rag.embeddings import EmbeddingEngine
from female_wellness.rag.retriever import WellnessRetriever
from female_wellness.rag.vector_store import BaseVectorStore, ChromaVectorStore
from female_wellness.schemas.wellness import RetrievedContext, SourceCitation

logger = get_logger("rag.pipeline")


class RAGPipeline:
    """Orchestrates document loading, chunking, embedding, vector storage, and evidence assembly."""

    SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        retriever: WellnessRetriever | None = None,
        loader: DocumentLoader | None = None,
        chunker: SentenceAwareChunker | None = None,
    ) -> None:
        """Initialize the pipeline with modular dependencies."""
        settings = get_settings()
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            collection_name=settings.CHROMA_COLLECTION_NAME,
        )
        self.embedding_engine = embedding_engine or EmbeddingEngine(
            model_name=settings.EMBEDDING_MODEL
        )
        self.retriever = retriever or WellnessRetriever(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine,
        )
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or SentenceAwareChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def ingest_file(
        self,
        file_path: Path | str,
        life_stage: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a single document file into the vector store.

        Workflow:
        1. Parse and extract text using DocumentLoader.
        2. Chunk text using SentenceAwareChunker.
        3. Generate dense vector embeddings in batch.
        4. Upsert chunks into vector store with complete metadata.

        Args:
            file_path: Path to target document file.
            life_stage: Target life stage override or None.
            category: Wellness category override or None.

        Returns:
            Dictionary summarizing ingestion metrics.
        """
        path = Path(file_path).resolve()
        logger.info("Ingesting document: %s", path.name)

        # 1. Load document
        loaded_doc = self.loader.load_file(path, life_stage=life_stage, category=category)

        # 2. Chunk document
        chunks = self.chunker.chunk_document(loaded_doc)
        if not chunks:
            raise DocumentIngestionError(f"Document '{path.name}' produced 0 chunks.")

        # 3. Generate embeddings in batch
        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_documents(texts)

        # 4. Attach embeddings to chunks
        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.embedding = emb

        # 5. Upsert to vector store
        indexed_count = self.vector_store.add_documents(chunks)

        logger.info(
            "Successfully indexed '%s': %d chunks into life stage '%s'",
            path.name,
            indexed_count,
            loaded_doc.metadata.life_stage,
        )

        return {
            "document_id": loaded_doc.metadata.document_id,
            "source": loaded_doc.metadata.source,
            "life_stage": loaded_doc.metadata.life_stage,
            "category": loaded_doc.metadata.category,
            "total_chunks": indexed_count,
            "file_type": loaded_doc.metadata.file_type,
            "status": "indexed",
        }

    def ingest_directory(
        self,
        raw_directory: Path | str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Recursively scan and ingest all supported wellness documents in a directory.

        Args:
            raw_directory: Root directory containing raw documents.
            dry_run: If True, parses and chunks without generating embeddings or modifying DB.

        Returns:
            Dictionary summarizing batch ingestion results.
        """
        root_dir = Path(raw_directory).resolve()
        if not root_dir.exists():
            raise DocumentIngestionError(f"Raw documents directory does not exist: {root_dir}")

        all_files = [
            f for f in root_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        logger.info(
            "Found %d candidate documents in '%s' (dry_run=%s)",
            len(all_files),
            root_dir,
            dry_run,
        )

        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        total_chunks = 0

        for file_path in all_files:
            try:
                if dry_run:
                    # In dry-run mode, test loading and chunking without calling embedder or DB
                    doc = self.loader.load_file(file_path)
                    chunks = self.chunker.chunk_document(doc)
                    total_chunks += len(chunks)
                    results.append({
                        "source": file_path.name,
                        "life_stage": doc.metadata.life_stage,
                        "category": doc.metadata.category,
                        "chunks_planned": len(chunks),
                        "status": "dry_run_success",
                    })
                else:
                    ingest_summary = self.ingest_file(file_path)
                    total_chunks += ingest_summary["total_chunks"]
                    results.append(ingest_summary)
            except Exception as exc:
                logger.error("Failed to ingest '%s': %s", file_path.name, exc)
                errors.append({"file": file_path.name, "error": str(exc)})

        return {
            "scanned_files": len(all_files),
            "successful_files": len(results),
            "failed_files": len(errors),
            "total_chunks": total_chunks,
            "dry_run": dry_run,
            "details": results,
            "errors": errors,
        }

    def retrieve_context(
        self,
        query: str,
        life_stage: str | None = None,
        category: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> RetrievedContext:
        """Retrieve relevant chunks and assemble evidence into structured markdown with citations.

        Args:
            query: Plaintext user query or topic.
            life_stage: Life stage filter or None.
            category: Wellness category filter or None.
            top_k: Maximum chunks to retrieve.
            min_similarity: Relevance threshold cutoff.

        Returns:
            RetrievedContext with chunks, deduplicated citations, and formatted markdown.
        """
        chunks = self.retriever.retrieve(
            query=query,
            life_stage=life_stage,
            category=category,
            top_k=top_k,
            min_similarity=min_similarity,
            include_general=True,
        )

        if not chunks:
            return RetrievedContext(
                query=query,
                life_stage=life_stage,
                category=category,
                chunks=[],
                citations=[],
                formatted_context="",
                has_sufficient_evidence=False,
            )

        citations: list[SourceCitation] = []
        context_parts: list[str] = []

        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.metadata
            page_str = f"Page {meta.page_number}" if meta.page_number is not None else "Page N/A"
            relevance_pct = round(chunk.similarity_score * 100, 1)

            # Excerpt for citation summary
            snippet = chunk.text[:140].replace("\n", " ").strip() + "..."

            citations.append(
                SourceCitation(
                    source=meta.source,
                    page_number=meta.page_number,
                    life_stage=meta.life_stage,
                    category=meta.category,
                    chunk_id=chunk.chunk_id,
                    similarity_score=chunk.similarity_score,
                    excerpt=snippet,
                )
            )

            # Markdown formatted evidence block
            context_parts.append(
                f"### Evidence [{idx}]: [Source: {meta.source}, {page_str}] (Relevance: {relevance_pct}%)\n"
                f"- **Life Stage:** `{meta.life_stage}` | **Category:** `{meta.category}`\n"
                f"> {chunk.text}\n"
            )

        formatted_markdown = "\n".join(context_parts)

        return RetrievedContext(
            query=query,
            life_stage=life_stage,
            category=category,
            chunks=chunks,
            citations=citations,
            formatted_context=formatted_markdown,
            has_sufficient_evidence=True,
        )

