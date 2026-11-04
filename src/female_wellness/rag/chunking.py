"""Sentence-boundary aware text chunker for the RAG ingestion pipeline."""

from female_wellness.core.exceptions import DocumentIngestionError
from female_wellness.core.logging_config import get_logger
from female_wellness.schemas.documents import ChunkMetadata, DocumentChunk, LoadedDocument
from female_wellness.utils.text_utils import split_into_sentences

logger = get_logger("rag.chunking")


class SentenceAwareChunker:
    """Chunks documents into semantically coherent segments respecting sentence boundaries.

    Maintains chunk overlap for context preservation and assigns accurate page-level
    metadata to every chunk.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        """Initialize chunker with size and overlap parameters.

        Args:
            chunk_size: Maximum target character length of each chunk.
            chunk_overlap: Desired character overlap between consecutive chunks.

        Raises:
            DocumentIngestionError: If overlap is greater than or equal to chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise DocumentIngestionError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text_block(self, text: str) -> list[str]:
        """Split a continuous text block into chunks respecting sentence boundaries.

        Args:
            text: Cleaned text block (typically a single page or section).

        Returns:
            List of chunk strings.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        sentences = split_into_sentences(text)
        if not sentences:
            return [text]

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If a single sentence exceeds chunk_size, split by words
            if sentence_len > self.chunk_size:
                # Flush existing buffer first
                if current_sentences:
                    chunk_str = " ".join(current_sentences).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                    current_sentences = []
                    current_len = 0

                # Split long sentence by word boundaries
                words = sentence.split()
                w_buf: list[str] = []
                w_len = 0
                for w in words:
                    if w_len + len(w) + 1 > self.chunk_size and w_buf:
                        chunks.append(" ".join(w_buf))
                        # Retain overlap words
                        overlap_words: list[str] = []
                        overlap_len = 0
                        for ow in reversed(w_buf):
                            if overlap_len + len(ow) + 1 <= self.chunk_overlap:
                                overlap_words.insert(0, ow)
                                overlap_len += len(ow) + 1
                            else:
                                break
                        w_buf = overlap_words
                        w_len = sum(len(x) + 1 for x in w_buf)

                    w_buf.append(w)
                    w_len += len(w) + 1

                if w_buf:
                    chunks.append(" ".join(w_buf))
                continue

            # Standard case: test if adding sentence exceeds chunk_size
            added_len = sentence_len + (1 if current_sentences else 0)
            if current_len + added_len > self.chunk_size and current_sentences:
                # Store completed chunk
                chunk_str = " ".join(current_sentences).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Build overlap seed from trailing sentences
                overlap_sentences: list[str] = []
                overlap_len = 0
                for prev in reversed(current_sentences):
                    prev_len = len(prev) + (1 if overlap_sentences else 0)
                    if overlap_len + prev_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev)
                        overlap_len += prev_len
                    else:
                        break

                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences) + max(
                    0, len(current_sentences) - 1
                )

            current_sentences.append(sentence)
            current_len += sentence_len + (1 if len(current_sentences) > 1 else 0)

        # Flush any remaining sentences in buffer
        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            if chunk_str and (not chunks or chunks[-1] != chunk_str):
                chunks.append(chunk_str)

        return chunks

    def chunk_document(self, document: LoadedDocument) -> list[DocumentChunk]:
        """Split a LoadedDocument into structured DocumentChunk items with full metadata.

        Processes page-by-page to accurately record the originating page number.

        Args:
            document: LoadedDocument with metadata and pages.

        Returns:
            List of DocumentChunk instances ready for embedding and indexing.
        """
        raw_chunks_with_pages: list[tuple[str, int | None]] = []

        if document.pages:
            for page in document.pages:
                page_chunks = self._chunk_text_block(page.text)
                for chunk_text in page_chunks:
                    raw_chunks_with_pages.append((chunk_text, page.page_number))
        else:
            # Fallback if no page breakdown is available
            text_chunks = self._chunk_text_block(document.full_text)
            for chunk_text in text_chunks:
                raw_chunks_with_pages.append((chunk_text, None))

        total_chunks = len(raw_chunks_with_pages)
        if total_chunks == 0:
            logger.warning("Document '%s' produced zero chunks.", document.metadata.source)
            return []

        doc_chunks: list[DocumentChunk] = []
        doc_id = document.metadata.document_id

        for idx, (chunk_text, page_num) in enumerate(raw_chunks_with_pages):
            chunk_id = f"{doc_id}_c{idx}"
            meta = ChunkMetadata(
                document_id=doc_id,
                source=document.metadata.source,
                life_stage=document.metadata.life_stage,
                category=document.metadata.category,
                content_type="educational",
                page_number=page_num,
                chunk_index=idx,
                total_chunks=total_chunks,
            )
            doc_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=meta,
                )
            )

        logger.debug(
            "Created %d chunks for document '%s' (size=%d, overlap=%d)",
            total_chunks,
            document.metadata.source,
            self.chunk_size,
            self.chunk_overlap,
        )

        return doc_chunks

