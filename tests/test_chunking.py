"""Unit tests for sentence-boundary chunking and metadata preservation."""

import pytest

from female_wellness.core.exceptions import DocumentIngestionError
from female_wellness.rag.chunking import SentenceAwareChunker
from female_wellness.schemas.documents import (
    ChunkMetadata,
    DocumentMetadata,
    LoadedDocument,
    RawPage,
)


def test_chunker_initialization_validation():
    """Verify error when overlap >= size."""
    with pytest.raises(DocumentIngestionError):
        SentenceAwareChunker(chunk_size=100, chunk_overlap=120)


def test_sentence_aware_chunking_boundaries():
    """Verify that chunker preserves sentence boundaries rather than cutting mid-sentence."""
    chunker = SentenceAwareChunker(chunk_size=120, chunk_overlap=30)
    text = (
        "Estrogen plays a key role in collagen synthesis. "
        "During the follicular phase, skin hydration reaches peak levels. "
        "Adequate sleep enhances nocturnal cellular regeneration."
    )
    doc_meta = DocumentMetadata(
        document_id="doc_test1",
        source="skin.txt",
        file_type="txt",
        life_stage="beauty_radiance",
        category="skin_health",
        file_hash="abc12345",
    )
    loaded = LoadedDocument(
        metadata=doc_meta,
        pages=[RawPage(page_number=1, text=text)],
        full_text=text,
    )

    chunks = chunker.chunk_document(loaded)
    assert len(chunks) >= 2

    for c in chunks:
        # Every chunk must end with valid sentence punctuation or boundary
        assert c.text.endswith((".", "!", "?")) or len(c.text) <= 120
        assert c.metadata.life_stage == "beauty_radiance"
        assert c.metadata.source == "skin.txt"
        assert c.metadata.page_number == 1
        assert c.metadata.total_chunks == len(chunks)


def test_multi_page_chunking_metadata():
    """Verify that page numbers are accurately tracked across multi-page documents."""
    chunker = SentenceAwareChunker(chunk_size=100, chunk_overlap=20)
    page1 = RawPage(page_number=1, text="Page 1 discusses ovarian follicle development.")
    page2 = RawPage(page_number=2, text="Page 2 explains luteinizing hormone surges before ovulation.")

    doc_meta = DocumentMetadata(
        document_id="doc_cycle1",
        source="ovulation.pdf",
        file_type="pdf",
        life_stage="cycle_fertility",
        category="ovulation_awareness",
        file_hash="def67890",
    )
    loaded = LoadedDocument(
        metadata=doc_meta,
        pages=[page1, page2],
        full_text=f"{page1.text}\n\n{page2.text}",
    )

    chunks = chunker.chunk_document(loaded)
    assert len(chunks) == 2
    assert chunks[0].metadata.page_number == 1
    assert chunks[1].metadata.page_number == 2


def test_chroma_metadata_serialization():
    """Verify flat dictionary conversion for ChromaDB requirements."""
    meta = ChunkMetadata(
        document_id="doc_xyz",
        source="guide.pdf",
        life_stage="athlete_performance",
        category="recovery_analysis",
        content_type="educational",
        page_number=4,
        chunk_index=2,
        total_chunks=5,
    )
    chroma_dict = meta.to_chroma_metadata()
    assert chroma_dict["document_id"] == "doc_xyz"
    assert chroma_dict["source"] == "guide.pdf"
    assert chroma_dict["page_number"] == 4
    assert chroma_dict["chunk_index"] == 2

    # Round trip
    reconstructed = ChunkMetadata.from_chroma_metadata(chroma_dict)
    assert reconstructed.document_id == meta.document_id
    assert reconstructed.page_number == 4
    assert reconstructed.category == "recovery_analysis"

