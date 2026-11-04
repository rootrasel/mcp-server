"""Unit tests for ChromaVectorStore: persistence, metadata filtering, and retrieval."""

from pathlib import Path

import pytest

from female_wellness.core.exceptions import VectorStoreError
from female_wellness.rag.vector_store import ChromaVectorStore
from female_wellness.schemas.documents import ChunkMetadata, DocumentChunk


@pytest.fixture
def temp_store(tmp_path: Path) -> ChromaVectorStore:
    """Create a ChromaVectorStore with a temporary directory."""
    persist_dir = tmp_path / "test_chroma"
    return ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name="test_female_wellness",
    )


def _make_dummy_chunk(
    doc_id: str,
    index: int,
    text: str,
    life_stage: str,
    category: str = "general",
    embedding: list[float] | None = None,
) -> DocumentChunk:
    """Helper to create a test chunk."""
    # Simple deterministic 4-dim unit vector for testing if none provided
    if embedding is None:
        embedding = [0.1 * (index + 1), 0.2, 0.3, 0.4]
    return DocumentChunk(
        chunk_id=f"{doc_id}_c{index}",
        text=text,
        metadata=ChunkMetadata(
            document_id=doc_id,
            source=f"{doc_id}.txt",
            life_stage=life_stage,
            category=category,
            content_type="educational",
            page_number=1,
            chunk_index=index,
            total_chunks=1,
        ),
        embedding=embedding,
    )


def test_empty_vector_store(temp_store: ChromaVectorStore):
    """Verify empty store returns empty search results and zero stats without error."""
    stats = temp_store.get_collection_stats()
    assert stats.total_chunks == 0
    assert stats.total_documents == 0

    results = temp_store.search_documents(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=5,
    )
    assert len(results) == 0


def test_add_and_retrieve_documents(temp_store: ChromaVectorStore):
    """Verify adding documents and basic semantic retrieval."""
    target_vec = [1.0, 0.0, 0.0, 0.0]
    other_vec = [0.0, 1.0, 0.0, 0.0]

    chunk1 = _make_dummy_chunk(
        "doc1", 0, "Hydration for skin barrier.", "beauty_radiance", "hydration", target_vec
    )
    chunk2 = _make_dummy_chunk(
        "doc2", 0, "Luteal phase temperature tracking.", "cycle_fertility", "hormonal_patterns", other_vec
    )

    added = temp_store.add_documents([chunk1, chunk2])
    assert added == 2

    # Query with target_vec
    results = temp_store.search_documents(query_embedding=target_vec, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "doc1_c0"
    assert results[0].similarity_score > 0.99  # Near exact match in cosine space


def test_metadata_filtering_life_stage(temp_store: ChromaVectorStore):
    """Verify filtering by life_stage returns only matching chunks."""
    vec = [0.5, 0.5, 0.5, 0.5]
    c1 = _make_dummy_chunk("doc_beauty", 0, "Collagen and hydration.", "beauty_radiance", embedding=vec)
    c2 = _make_dummy_chunk("doc_cycle", 0, "Ovulation cycle.", "cycle_fertility", embedding=vec)
    c3 = _make_dummy_chunk("doc_general", 0, "General nutrition and sleep.", "general", embedding=vec)

    temp_store.add_documents([c1, c2, c3])

    # Search with beauty_radiance and include_general=True (default)
    results_beauty = temp_store.search_documents(
        query_embedding=vec,
        life_stage="beauty_radiance",
        include_general=True,
    )
    stages = {r.metadata.life_stage for r in results_beauty}
    assert "beauty_radiance" in stages
    assert "general" in stages
    assert "cycle_fertility" not in stages

    # Search with include_general=False
    results_strict = temp_store.search_documents(
        query_embedding=vec,
        life_stage="beauty_radiance",
        include_general=False,
    )
    assert len(results_strict) == 1
    assert results_strict[0].metadata.life_stage == "beauty_radiance"


def test_metadata_filtering_category(temp_store: ChromaVectorStore):
    """Verify category filtering."""
    vec = [0.5, 0.5, 0.5, 0.5]
    c1 = _make_dummy_chunk("doc1", 0, "Skin barrier", "beauty_radiance", "skin_health", vec)
    c2 = _make_dummy_chunk("doc2", 0, "Hair health", "beauty_radiance", "hair_health", vec)
    temp_store.add_documents([c1, c2])

    results = temp_store.search_documents(
        query_embedding=vec,
        life_stage="beauty_radiance",
        category="skin_health",
        include_general=False,
    )
    assert len(results) == 1
    assert results[0].metadata.category == "skin_health"


def test_threshold_cutoff(temp_store: ChromaVectorStore):
    """Verify min_similarity filters out distant results."""
    target_vec = [1.0, 0.0, 0.0, 0.0]
    opposite_vec = [-1.0, 0.0, 0.0, 0.0]

    c1 = _make_dummy_chunk("doc1", 0, "Match", "general", embedding=target_vec)
    c2 = _make_dummy_chunk("doc2", 0, "Opposite", "general", embedding=opposite_vec)
    temp_store.add_documents([c1, c2])

    # High cutoff: should only return c1
    results = temp_store.search_documents(
        query_embedding=target_vec,
        min_similarity=0.8,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "doc1_c0"


def test_delete_document(temp_store: ChromaVectorStore):
    """Verify deleting a document removes all its chunks."""
    vec = [0.1, 0.2, 0.3, 0.4]
    c1 = _make_dummy_chunk("doc_del", 0, "Chunk 0", "general", embedding=vec)
    c2 = _make_dummy_chunk("doc_del", 1, "Chunk 1", "general", embedding=vec)
    c3 = _make_dummy_chunk("doc_keep", 0, "Keep this", "general", embedding=vec)

    temp_store.add_documents([c1, c2, c3])
    stats_before = temp_store.get_collection_stats()
    assert stats_before.total_chunks == 3
    assert stats_before.total_documents == 2

    deleted_count = temp_store.delete_document("doc_del")
    assert deleted_count == 2

    stats_after = temp_store.get_collection_stats()
    assert stats_after.total_chunks == 1
    assert stats_after.total_documents == 1


def test_clear_collection(temp_store: ChromaVectorStore):
    """Verify clear_collection wipes all chunks and resets."""
    vec = [0.1, 0.2, 0.3, 0.4]
    c = _make_dummy_chunk("doc1", 0, "Test", "general", embedding=vec)
    temp_store.add_documents([c])
    assert temp_store.get_collection_stats().total_chunks == 1

    temp_store.clear_collection()
    assert temp_store.get_collection_stats().total_chunks == 0


def test_missing_embeddings_raises_error(temp_store: ChromaVectorStore):
    """Verify error when attempting to add chunks without embeddings."""
    chunk_no_embed = DocumentChunk(
        chunk_id="no_emb",
        text="Sample text",
        metadata=ChunkMetadata(
            document_id="doc_x",
            source="x.txt",
            life_stage="general",
            chunk_index=0,
        ),
        embedding=None,
    )
    with pytest.raises(VectorStoreError) as exc_info:
        temp_store.add_documents([chunk_no_embed])
    assert "lack embeddings" in str(exc_info.value)
