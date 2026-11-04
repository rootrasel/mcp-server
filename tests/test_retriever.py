"""Integration and unit tests for WellnessRetriever and RAGPipeline."""

from female_wellness.rag.pipeline import RAGPipeline
from female_wellness.rag.retriever import WellnessRetriever


def test_retriever_live_query():
    """Verify retriever retrieves relevant chunks for female cycle inquiry from ChromaDB."""
    retriever = WellnessRetriever()
    results = retriever.retrieve(
        query="How does basal body temperature change after ovulation?",
        life_stage="cycle_fertility",
        top_k=3,
        min_similarity=0.35,
    )

    assert len(results) > 0
    top = results[0]
    assert top.metadata.life_stage in ("cycle_fertility", "general")
    assert "temperature" in top.text.lower() or "progesterone" in top.text.lower()
    assert top.similarity_score >= 0.35


def test_retriever_life_stage_isolation():
    """Verify that querying a specific life stage does not return chunks from an unrelated life stage."""
    retriever = WellnessRetriever()
    results = retriever.retrieve(
        query="Managing hot flashes and night sweats with thermoregulation",
        life_stage="perimenopause",
        top_k=5,
        min_similarity=0.30,
        include_general=False,
    )

    assert len(results) > 0
    for r in results:
        assert r.metadata.life_stage == "perimenopause"
        assert r.metadata.life_stage != "athlete_performance"
        assert r.metadata.life_stage != "pregnancy_postpartum"


def test_retriever_empty_for_unrelated_query_high_threshold():
    """Verify retriever cleanly returns empty list when query does not meet high similarity threshold."""
    retriever = WellnessRetriever()
    results = retriever.retrieve(
        query="Quantum gravity string theory supersymmetry manifolds",
        life_stage="beauty_radiance",
        min_similarity=0.85,
    )
    assert len(results) == 0


def test_rag_pipeline_retrieve_context():
    """Verify RAGPipeline.retrieve_context builds structured markdown and citations."""
    pipeline = RAGPipeline()
    context = pipeline.retrieve_context(
        query="Nutritional support for skin barrier and hydration",
        life_stage="beauty_radiance",
        top_k=2,
    )

    assert context.has_sufficient_evidence is True
    assert len(context.chunks) >= 1
    assert len(context.citations) >= 1
    assert "Evidence [1]" in context.formatted_context
    assert "skin_barrier_hydration_guide.md" in context.formatted_context

    citation = context.citations[0]
    assert citation.source == "skin_barrier_hydration_guide.md"
    assert citation.similarity_score > 0.40
    assert len(citation.excerpt) > 10


def test_pipeline_dry_run_ingestion(tmp_path):
    """Verify dry-run ingestion reports planned chunks without touching the database."""
    test_file = tmp_path / "test_doc.md"
    test_file.write_text("# Test Title\n\nThis is a sample wellness test paragraph.", encoding="utf-8")

    pipeline = RAGPipeline()
    summary = pipeline.ingest_directory(tmp_path, dry_run=True)

    assert summary["scanned_files"] == 1
    assert summary["successful_files"] == 1
    assert summary["dry_run"] is True
    assert summary["total_chunks"] >= 1
