"""Unit tests for the Sentence Transformers local embedding engine."""

import math

from female_wellness.rag.embeddings import EmbeddingEngine


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two unit-normalized vectors."""
    return sum(a * b for a, b in zip(vec1, vec2, strict=True))


def test_embedding_dimension():
    """Verify embedding engine reports 384 dimensions for bge-small-en-v1.5."""
    engine = EmbeddingEngine()
    assert engine.dimension == 384


def test_embed_single_text():
    """Verify single text produces a vector of length 384 with float values."""
    engine = EmbeddingEngine()
    vec = engine.embed_text("Hormonal balance during the luteal phase.")

    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


def test_vector_normalization():
    """Verify embeddings are unit-normalized (L2 norm is approximately 1.0)."""
    engine = EmbeddingEngine()
    vec = engine.embed_text("Skin barrier repair with ceramides and hydration.")
    norm = math.sqrt(sum(x * x for x in vec))

    assert math.isclose(norm, 1.0, rel_tol=1e-3)


def test_batch_embedding():
    """Verify batch encoding returns correct count and shapes."""
    engine = EmbeddingEngine()
    texts = [
        "First trimester morning sickness education.",
        "Pelvic floor recovery exercises in postpartum.",
        "Vasomotor symptoms during perimenopause.",
    ]
    vectors = engine.embed_documents(texts, batch_size=2)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384


def test_empty_text_handling():
    """Verify empty text returns zero vector without crashing."""
    engine = EmbeddingEngine()
    vec = engine.embed_text("   ")

    assert len(vec) == 384
    assert all(x == 0.0 for x in vec)


def test_semantic_similarity_ranking():
    """Verify semantically aligned queries yield significantly higher similarity than unrelated queries."""
    engine = EmbeddingEngine()
    base_vec = engine.embed_text("Ovulation tracking and basal body temperature monitoring.")
    related_vec = engine.embed_text("Fertile window awareness and luteinizing hormone surge.")
    unrelated_vec = engine.embed_text("Compiler optimization in Linux kernel device drivers.")

    sim_related = _cosine_similarity(base_vec, related_vec)
    sim_unrelated = _cosine_similarity(base_vec, unrelated_vec)

    assert sim_related > sim_unrelated
    assert sim_related > 0.60

