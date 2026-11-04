"""Local embedding generation engine using Sentence Transformers.

Implements BAAI/bge-small-en-v1.5 dense vector representations for both
query retrieval and document indexing.
"""

from typing import Any

from sentence_transformers import SentenceTransformer

from female_wellness.core.constants import DEFAULT_EMBEDDING_MODEL
from female_wellness.core.exceptions import FemaleWellnessError
from female_wellness.core.logging_config import get_logger

logger = get_logger("rag.embeddings")


class EmbeddingEngine:
    """Manages local dense embedding generation with Sentence Transformers.

    Uses lazy initialization to avoid slowing down server boot or wasting memory
    until vector generation is actually invoked.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Initialize the embedding engine.

        Args:
            model_name: Hugging Face model identifier or local checkpoint path.
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for the SentenceTransformer model instance."""
        if self._model is None:
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            try:
                # Try loading from local HuggingFace cache first for instant offline startup
                try:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(self.model_name)
                # Compute and cache dimension
                dummy = self._model.encode("dim test", normalize_embeddings=True)
                self._dimension = len(dummy)
                logger.info(
                    "SentenceTransformer model '%s' loaded successfully (dim=%d).",
                    self.model_name,
                    self._dimension,
                )
            except Exception as exc:
                raise FemaleWellnessError(
                    f"Failed to load embedding model '{self.model_name}': {exc}",
                    details={"model_name": self.model_name, "error": str(exc)},
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        """Vector dimensionality of the loaded embedding model (384 for bge-small-en-v1.5)."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        return self._dimension or 384

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized dense vector embedding for a single text or query string.

        Args:
            text: Input string to embed.

        Returns:
            List of floats representing the unit-normalized dense vector.
        """
        cleaned = text.strip()
        if not cleaned:
            # Return zero vector of model dimension if text is empty
            return [0.0] * self.dimension

        embedding: Any = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(x) for x in embedding.tolist()]

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate normalized dense vector embeddings for a batch of document chunks.

        Args:
            texts: List of document strings to embed.
            batch_size: Batch size for model inference.

        Returns:
            List of embedding vectors, one for each input text.
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() if t.strip() else " " for t in texts]

        embeddings: Any = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row.tolist()] for row in embeddings]

