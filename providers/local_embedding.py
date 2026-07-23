"""Local sentence-transformers embedding provider."""

from __future__ import annotations

from config import EMBEDDING_DIMENSIONS
from embedding_interface import EmbeddingInterface
from exceptions import EmbeddingDimensionMismatchError, ModelUnavailableError

_LOCAL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_LOCAL_EMBEDDING_DIM = 384


class LocalEmbeddingProvider(EmbeddingInterface):
    """Local bge-small-en-v1.5 embedding implementation."""

    def __init__(self) -> None:
        self._model = None

    @property
    def dimension(self) -> int:
        return _LOCAL_EMBEDDING_DIM

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(_LOCAL_MODEL_NAME)
        except ImportError as exc:
            raise ModelUnavailableError(
                "sentence-transformers package not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise ModelUnavailableError(
                f"Failed to load local embedding model {_LOCAL_MODEL_NAME}: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            self._load_model()
        try:
            # sentence-transformers encodes a list natively (fast, vectorized).
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            vectors = [row.tolist() for row in embeddings]
        except Exception as exc:
            raise ModelUnavailableError(f"Local embedding model failed: {exc}") from exc

        for vector in vectors:
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise EmbeddingDimensionMismatchError(
                    f"Local model {_LOCAL_MODEL_NAME} returned a {len(vector)}-"
                    f"dimensional vector, expected {EMBEDDING_DIMENSIONS}."
                )
        return vectors
