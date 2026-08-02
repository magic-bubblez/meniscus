"""Local sentence-transformers embedding provider (bge-base, 768d)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from meniscus.config import EMBEDDING_DIMENSIONS
from meniscus.embedding_interface import EmbeddingInterface
from meniscus.exceptions import EmbeddingDimensionMismatchError, ModelUnavailableError

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

_LOCAL_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_LOCAL_EMBEDDING_DIM = 768
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _model_is_cached() -> bool:
    folder = "models--" + _LOCAL_MODEL_NAME.replace("/", "--")
    hf_home = os.environ.get("HF_HOME")
    roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        f"{hf_home}/hub" if hf_home else None,
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    ]
    return any(root and (Path(root) / folder).exists() for root in roots)


# Cached model -> load offline so startup never blocks on a Hugging Face network call.
if _model_is_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class LocalEmbeddingProvider(EmbeddingInterface):
    """Local bge-base-en-v1.5 embedding implementation."""

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

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([_QUERY_PREFIX + text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            self._load_model()
        try:
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
