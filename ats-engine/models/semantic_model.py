"""Sentence-transformers wrapper (spec section 16).

Loads `all-MiniLM-L6-v2` once per process, caches it inside the project
directory (config.MODELS_DIR), and never downloads anything elsewhere.
Batched encode + cosine similarity helpers included.
"""

from __future__ import annotations

import numpy as np

from config import MODELS_DIR, SEMANTIC_MODEL_NAME


class SemanticModelUnavailable(RuntimeError):
    """Raised when sentence-transformers or the model cannot be loaded."""


_model = None          # singleton, loaded lazily


def get_model():
    """Load the model once per process; raise SemanticModelUnavailable on failure."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SemanticModelUnavailable(
            "sentence-transformers is not installed. "
            "Install it (pip install sentence-transformers) or run with --no-semantic."
        ) from e
    try:
        _model = SentenceTransformer(
            SEMANTIC_MODEL_NAME,
            cache_folder=str(MODELS_DIR),       # downloads stay inside the repo
        )
    except Exception as e:
        raise SemanticModelUnavailable(
            f"Could not load '{SEMANTIC_MODEL_NAME}' "
            f"(looked in cache dir {MODELS_DIR}): {e}"
        ) from e
    return _model


def encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts (L2-normalized so cosine == dot product)."""
    model = get_model()
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between two embedding batches."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    return np.clip(a @ b.T, -1.0, 1.0)
