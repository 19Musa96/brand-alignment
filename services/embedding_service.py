"""Embedding and contextual relatedness scoring.

NOTE: Cosine similarity measures contextual RELATEDNESS, not alignment.
Alignment (positive vs. negative) is determined by the LLM, not similarity.

Similarity in [-1, 1] is mapped to [0, 100] via ((sim + 1) / 2) * 100 for readability.
"""

import os
from typing import Optional

from google import genai
import numpy as np

_MODEL_NAME = "gemini-embedding-001"
_configured: bool = False
_client: Optional[genai.Client] = None


def _configure_client() -> None:
    """Configure the client once using the API key from the environment."""
    global _configured, _client
    if _configured:
        return
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is required for embeddings.")
    _client = genai.Client(api_key=api_key)
    _configured = True


def load_embedding_model() -> str:
    """Ensure the client is configured and return the embedding model name."""
    _configure_client()
    return _MODEL_NAME


def _embed_text(text: str) -> np.ndarray:
    model_name = load_embedding_model()
    if _client is None:
        raise RuntimeError("Gemini client not configured for embeddings.")
    result = _client.models.embed_content(model=model_name, contents=text)
    embeddings = getattr(result, "embeddings", None)
    if not embeddings:
        raise RuntimeError("Gemini embedding response was empty.")
    values = getattr(embeddings[0], "values", None)
    if values is None:
        raise RuntimeError("Gemini embedding response did not include values.")
    return np.asarray(values, dtype=float)


def compute_alignment_score(text_a: str, text_b: str) -> float:
    """
    Compute a contextual relatedness score between two texts (0-100).

    NOTE: This is RELATEDNESS (similarity), not alignment direction.
    High relatedness just means the texts are contextually close.
    The LLM determines whether that relatedness is positive (✓) or negative (✗).

    Cosine similarity in [-1, 1] is mapped to [0, 100] via
    ((sim + 1) / 2) * 100 and clamped.
    """

    if not text_a or not text_a.strip():
        raise ValueError("text_a must be a non-empty string.")
    if not text_b or not text_b.strip():
        raise ValueError("text_b must be a non-empty string.")

    emb_a = _embed_text(text_a.strip())
    emb_b = _embed_text(text_b.strip())

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "Embedding norm is zero; cannot compute cosine similarity.")

    cosine_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    normalized = ((cosine_sim + 1.0) / 2.0) * 100.0
    if normalized < 0:
        normalized = 0.0
    if normalized > 100:
        normalized = 100.0
    return normalized
