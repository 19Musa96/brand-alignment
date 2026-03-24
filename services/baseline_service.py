"""Contrastive baseline scoring service.

Pre-computes pairwise embedding similarities from a diverse corpus of
unrelated entities.  The median similarity per axis is used as a
"random-pair baseline" that is subtracted from raw similarity values
*before* the existing non-linear normalization, improving discrimination
between genuinely related and unrelated entity pairs.

Baseline values are cached to ``utils/baseline_cache.json`` so the
expensive embedding calls happen only once (or when the corpus changes).
"""

import json
import os
import hashlib
from itertools import combinations
from typing import Optional

import numpy as np

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "utils", "baseline_corpus.json")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "utils", "baseline_cache.json")

# All axes used across both domain-relatedness and value-alignment configs.
ALL_AXES = [
    "primary_domain", "sub_domains", "economic_model", "target_audience",
    "core_mission", "value_signals", "cultural_positioning",
    "power_positioning", "controversy_themes",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _corpus_hash(corpus: list[dict]) -> str:
    """Deterministic hash of the corpus so we can detect changes."""
    raw = json.dumps(corpus, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_corpus() -> list[dict]:
    with open(CORPUS_PATH, "r") as f:
        data = json.load(f)
    return data["entities"]


def _load_cache() -> Optional[dict]:
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def _save_cache(payload: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_baseline(force: bool = False) -> dict[str, float]:
    """Return per-axis median cosine similarity across random entity pairs.

    Results are cached in ``utils/baseline_cache.json``.  Pass
    ``force=True`` to recompute even if the cache exists.

    Returns a dict like::

        {"primary_domain": 0.42, "core_mission": 0.38, ...}
    """
    # Lazy import to avoid circular dependency at module level.
    from services.embedding_service import _embed_text

    corpus = _load_corpus()
    corpus_h = _corpus_hash(corpus)

    # Try cache first.
    if not force:
        cache = _load_cache()
        if cache and cache.get("corpus_hash") == corpus_h:
            return cache["baselines"]

    print("[baseline] Computing contrastive baselines from corpus …")

    # Embed every axis value for every entity.
    # Structure: {axis: {entity_index: np.ndarray}}
    embeddings: dict[str, dict[int, np.ndarray]] = {ax: {} for ax in ALL_AXES}

    for idx, entity in enumerate(corpus):
        for axis in ALL_AXES:
            value = entity.get(axis, "")
            if isinstance(value, list):
                value = ", ".join(value)
            if not value.strip():
                continue
            emb = _embed_text(value.strip())
            embeddings[axis][idx] = emb

    # Compute pairwise cosine similarities per axis and take the median.
    baselines: dict[str, float] = {}
    for axis in ALL_AXES:
        axis_embs = embeddings[axis]
        indices = list(axis_embs.keys())
        if len(indices) < 2:
            baselines[axis] = 0.0
            continue
        sims = [
            _cosine_similarity(axis_embs[i], axis_embs[j])
            for i, j in combinations(indices, 2)
        ]
        baselines[axis] = float(np.median(sims))

    # Persist.
    _save_cache({
        "corpus_hash": corpus_h,
        "baselines": baselines,
    })
    print(f"[baseline] Done. Baselines: {baselines}")
    return baselines


def get_baseline(axis: str) -> float:
    """Return the cached baseline for a single axis.

    Triggers computation if no cache exists yet.
    """
    baselines = compute_baseline()
    return baselines.get(axis, 0.0)


def get_all_baselines() -> dict[str, float]:
    """Return all cached per-axis baselines (computing if necessary)."""
    return compute_baseline()
