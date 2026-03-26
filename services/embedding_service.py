"""Embedding and contextual relatedness scoring.

NOTE: Cosine similarity measures contextual RELATEDNESS, not alignment.
Alignment (positive vs. negative) is determined by the LLM, not similarity.

Similarity in [-1, 1] is mapped to [0, 100] via ((sim + 1) / 2) * 100 for readability.
"""

import math
import os
import re
import threading
from typing import Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
import numpy as np

from services.retry import gemini_retry

_configured: bool = False
_client: Optional[genai.Client] = None
_client_lock = threading.Lock()
_embed_cache: dict[str, np.ndarray] = {}
_embed_cache_lock = threading.Lock()

# preload once
with open("./utils/prompts/identity_signals.txt", "r") as f:
    IDENTITY_SIGNAL_TEMPLATE = f.read()

with open("./utils/schema.json", "r") as f:
    JSON_SCHEMA = f.read()

with open("./utils/domain_relatedness_config.json", "r") as f:
    _domain_raw = json.load(f)
    DOMAIN_RELATEDNESS_WEIGHTS = _domain_raw["weights"]
    DOMAIN_RELATEDNESS_NORMALIZATION = _domain_raw.get("normalization", {})
    DOMAIN_LLM_JUDGE_CONFIG = _domain_raw.get("llm_judge", {})
    DOMAIN_BASELINE_CONFIG = _domain_raw.get("baseline", {})

with open("./utils/value_alignment_config.json", "r") as f:
    _value_raw = json.load(f)
    VALUE_ALIGNMENT_WEIGHTS = _value_raw["weights"]
    VALUE_ALIGNMENT_NORMALIZATION = _value_raw.get("normalization", {})
    VALUE_LLM_JUDGE_CONFIG = _value_raw.get("llm_judge", {})
    VALUE_BASELINE_CONFIG = _value_raw.get("baseline", {})

def _configure_client() -> None:
    """Configure the client once using the API key from the environment. Thread-safe."""
    global _configured, _client
    if _configured:
        return
    with _client_lock:
        if _configured:
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required for embeddings.")
        _client = genai.Client(api_key=api_key)
        _configured = True

def load_model(model_name: str) -> str:
    """Ensure the client is configured and return the embedding model name."""
    _configure_client()
    return model_name

def normalize_score(raw_score: float, normalization_config: dict) -> float:
    """Apply non-linear normalization to spread scores across the full 0-100 range.

    Supports two methods:
    - "sigmoid": Logistic S-curve centered at `midpoint` with configurable `steepness`.
      Penalizes weak similarities more aggressively and spreads scores that cluster
      in the 50-75 band.
    - "power": Power-curve transformation (score/100)^exponent * 100.
      Exponent > 1 pushes mediocre scores lower; exponent < 1 boosts them.

    If no valid method is configured, returns the raw score unchanged.
    """
    method = normalization_config.get("method", "none")

    if method == "sigmoid":
        midpoint = normalization_config.get("midpoint", 50.0)
        steepness = normalization_config.get("steepness", 0.1)
        # Sigmoid: 1 / (1 + e^(-k*(x - midpoint)))
        # Rescaled so that 0 maps to ~0 and 100 maps to ~100
        raw_sigmoid = 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))
        sig_at_0 = 1.0 / (1.0 + math.exp(-steepness * (0.0 - midpoint)))
        sig_at_100 = 1.0 / (1.0 + math.exp(-steepness * (100.0 - midpoint)))
        normalized = (raw_sigmoid - sig_at_0) / (sig_at_100 - sig_at_0) * 100.0
        return max(0.0, min(100.0, normalized))

    if method == "power":
        exponent = normalization_config.get("exponent", 1.5)
        normalized = ((raw_score / 100.0) ** exponent) * 100.0
        return max(0.0, min(100.0, normalized))

    # No normalization
    return raw_score


@gemini_retry
def _embed_text_uncached(text: str) -> np.ndarray:

    model_name = load_model("gemini-embedding-001")

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


def _embed_text(text: str) -> np.ndarray:
    """Cached wrapper around embedding API. Thread-safe."""
    with _embed_cache_lock:
        if text in _embed_cache:
            return _embed_cache[text]
    result = _embed_text_uncached(text)
    with _embed_cache_lock:
        _embed_cache[text] = result
    return result


@gemini_retry
def _embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed multiple texts in a single API call. Thread-safe with cache."""
    _configure_client()
    if _client is None:
        raise RuntimeError("Gemini client not configured for embeddings.")

    # Separate cached from uncached texts
    uncached_texts = []
    uncached_indices = []
    results = [None] * len(texts)

    with _embed_cache_lock:
        for i, text in enumerate(texts):
            if text in _embed_cache:
                results[i] = _embed_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

    if uncached_texts:
        response = _client.models.embed_content(
            model="gemini-embedding-001", contents=uncached_texts
        )
        embeddings = getattr(response, "embeddings", None)
        if not embeddings or len(embeddings) != len(uncached_texts):
            raise RuntimeError("Gemini batch embedding response was empty or incomplete.")
        with _embed_cache_lock:
            for j, (result_idx, emb) in enumerate(zip(uncached_indices, embeddings)):
                values = getattr(emb, "values", None)
                if values is None:
                    raise RuntimeError("Gemini embedding response did not include values.")
                arr = np.asarray(values, dtype=float)
                results[result_idx] = arr
                _embed_cache[uncached_texts[j]] = arr

    return results

def compute_alignment_score(text_a: str, text_b: str,
                            baseline_similarity: float = 0.0) -> float:
    """
    Compute a contextual relatedness score between two texts (0-100).

    NOTE: This is RELATEDNESS (similarity), not alignment direction.
    High relatedness just means the texts are contextually close.
    The LLM determines whether that relatedness is positive (✓) or negative (✗).

    When ``baseline_similarity`` is provided (from the contrastive baseline
    corpus), it is subtracted from the raw cosine similarity *before* mapping
    to [0, 100].  This re-centres the scale so that a "random unrelated pair"
    similarity maps close to 0, improving discrimination between genuinely
    related and unrelated entities.

    The adjusted similarity is linearly mapped from its effective range to
    [0, 100] and clamped.
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

    # Subtract contrastive baseline so unrelated pairs centre near zero.
    adjusted_sim = cosine_sim - baseline_similarity

    # Map adjusted similarity to [0, 100] with the baseline at 50.
    # Above baseline (adjusted > 0): map [0, 1 - baseline] → [50, 100]
    # Below baseline (adjusted < 0): map [-(1 + baseline), 0] → [0, 50]
    if adjusted_sim >= 0:
        max_above = 1.0 - baseline_similarity
        normalized = 50.0 + (adjusted_sim / max_above) * 50.0 if max_above > 0 else 50.0
    else:
        max_below = 1.0 + baseline_similarity
        normalized = 50.0 + (adjusted_sim / max_below) * 50.0 if max_below > 0 else 50.0

    return max(0.0, min(100.0, normalized))

@gemini_retry
def extract_identity_signals(text: str) -> dict:
    """
    Extract identity signals from the text using the gemini model.
    """
    # Truncate to reduce LLM processing time while retaining key identity signals
    MAX_CHARS = 3000
    truncated_text = text[:MAX_CHARS]

    identity_signal_prompt = IDENTITY_SIGNAL_TEMPLATE.format(wikipedia_text=truncated_text,
                                                           schema_json=JSON_SCHEMA)
    
    model_name = load_model("gemini-2.5-flash-lite")

    response = _client.models.generate_content(model=model_name, contents=identity_signal_prompt)
    identity_signal_str = getattr(response, "text", None) or ""
    identity_signal_str = parse_response(identity_signal_str)
    # print(100*'*','\n',identity_signal_str)
    identity_signal_json = json.loads(identity_signal_str)
    # For demonstration, we'll return an empty dict.
    # print(identity_signal_json,type(identity_signal_json))

    return identity_signal_json

@gemini_retry
def llm_judge_axis_score(axis_name: str, value_a: str, value_b: str) -> float:
    """Ask Gemini to rate alignment between two entity values on a 0-10 scale.

    Returns a score in the 0-100 range (LLM's 0-10 rating multiplied by 10).
    """
    _configure_client()
    if _client is None:
        raise RuntimeError("Gemini client not configured.")

    prompt = (
        f"You are an expert brand alignment analyst. "
        f"Rate the alignment between two entities on the '{axis_name}' axis from 0 to 10, "
        f"where 0 means completely unrelated/misaligned and 10 means perfectly aligned.\n\n"
        f"Entity A ({axis_name}): {value_a}\n"
        f"Entity B ({axis_name}): {value_b}\n\n"
        f"Respond with ONLY a single number from 0 to 10 (integers or one decimal place). "
        f"Do not include any other text."
    )

    model_name = load_model("gemini-2.5-flash-lite")
    response = _client.models.generate_content(model=model_name, contents=prompt)
    response_text = (getattr(response, "text", None) or "").strip()

    try:
        rating = float(response_text)
        rating = max(0.0, min(10.0, rating))
    except ValueError:
        # Fallback: try to extract a number from the response
        match = re.search(r"(\d+\.?\d*)", response_text)
        if match:
            rating = float(match.group(1))
            rating = max(0.0, min(10.0, rating))
        else:
            # If LLM fails to produce a number, return neutral score
            rating = 5.0

    return rating * 10.0  # Convert 0-10 to 0-100


def _cosine_sim(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def _sim_to_score(cosine_sim: float, baseline_similarity: float = 0.0) -> float:
    """Map cosine similarity to 0-100 score with baseline subtraction."""
    adjusted_sim = cosine_sim - baseline_similarity
    if adjusted_sim >= 0:
        max_above = 1.0 - baseline_similarity
        normalized = 50.0 + (adjusted_sim / max_above) * 50.0 if max_above > 0 else 50.0
    else:
        max_below = 1.0 + baseline_similarity
        normalized = 50.0 + (adjusted_sim / max_below) * 50.0 if max_below > 0 else 50.0
    return max(0.0, min(100.0, normalized))


def compute_entity_relatedness(entity_a: dict, entity_b: dict, weights: dict,
                                normalization_config: dict | None = None,
                                llm_judge_config: dict | None = None,
                                baseline_config: dict | None = None) -> float:
    """
    Compute a weighted score between two entities based on axis weights (0-100).

    Uses batch embedding to minimize API roundtrips and concurrent LLM judge calls.
    """

    alignment_scores_tuples = {}
    alignment_scores = {}

    default_llm_weight = (llm_judge_config or {}).get("default_weight", 0.0)
    per_axis_llm_weights = (llm_judge_config or {}).get("axis_weights", {})

    # Load contrastive baselines if enabled.
    baselines: dict[str, float] = {}
    if baseline_config and baseline_config.get("enabled", False):
        from services.baseline_service import get_all_baselines
        baselines = get_all_baselines()
        print(f"[contrastive-baseline] Using per-axis baselines: {baselines}")

    # Collect all axis text pairs and prepare for batch embedding
    axis_pairs: list[tuple[str, float, str, str]] = []  # (axis_name, weight, val_a, val_b)
    all_texts: list[str] = []
    text_to_idx: dict[str, int] = {}

    for k, weight in weights.items():
        val_a = entity_a.get(k, "")
        val_b = entity_b.get(k, "")
        val_a = ",".join(val_a) if isinstance(val_a, list) else val_a
        val_b = ",".join(val_b) if isinstance(val_b, list) else val_b

        print(f"Entity A value for {k}: {val_a}")
        print(f"Entity B value for {k}: {val_b}")

        if val_a == "" or val_b == "":
            continue

        val_a = val_a.strip()
        val_b = val_b.strip()
        axis_pairs.append((k, weight, val_a, val_b))

        # De-duplicate texts for batch embedding
        for t in (val_a, val_b):
            if t not in text_to_idx:
                text_to_idx[t] = len(all_texts)
                all_texts.append(t)

    if not axis_pairs:
        return 0.0, {}

    # Batch embed all unique texts in one API call
    embeddings = _embed_batch(all_texts)

    # Identify which axes need LLM judge scores
    llm_judge_tasks = []
    for k, weight, val_a, val_b in axis_pairs:
        llm_weight = per_axis_llm_weights.get(k, default_llm_weight)
        if llm_weight > 0:
            llm_judge_tasks.append((k, val_a, val_b))

    # Fire all LLM judge calls concurrently
    llm_scores: dict[str, float] = {}
    if llm_judge_tasks:
        with ThreadPoolExecutor(max_workers=len(llm_judge_tasks)) as executor:
            futures = {
                executor.submit(llm_judge_axis_score, k, va, vb): k
                for k, va, vb in llm_judge_tasks
            }
            for future in as_completed(futures):
                axis_name = futures[future]
                llm_scores[axis_name] = future.result()

    # Compute final blended scores per axis
    for k, weight, val_a, val_b in axis_pairs:
        emb_a = embeddings[text_to_idx[val_a]]
        emb_b = embeddings[text_to_idx[val_b]]
        cosine_sim = _cosine_sim(emb_a, emb_b)
        axis_baseline = baselines.get(k, 0.0)
        embedding_score = _sim_to_score(cosine_sim, axis_baseline)

        llm_weight = per_axis_llm_weights.get(k, default_llm_weight)
        if llm_weight > 0 and k in llm_scores:
            llm_score = llm_scores[k]
            blended_score = (1.0 - llm_weight) * embedding_score + llm_weight * llm_score
            print(f"Axis {k}: embedding={embedding_score:.1f}, llm_judge={llm_score:.1f}, "
                  f"blend_weight={llm_weight}, blended={blended_score:.1f}")
        else:
            blended_score = embedding_score

        alignment_scores_tuples[k] = (weight, blended_score)
        alignment_scores[k] = blended_score

    total_weight = sum(w for w, _ in alignment_scores_tuples.values())

    print(alignment_scores)

    weighted_score = sum((w / total_weight) * score for w, score in alignment_scores_tuples.values())

    # Apply non-linear normalization to spread scores across full 0-100 range
    if normalization_config:
        weighted_score = normalize_score(weighted_score, normalization_config)

    return weighted_score, alignment_scores

def _prepare_axes(entity_a: dict, entity_b: dict, weights: dict):
    """Prepare axis pairs and collect unique texts for a weight config."""
    axis_pairs = []
    unique_texts = set()
    for k, weight in weights.items():
        val_a = entity_a.get(k, "")
        val_b = entity_b.get(k, "")
        val_a = ",".join(val_a) if isinstance(val_a, list) else val_a
        val_b = ",".join(val_b) if isinstance(val_b, list) else val_b
        if val_a.strip() and val_b.strip():
            val_a, val_b = val_a.strip(), val_b.strip()
            axis_pairs.append((k, weight, val_a, val_b))
            unique_texts.add(val_a)
            unique_texts.add(val_b)
    return axis_pairs, unique_texts


def _try_precomputed(entity_name: str) -> dict | None:
    """Look up pre-computed identity signals and embeddings for an entity.

    If found, loads the profile and pre-populates the in-memory embedding
    cache so subsequent ``_embed_text`` calls hit the cache instantly.

    Returns the identity signals dict or ``None`` if not in the database.
    """
    try:
        from services.entity_db import lookup_entity, init_db
        init_db()
        record = lookup_entity(entity_name)
    except Exception:
        return None

    if record is None:
        return None

    # Pre-populate the in-memory embedding cache with stored vectors.
    with _embed_cache_lock:
        for axis, text_value in record["axis_texts"].items():
            if text_value not in _embed_cache:
                _embed_cache[text_value] = record["embeddings"][axis]

    print(f"[precomputed] Using pre-computed profile for '{entity_name}' "
          f"({len(record['embeddings'])} axes cached)")
    return record["profile"]


def compute_final_alignment_score(text_a: str, text_b: str,
                                  precomputed_a: dict | None = None,
                                  precomputed_b: dict | None = None) -> float:
    """
    Compute a final alignment score by combining domain and value scores with weights.
    Non-linear normalization is applied per the scoring config files.

    When ``precomputed_a`` or ``precomputed_b`` are provided (pre-computed
    identity signal dicts from the entity database), the corresponding
    identity extraction and embedding API calls are skipped entirely.

    Optimized pipeline:
    1. Identity extraction (2 LLM calls, concurrent) — skipped for pre-computed entities
    2. Batch embed ALL axis texts (1 API call) — skipped for pre-computed entities
    3. ALL LLM judge calls across both categories (9 calls, all concurrent)
    4. Compute scores locally (pure math, instant)
    """
    # Step 1: Extract identity signals — use pre-computed profiles when available
    entity_a = precomputed_a
    entity_b = precomputed_b
    need_extract = []
    if entity_a is None:
        need_extract.append(("a", text_a))
    if entity_b is None:
        need_extract.append(("b", text_b))

    if len(need_extract) == 2:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(extract_identity_signals, text_a)
            future_b = executor.submit(extract_identity_signals, text_b)
            entity_a = future_a.result()
            entity_b = future_b.result()
    elif len(need_extract) == 1:
        label, text = need_extract[0]
        result = extract_identity_signals(text)
        if label == "a":
            entity_a = result
        else:
            entity_b = result

    # Step 2: Prepare all axis data and batch embed ALL texts in one API call
    domain_pairs, domain_texts = _prepare_axes(entity_a, entity_b, DOMAIN_RELATEDNESS_WEIGHTS)
    value_pairs, value_texts = _prepare_axes(entity_a, entity_b, VALUE_ALIGNMENT_WEIGHTS)
    all_unique_texts = list(domain_texts | value_texts)

    # Filter out texts already in the embedding cache (pre-computed entities)
    with _embed_cache_lock:
        uncached_texts = [t for t in all_unique_texts if t not in _embed_cache]
    if uncached_texts:
        _embed_batch(uncached_texts)  # Single API call, pre-populates cache

    # Step 3: Fire ALL LLM judge calls across both categories concurrently
    domain_llm_config = DOMAIN_LLM_JUDGE_CONFIG or {}
    value_llm_config = VALUE_LLM_JUDGE_CONFIG or {}
    domain_default_w = domain_llm_config.get("default_weight", 0.0)
    domain_axis_w = domain_llm_config.get("axis_weights", {})
    value_default_w = value_llm_config.get("default_weight", 0.0)
    value_axis_w = value_llm_config.get("axis_weights", {})

    llm_judge_tasks = []
    for k, _w, va, vb in domain_pairs:
        if domain_axis_w.get(k, domain_default_w) > 0:
            llm_judge_tasks.append((k, va, vb))
    for k, _w, va, vb in value_pairs:
        if value_axis_w.get(k, value_default_w) > 0:
            llm_judge_tasks.append((k, va, vb))

    all_llm_scores: dict[str, float] = {}
    if llm_judge_tasks:
        with ThreadPoolExecutor(max_workers=len(llm_judge_tasks)) as executor:
            futures = {
                executor.submit(llm_judge_axis_score, k, va, vb): k
                for k, va, vb in llm_judge_tasks
            }
            for future in as_completed(futures):
                axis_name = futures[future]
                all_llm_scores[axis_name] = future.result()

    # Step 4: Compute scores locally (pure math, no API calls)
    # Load baselines once
    baselines: dict[str, float] = {}
    if (DOMAIN_BASELINE_CONFIG and DOMAIN_BASELINE_CONFIG.get("enabled", False)) or \
       (VALUE_BASELINE_CONFIG and VALUE_BASELINE_CONFIG.get("enabled", False)):
        from services.baseline_service import get_all_baselines
        baselines = get_all_baselines()

    def _score_category(axis_pairs, llm_config, normalization_config):
        default_w = (llm_config or {}).get("default_weight", 0.0)
        axis_w = (llm_config or {}).get("axis_weights", {})
        scores_tuples = {}
        scores_dict = {}
        for k, weight, val_a, val_b in axis_pairs:
            emb_a = _embed_text(val_a)  # Hits cache (instant)
            emb_b = _embed_text(val_b)  # Hits cache (instant)
            sim = _cosine_sim(emb_a, emb_b)
            embedding_score = _sim_to_score(sim, baselines.get(k, 0.0))
            llm_w = axis_w.get(k, default_w)
            if llm_w > 0 and k in all_llm_scores:
                blended = (1.0 - llm_w) * embedding_score + llm_w * all_llm_scores[k]
            else:
                blended = embedding_score
            scores_tuples[k] = (weight, blended)
            scores_dict[k] = blended

        if not scores_tuples:
            return 0.0, {}
        total_w = sum(w for w, _ in scores_tuples.values())
        weighted = sum((w / total_w) * s for w, s in scores_tuples.values())
        if normalization_config:
            weighted = normalize_score(weighted, normalization_config)
        return weighted, scores_dict

    domain_score, domain_score_dict = _score_category(
        domain_pairs, DOMAIN_LLM_JUDGE_CONFIG, DOMAIN_RELATEDNESS_NORMALIZATION)
    value_score, value_score_dict = _score_category(
        value_pairs, VALUE_LLM_JUDGE_CONFIG, VALUE_ALIGNMENT_NORMALIZATION)

    return domain_score, value_score, domain_score_dict, value_score_dict, entity_a, entity_b

def parse_response(text):
    response = text.replace("```json", "").replace("```", "").strip()
    return response