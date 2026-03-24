"""Embedding and contextual relatedness scoring.

NOTE: Cosine similarity measures contextual RELATEDNESS, not alignment.
Alignment (positive vs. negative) is determined by the LLM, not similarity.

Similarity in [-1, 1] is mapped to [0, 100] via ((sim + 1) / 2) * 100 for readability.
"""

import math
import os
import re
from typing import Optional
import json

from google import genai
import numpy as np

from services.retry import gemini_retry

_configured: bool = False
_client: Optional[genai.Client] = None

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
def _embed_text(text: str) -> np.ndarray:

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
    # Optional: truncate long wikipedia text
    MAX_CHARS = 6000
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


def compute_entity_relatedness(entity_a: dict, entity_b: dict, weights: dict,
                                normalization_config: dict | None = None,
                                llm_judge_config: dict | None = None,
                                baseline_config: dict | None = None) -> float:
    """
    Compute a weighted score between two entities based on axis weights (0-100).

    After computing the weighted average of per-axis cosine similarities,
    a non-linear normalization (sigmoid or power-curve) is applied to spread
    scores across the full 0-100 range and penalize weak similarities.

    When llm_judge_config is provided, each axis score is a weighted blend of
    embedding cosine similarity and an LLM judgment score:
        blended = (1 - llm_weight) * embedding_score + llm_weight * llm_score
    The llm_weight is configurable per axis via llm_judge_config.

    When baseline_config is provided and enabled, per-axis contrastive
    baselines are subtracted from raw cosine similarities before the
    [0, 100] mapping, improving discrimination between related and
    unrelated entity pairs.

    entity - dict with keys corresponding to the config axes
    weights - dict with keys as axes and values as weights
    normalization_config - optional dict with 'method' and tuning parameters
    llm_judge_config - optional dict with per-axis llm_judge_weight (0.0-1.0)
    baseline_config - optional dict with 'enabled' flag for contrastive baseline
    """

    alignment_scores_tuples = {}
    alignment_scores = {}

    default_llm_weight = (llm_judge_config or {}).get("default_weight", 0.0)
    per_axis_weights = (llm_judge_config or {}).get("axis_weights", {})

    # Load contrastive baselines if enabled.
    baselines: dict[str, float] = {}
    if baseline_config and baseline_config.get("enabled", False):
        from services.baseline_service import get_all_baselines
        baselines = get_all_baselines()
        print(f"[contrastive-baseline] Using per-axis baselines: {baselines}")

    for k, weight in weights.items():
        print(f"Computing relatedness for axis: {k}")
        entity_a_value = entity_a.get(k, "")
        entity_b_value = entity_b.get(k, "")

        entity_a_value = ",".join(entity_a_value) if isinstance(entity_a_value, list) else entity_a_value
        entity_b_value = ",".join(entity_b_value) if isinstance(entity_b_value, list) else entity_b_value

        print(f"Entity A value for {k}: {entity_a_value}")
        print(f"Entity B value for {k}: {entity_b_value}")

        if not entity_a_value == "" and not entity_b_value == "":
            axis_baseline = baselines.get(k, 0.0)
            embedding_score = compute_alignment_score(
                entity_a_value, entity_b_value,
                baseline_similarity=axis_baseline,
            )

            # Blend with LLM judge if configured
            llm_weight = per_axis_weights.get(k, default_llm_weight)
            if llm_weight > 0:
                llm_score = llm_judge_axis_score(k, entity_a_value, entity_b_value)
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

def compute_final_alignment_score(text_a: str, text_b: str) -> float:
    """
    Compute a final alignment score by combining domain and value scores with weights.
    Non-linear normalization is applied per the scoring config files.
    """
    entity_a = extract_identity_signals(text_a)
    entity_b = extract_identity_signals(text_b)

    domain_score, domain_score_dict = compute_entity_relatedness(
        entity_a=entity_a,
        entity_b=entity_b,
        weights=DOMAIN_RELATEDNESS_WEIGHTS,
        normalization_config=DOMAIN_RELATEDNESS_NORMALIZATION,
        llm_judge_config=DOMAIN_LLM_JUDGE_CONFIG,
        baseline_config=DOMAIN_BASELINE_CONFIG,
    )

    value_score, value_score_dict = compute_entity_relatedness(
        entity_a=entity_a,
        entity_b=entity_b,
        weights=VALUE_ALIGNMENT_WEIGHTS,
        normalization_config=VALUE_ALIGNMENT_NORMALIZATION,
        llm_judge_config=VALUE_LLM_JUDGE_CONFIG,
        baseline_config=VALUE_BASELINE_CONFIG,
    )

    return domain_score, value_score, domain_score_dict, value_score_dict, entity_a, entity_b

def parse_response(text):
    response = text.replace("```json", "").replace("```", "").strip()
    return response