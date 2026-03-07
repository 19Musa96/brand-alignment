"""Embedding and contextual relatedness scoring.

NOTE: Cosine similarity measures contextual RELATEDNESS, not alignment.
Alignment (positive vs. negative) is determined by the LLM, not similarity.

Similarity in [-1, 1] is mapped to [0, 100] via ((sim + 1) / 2) * 100 for readability.
"""

import os
from typing import Optional
import json

from google import genai
import numpy as np

_configured: bool = False
_client: Optional[genai.Client] = None

# preload once
with open("./utils/prompts/identity_signals.txt", "r") as f:
    IDENTITY_SIGNAL_TEMPLATE = f.read()

with open("./utils/schema.json", "r") as f:
    JSON_SCHEMA = f.read()

with open("./utils/domain_relatedness_config.json", "r") as f:
    DOMAIN_RELATEDNESS_CONFIG = json.load(f)

with open("./utils/value_alignment_config.json", "r") as f:
    VALUE_ALIGNMENT_CONFIG = json.load(f)

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
    if cosine_sim < 0:
        normalized = ((cosine_sim + 1.0) / 2.0) * 100.0
    else:
        normalized = cosine_sim*100.0

    if normalized < 0:
        normalized = 0.0
    if normalized > 100:
        normalized = 100.0
    return normalized

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

def compute_entity_relatedness(entity_a: dict, entity_b: dict, config: dict) -> float:
    """
    Compute a weightedscore between two entities based on the config (0-100).
    
    entity - dict with keys corresponding to the config axes (e.g. primary_domain, sub_domains, etc.)
    config - dict with keys corresponding to the entity axes and values as weights (e.g. "primary_domain": 0.4, "sub_domains": 0.2, etc.)
    """
    
    alignment_scores_tuples = {}
    alignment_scores = {}

    for k,weight in config.items():
        print(f"Computing relatedness for axis: {k}")
        entity_a_value = entity_a.get(k, "")
        entity_b_value = entity_b.get(k, "")

        entity_a_value = ",".join(entity_a_value) if isinstance(entity_a_value, list) else entity_a_value
        entity_b_value = ",".join(entity_b_value) if isinstance(entity_b_value, list) else entity_b_value

        print(f"Entity A value for {k}: {entity_a_value}")
        print(f"Entity B value for {k}: {entity_b_value}")

        if not entity_a_value == "" and not entity_b_value == "":
            score = compute_alignment_score(entity_a_value, entity_b_value)
            alignment_scores_tuples[k] = (weight,score)
            alignment_scores[k] = score
        
    
    total_weight = sum(w for w, _ in alignment_scores_tuples.values())

    print(alignment_scores)
    
    weighted_score = sum((w / total_weight) * score for w, score in alignment_scores_tuples.values())

    return weighted_score,alignment_scores

def compute_final_alignment_score(text_a: str, text_b: str) -> float:
    """
    Compute a final alignment score by combining domain and value scores with weights.
    """
    entity_a = extract_identity_signals(text_a)
    entity_b = extract_identity_signals(text_b)

    domain_score,domain_score_dict = compute_entity_relatedness(
        entity_a=entity_a,
        entity_b=entity_b,
        config=DOMAIN_RELATEDNESS_CONFIG
    )

    value_score,value_score_dict = compute_entity_relatedness(
        entity_a=entity_a,
        entity_b=entity_b,
        config=VALUE_ALIGNMENT_CONFIG
    )
    
    return domain_score, value_score,domain_score_dict,value_score_dict

def parse_response(text):
    response = text.replace("```json", "").replace("```", "").strip()
    return response

def relationship_label(domain, alignment):

    if domain > 70 and alignment > 70:
        return "Strong Strategic Alignment"

    if domain > 70 and alignment < 40:
        return "Competitive or Adversarial Relationship"

    if domain < 40 and alignment > 70:
        return "Shared Values but Different Domains"

    return "Low Strategic Connection"