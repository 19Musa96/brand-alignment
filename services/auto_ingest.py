"""Auto-ingestion service for entities discovered during analysis.

When a new entity (celebrity/brand) is analyzed but not found in the pre-computed
database, this service automatically ingests it for future lookups.

Stores already-computed identity profiles and embeddings without re-computation,
avoiding duplicate API calls.
"""

import logging
import threading

import numpy as np

from services.entity_db import upsert_entity, store_embeddings_batch, has_entity

log = logging.getLogger(__name__)


def store_analyzed_entity(
    name: str,
    entity_type: str,
    identity_profile: dict,
) -> bool:
    """Store an analyzed entity's identity profile to the database.

    Re-embeds axis texts using the cached embedding function (cache hits only).
    Non-blocking, logs errors gracefully.

    Args:
        name: str - entity name
        entity_type: str - "person" or "organization"
        identity_profile: dict - extracted identity signals (e.g., {"primary_domain": "Tech", ...})

    Returns:
        bool - True if stored successfully, False if skipped or failed
    """
    try:
        if has_entity(name):
            log.info("SKIP %s (already in database)", name)
            return True

        # Import locally to avoid circular dependency
        from services.embedding_service import _embed_text

        # Store the entity profile
        entity_id = upsert_entity(name, entity_type, identity_profile)

        # Re-embed axis texts (will be cache hits since they were computed during analysis)
        axis_data = []
        for axis, text_value in identity_profile.items():
            if isinstance(text_value, list):
                text_value = ", ".join(text_value)
            text_value = str(text_value).strip()

            if text_value:
                try:
                    embedding = _embed_text(text_value)
                    axis_data.append((axis, text_value, embedding))
                except Exception as emb_exc:
                    log.warning("Failed to embed axis %s for %s: %s", axis, name, emb_exc)

        if axis_data:
            store_embeddings_batch(entity_id, axis_data)
            log.info("Auto-ingested entity: %s (%s) with %d axes", name, entity_type, len(axis_data))
        else:
            log.warning("No axes stored for %s (no embeddings succeeded)", name)

        return True

    except Exception as exc:
        log.error("Auto-ingest failed for %s: %s", name, exc)
        return False


def process_pending_ingests(pending_list: list[dict]) -> None:
    """Store pre-computed entity profiles in the database using a background thread.

    Embeddings are re-computed from cached axis texts (cache hits only, no API calls).

    Args:
        pending_list: list of dicts with keys:
            - "name": str - entity name
            - "type": str - "person" or "organization"
            - "identity_profile": dict - extracted identity signals
    """
    def store_in_background():
        for item in pending_list:
            try:
                store_analyzed_entity(
                    item["name"],
                    item["type"],
                    item["identity_profile"],
                )
            except Exception as exc:
                log.error("Unexpected error during background storage of %s: %s", item["name"], exc)

    # Launch as daemon thread so it doesn't block app shutdown
    thread = threading.Thread(target=store_in_background, daemon=True)
    thread.start()
