#!/usr/bin/env python3
"""CLI script to build the pre-computed entity embedding database.

Usage
-----
    # Ingest all entities from the curated list
    python ingest_entities.py

    # Ingest only the first 50 entities (useful for testing)
    python ingest_entities.py --limit 50

    # Force re-ingestion of already-processed entities
    python ingest_entities.py --force

    # Ingest only persons or organizations
    python ingest_entities.py --type person
    python ingest_entities.py --type organization

    # Show database stats
    python ingest_entities.py --stats

    # Resume from a specific entity (skip entities before it)
    python ingest_entities.py --resume "Lionel Messi"

Pipeline per entity
-------------------
1. Fetch Wikipedia text via ``wikipedia_service``
2. Extract identity signals via ``embedding_service.extract_identity_signals``
3. Embed all axis texts via ``embedding_service._embed_batch``
4. Store profile + embeddings in SQLite via ``entity_db``
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path so imports work when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from services.entity_db import (
    init_db,
    upsert_entity,
    store_embeddings_batch,
    entity_count,
    has_entity,
    lookup_entity,
)
from services.embedding_service import (
    extract_identity_signals,
    _embed_batch,
    _embed_text,
)
from services.wikipedia_service import (
    get_entity_text,
    get_entity_text_by_title,
    DisambiguationError,
)
from services.baseline_service import ALL_AXES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CURATED_PATH = os.path.join(os.path.dirname(__file__), "data", "curated_entities.json")


def load_curated_entities(entity_type: str | None = None) -> list[tuple[str, str]]:
    """Load the curated entity list. Returns list of (name, type) tuples."""
    with open(CURATED_PATH, "r") as f:
        data = json.load(f)

    entities = []
    if entity_type is None or entity_type == "person":
        for name in data["entities"]["persons"]:
            entities.append((name, "person"))
    if entity_type is None or entity_type == "organization":
        for name in data["entities"]["organizations"]:
            entities.append((name, "organization"))
    return entities


def _axis_text(profile: dict, axis: str) -> str:
    """Extract the text value for an axis from an identity profile."""
    val = profile.get(axis, "")
    if isinstance(val, list):
        val = ", ".join(val)
    return val.strip()


def ingest_entity(name: str, entity_type: str, force: bool = False) -> bool:
    """Ingest a single entity. Returns True on success, False on skip/error."""
    if not force and has_entity(name):
        log.info("SKIP %s (already in database)", name)
        return True

    expected_type = entity_type  # "person" or "organization"

    # Step 1: Fetch Wikipedia text
    try:
        wiki_data = get_entity_text(name, expected_type=expected_type)
    except DisambiguationError as exc:
        # Pick the first candidate that looks reasonable
        if exc.candidates:
            first_title = exc.candidates[0][0]
            log.warning("DISAMBIG %s -> picking first candidate: %s", name, first_title)
            try:
                wiki_data = get_entity_text_by_title(first_title, expected_type=expected_type)
            except Exception as inner_exc:
                log.error("FAIL %s (disambiguation fallback): %s", name, inner_exc)
                return False
        else:
            log.error("FAIL %s (disambiguation with no candidates)", name)
            return False
    except Exception as exc:
        log.error("FAIL %s (Wikipedia): %s", name, exc)
        return False

    wiki_text = wiki_data.get("text", "")
    if not wiki_text.strip():
        log.error("FAIL %s (empty Wikipedia text)", name)
        return False

    # Step 2: Extract identity signals
    try:
        profile = extract_identity_signals(wiki_text)
    except Exception as exc:
        log.error("FAIL %s (identity extraction): %s", name, exc)
        return False

    # Step 3: Collect and batch-embed all axis texts
    axis_texts = {}
    for axis in ALL_AXES:
        text = _axis_text(profile, axis)
        if text:
            axis_texts[axis] = text

    if not axis_texts:
        log.error("FAIL %s (no axis texts extracted)", name)
        return False

    unique_texts = list(set(axis_texts.values()))
    try:
        embeddings_list = _embed_batch(unique_texts)
    except Exception as exc:
        log.error("FAIL %s (embedding): %s", name, exc)
        return False

    text_to_emb = dict(zip(unique_texts, embeddings_list))

    # Step 4: Store in database
    entity_id = upsert_entity(name, entity_type, profile)

    axis_data = []
    for axis, text in axis_texts.items():
        emb = text_to_emb.get(text)
        if emb is not None:
            axis_data.append((axis, text, emb))

    store_embeddings_batch(entity_id, axis_data)
    log.info("OK %s (%d axes embedded)", name, len(axis_data))
    return True


def show_stats() -> None:
    """Print database statistics."""
    count = entity_count()
    print(f"\nDatabase: {os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'entities.db'))}")
    print(f"Total entities: {count}")

    if count > 0:
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), "data", "entities.db")
        conn = sqlite3.connect(db_path)
        persons = conn.execute("SELECT COUNT(*) FROM entities WHERE entity_type='person'").fetchone()[0]
        orgs = conn.execute("SELECT COUNT(*) FROM entities WHERE entity_type='organization'").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        avg_axes = emb_count / count if count > 0 else 0
        conn.close()
        print(f"  Persons: {persons}")
        print(f"  Organizations: {orgs}")
        print(f"  Total embeddings: {emb_count}")
        print(f"  Avg axes/entity: {avg_axes:.1f}")

        # Database file size
        db_size = os.path.getsize(db_path)
        if db_size > 1_000_000:
            print(f"  Database size: {db_size / 1_000_000:.1f} MB")
        else:
            print(f"  Database size: {db_size / 1_000:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Build the pre-computed entity embedding database.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit the number of entities to ingest (0 = all).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest entities that are already in the database.",
    )
    parser.add_argument(
        "--type", choices=["person", "organization"], default=None,
        help="Only ingest entities of this type.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume ingestion starting from this entity name (skips earlier entries).",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show database statistics and exit.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay in seconds between entities (rate limiting). Default: 0.5s.",
    )
    args = parser.parse_args()

    # Initialise the database tables.
    init_db()

    if args.stats:
        show_stats()
        return

    entities = load_curated_entities(entity_type=args.type)

    # Handle --resume: skip entities until we find the resume target.
    if args.resume:
        resume_lower = args.resume.lower()
        found = False
        for i, (name, _) in enumerate(entities):
            if name.lower() == resume_lower:
                entities = entities[i:]
                found = True
                break
        if not found:
            log.error("Resume entity '%s' not found in curated list.", args.resume)
            sys.exit(1)
        log.info("Resuming from '%s' (%d entities remaining).", args.resume, len(entities))

    if args.limit > 0:
        entities = entities[:args.limit]

    total = len(entities)
    log.info("Starting ingestion of %d entities...", total)

    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for i, (name, entity_type) in enumerate(entities, 1):
        log.info("[%d/%d] Processing: %s (%s)", i, total, name, entity_type)
        try:
            result = ingest_entity(name, entity_type, force=args.force)
            if result:
                success += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            log.warning("\nInterrupted! Progress saved. Use --resume to continue.")
            log.info("Last attempted: %s", name)
            break
        except Exception as exc:
            log.error("UNEXPECTED ERROR for %s: %s", name, exc)
            failed += 1

        # Rate limiting delay between entities
        if i < total and args.delay > 0:
            time.sleep(args.delay)

    elapsed = time.time() - start_time
    log.info(
        "\nIngestion complete: %d success, %d failed out of %d (%.1fs elapsed)",
        success, failed, total, elapsed,
    )
    show_stats()


if __name__ == "__main__":
    main()
