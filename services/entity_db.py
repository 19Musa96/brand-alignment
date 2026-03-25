"""Pre-computed entity embedding database.

Stores identity signal profiles and per-axis embeddings for a curated list
of well-known celebrities and brands in a SQLite database.  Enables fast
Top-N similarity search without live Gemini embedding API calls for indexed
entities.

Database schema
---------------
entities
    id          INTEGER PRIMARY KEY
    name        TEXT UNIQUE          -- canonical entity name
    entity_type TEXT                 -- "person" or "organization"
    profile     TEXT                 -- JSON-encoded identity signals dict
    created_at  TEXT                 -- ISO-8601 timestamp

embeddings
    id          INTEGER PRIMARY KEY
    entity_id   INTEGER REFERENCES entities(id)
    axis        TEXT                 -- e.g. "primary_domain"
    text_value  TEXT                 -- the axis text that was embedded
    vector      BLOB                -- numpy float64 array as bytes
    UNIQUE(entity_id, axis)
"""

import json
import os
import sqlite3
import threading
from typing import Optional

import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "entities.db")

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection (thread-safe)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            entity_type TEXT    NOT NULL,
            profile     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            axis        TEXT    NOT NULL,
            text_value  TEXT    NOT NULL,
            vector      BLOB    NOT NULL,
            UNIQUE(entity_id, axis)
        );

        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_embeddings_entity_axis ON embeddings(entity_id, axis);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Write operations (used by the ingestion script)
# ---------------------------------------------------------------------------

def upsert_entity(name: str, entity_type: str, profile: dict) -> int:
    """Insert or update an entity's identity profile. Returns the entity id."""
    conn = _get_conn()
    profile_json = json.dumps(profile, ensure_ascii=False)
    conn.execute(
        """INSERT INTO entities (name, entity_type, profile)
           VALUES (?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
               entity_type = excluded.entity_type,
               profile     = excluded.profile,
               created_at  = datetime('now')""",
        (name, entity_type, profile_json),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM entities WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    return row[0]


def store_embedding(entity_id: int, axis: str, text_value: str, vector: np.ndarray) -> None:
    """Store a single axis embedding for an entity."""
    conn = _get_conn()
    blob = vector.astype(np.float64).tobytes()
    conn.execute(
        """INSERT INTO embeddings (entity_id, axis, text_value, vector)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(entity_id, axis) DO UPDATE SET
               text_value = excluded.text_value,
               vector     = excluded.vector""",
        (entity_id, axis, text_value, blob),
    )
    conn.commit()


def store_embeddings_batch(entity_id: int, axis_data: list[tuple[str, str, np.ndarray]]) -> None:
    """Store multiple axis embeddings for an entity in one transaction.

    *axis_data* is a list of ``(axis, text_value, vector)`` tuples.
    """
    conn = _get_conn()
    rows = [
        (entity_id, axis, text_value, vector.astype(np.float64).tobytes())
        for axis, text_value, vector in axis_data
    ]
    conn.executemany(
        """INSERT INTO embeddings (entity_id, axis, text_value, vector)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(entity_id, axis) DO UPDATE SET
               text_value = excluded.text_value,
               vector     = excluded.vector""",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read operations (used at query time)
# ---------------------------------------------------------------------------

def _blob_to_vector(blob: bytes) -> np.ndarray:
    """Convert a BLOB back to a numpy float64 array."""
    return np.frombuffer(blob, dtype=np.float64)


def lookup_entity(name: str) -> Optional[dict]:
    """Look up a pre-computed entity by name (case-insensitive).

    Returns ``None`` if the entity is not in the database, otherwise::

        {
            "id": int,
            "name": str,
            "entity_type": str,
            "profile": dict,        # identity signals
            "embeddings": {          # axis -> np.ndarray
                "primary_domain": array(...),
                ...
            },
            "axis_texts": {          # axis -> original text value
                "primary_domain": "Consumer electronics...",
                ...
            },
        }
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, name, entity_type, profile FROM entities WHERE name = ? COLLATE NOCASE",
        (name,),
    ).fetchone()
    if row is None:
        return None

    entity_id, db_name, entity_type, profile_json = row
    profile = json.loads(profile_json)

    emb_rows = conn.execute(
        "SELECT axis, text_value, vector FROM embeddings WHERE entity_id = ?",
        (entity_id,),
    ).fetchall()

    embeddings = {}
    axis_texts = {}
    for axis, text_value, blob in emb_rows:
        embeddings[axis] = _blob_to_vector(blob)
        axis_texts[axis] = text_value

    return {
        "id": entity_id,
        "name": db_name,
        "entity_type": entity_type,
        "profile": profile,
        "embeddings": embeddings,
        "axis_texts": axis_texts,
    }


def get_all_entities_with_embeddings(axis: str) -> list[tuple[int, str, str, np.ndarray]]:
    """Return all entities that have an embedding for the given axis.

    Returns a list of ``(entity_id, entity_name, entity_type, vector)`` tuples.
    """
    conn = _get_conn()
    rows = conn.execute(
        """SELECT e.id, e.name, e.entity_type, em.vector
           FROM entities e
           JOIN embeddings em ON e.id = em.entity_id
           WHERE em.axis = ?""",
        (axis,),
    ).fetchall()
    return [(eid, name, etype, _blob_to_vector(blob)) for eid, name, etype, blob in rows]


def find_similar_entities(
    query_vector: np.ndarray,
    axis: str,
    entity_type: Optional[str] = None,
    top_n: int = 10,
    exclude_names: Optional[list[str]] = None,
) -> list[dict]:
    """Find the Top-N most similar entities to a query vector on a given axis.

    Returns a list of dicts sorted by descending cosine similarity::

        [{"name": str, "entity_type": str, "similarity": float, "entity_id": int}, ...]
    """
    candidates = get_all_entities_with_embeddings(axis)
    if entity_type:
        candidates = [(eid, n, et, v) for eid, n, et, v in candidates if et == entity_type]
    if exclude_names:
        lower_exclude = {name.lower() for name in exclude_names}
        candidates = [(eid, n, et, v) for eid, n, et, v in candidates if n.lower() not in lower_exclude]

    if not candidates:
        return []

    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return []

    results = []
    for eid, name, etype, vec in candidates:
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            continue
        sim = float(np.dot(query_vector, vec) / (query_norm * vec_norm))
        results.append({"name": name, "entity_type": etype, "similarity": sim, "entity_id": eid})

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_n]


def find_top_matches(
    query_profile: dict,
    query_embeddings: dict[str, np.ndarray],
    weights: dict[str, float],
    entity_type: Optional[str] = None,
    top_n: int = 10,
    exclude_names: Optional[list[str]] = None,
) -> list[dict]:
    """Multi-axis Top-N similarity search using weighted cosine similarity.

    Computes a weighted average of per-axis cosine similarities across all
    axes present in both the query and database, then returns the Top-N results.

    Returns::

        [{"name": str, "entity_type": str, "weighted_similarity": float,
          "axis_similarities": {axis: float, ...}}, ...]
    """
    # Collect per-entity scores across all axes
    entity_scores: dict[int, dict] = {}  # entity_id -> {name, type, axis_sims}

    for axis, weight in weights.items():
        if axis not in query_embeddings:
            continue
        query_vec = query_embeddings[axis]
        # Use a large top_n to get all candidates; we aggregate and re-sort later.
        axis_results = find_similar_entities(
            query_vec, axis, entity_type=entity_type,
            top_n=10000,
            exclude_names=exclude_names,
        )
        for r in axis_results:
            eid = r["entity_id"]
            if eid not in entity_scores:
                entity_scores[eid] = {
                    "name": r["name"],
                    "entity_type": r["entity_type"],
                    "axis_sims": {},
                }
            entity_scores[eid]["axis_sims"][axis] = r["similarity"]

    if not entity_scores:
        return []

    # Compute weighted average similarity per entity
    results = []
    total_weight = sum(w for ax, w in weights.items() if ax in query_embeddings)
    if total_weight == 0:
        return []

    for eid, data in entity_scores.items():
        weighted_sum = 0.0
        for axis, weight in weights.items():
            if axis in data["axis_sims"]:
                weighted_sum += (weight / total_weight) * data["axis_sims"][axis]
        results.append({
            "name": data["name"],
            "entity_type": data["entity_type"],
            "weighted_similarity": weighted_sum,
            "axis_similarities": data["axis_sims"],
        })

    results.sort(key=lambda r: r["weighted_similarity"], reverse=True)
    return results[:top_n]


def entity_count() -> int:
    """Return the total number of entities in the database."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM entities").fetchone()
    return row[0] if row else 0


def has_entity(name: str) -> bool:
    """Check if an entity exists in the database (case-insensitive)."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM entities WHERE name = ? COLLATE NOCASE LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def delete_entity(name: str) -> bool:
    """Delete an entity and its embeddings. Returns True if deleted."""
    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM entities WHERE name = ? COLLATE NOCASE", (name,)
    )
    conn.commit()
    return cursor.rowcount > 0
