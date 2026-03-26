# Auto-Ingest Feature

## Overview

When a user analyzes a celebrity-brand alignment for entities **not in the pre-computed database**, the system automatically stores the computed identity profiles for future fast lookups.

## How It Works

### Flow

1. **Analysis Request**: User enters two entities and clicks "Analyze Alignment"
2. **Precomputed Check**: `_try_precomputed()` checks if entities exist in the SQLite database
3. **Wikipedia + Scoring**: If not precomputed:
   - Fetch Wikipedia text
   - Extract identity signals via Gemini LLM
   - Compute axis embeddings (single batch API call)
   - Compute alignment scores
4. **Queue for Storage**: After results cached, queue any missing entities for background storage
5. **Background Thread**: Daemon thread calls `store_analyzed_entity()` for each queued entity
   - Re-embeds axis texts using **cached embedding function** (cache hits only, no API calls)
   - Stores identity profile + embeddings in SQLite
6. **Next Search**: Entity now found in database, skipping Wikipedia + extraction entirely

### Benefits

- ✅ **Zero Duplicate API Calls**: Uses cached embeddings computed during analysis
- ✅ **Non-Blocking**: Background threads don't delay returning analysis results to user
- ✅ **Idempotent**: Safe to call multiple times—duplicates are skipped (ON CONFLICT logic)
- ✅ **Graceful Error Handling**: Failures logged, don't block UI or propagate to user
- ✅ **Efficient**: Re-embedding is instant (pure cache lookups, no API roundtrips)

## Files Modified

### `services/auto_ingest.py` (NEW)
Two key functions:

**`store_analyzed_entity(name, entity_type, identity_profile) -> bool`**
- Stores pre-computed identity profile to database
- Re-embeds axis texts using cached embedding function (cache hits only, no API calls)
- Logs errors gracefully, returns True if succeeded
- Skips if entity already exists (idempotent)

**`process_pending_ingests(pending_list) -> None`**
- Takes list of dicts: `{"name": str, "type": str, "identity_profile": dict}`
- Launches daemon thread to process storage
- Non-blocking—doesn't wait for completion
- Each dict contains the pre-computed identity signals from analysis

### `app.py` (MODIFIED)
In `_run_analysis()` after successful analysis (lines 144-160):

1. Tracks which entities were NOT precomputed:
   ```python
   precomputed_a = _try_precomputed(lookup_a)  # None if not in DB
   precomputed_b = _try_precomputed(lookup_b)
   ```

2. After caching results, queues missing entities with their computed profiles:
   ```python
   if precomputed_a is None:
       pending_ingest.append({"name": lookup_a, "type": "person", "identity_profile": identity_a})
   if precomputed_b is None:
       pending_ingest.append({"name": lookup_b, "type": "organization", "identity_profile": identity_b})

   if pending_ingest:
       process_pending_ingests(pending_ingest)
   ```

## Edge Cases Handled

| Case | Handling |
|------|----------|
| **Entity Already in DB** | Skipped (checked by `has_entity()` before insert) |
| **Missing Axis Embeddings** | Logged as warning, partial storage (only successfully embedded axes) |
| **No API Calls in Background** | All embeddings are cache hits (computed during main analysis) |
| **Embedding Cache Miss** | Unlikely—embeddings from analysis are cached; graceful fallback if rare |
| **App Restart** | Pending storage lost (acceptable—next search re-triggers if needed) |
| **Concurrent Sessions** | Each Streamlit session has isolated `st.session_state` |
| **Concurrent Writes** | SQLite WAL mode prevents locking issues |

## Testing

### Manual Test: New Entity Auto-Ingest
1. Start the Streamlit app
2. Search for a celebrity/brand NOT in the curated list
3. Complete the alignment analysis
4. Wait 2-3 seconds for background ingestion
5. Verify: Run `python ingest_entities.py --stats` and check entity count increased

### Manual Test: Already Ingested Entity
1. Search for same entity again
2. Verify: `_try_precomputed()` finds it instantly (no background ingest needed)
3. Database is not modified (ON CONFLICT is idempotent)

### Manual Test: Disambiguation Resolution
1. Search for ambiguous entity (e.g., "John Smith")
2. Select from disambiguation picker
3. Analyze successfully
4. Verify: Database stores **resolved Wikipedia title**, not ambiguous input

### Manual Test: No Duplicate API Calls
1. Enable API call logging/monitoring in embedding_service
2. Analyze a new entity pair (should see: 2 identity extraction calls, 1 embedding batch call)
3. Check that background storage does NOT make new API calls
4. Terminal logs should show: "Auto-ingested entity: [Name] ([type]) with X axes"
5. No new Gemini API calls should appear after "Analyzing alignment..." spinner ends

## Database Impact

- **New columns**: None (uses existing schema)
- **Schema changes**: None (uses existing `entities` and `embeddings` tables)
- **Indexes**: No changes (existing indexes on `name` and `entity_id, axis` used)
- **Concurrency**: WAL mode prevents blocking during writes

## Performance Notes

- **Non-blocking**: Analysis results return immediately; ingestion happens asynchronously
- **Threading**: Uses Python `threading.Thread` with `daemon=True`
  - Survives Streamlit reruns
  - Doesn't block app shutdown
  - One thread per pending batch (typically 1-2 entities per analysis)
- **Database**: SQLite with WAL mode allows concurrent reads during writes

## Future Enhancements

1. **UI Notification**: Show subtle badge when entities are being ingested ("Added X to database")
2. **Ingest Progress**: Track completion and display in session state
3. **Batch Optimization**: Group multiple pending ingests into single thread
4. **Rate Limiting**: Add configurable delay between background ingests to avoid API throttling
5. **Analytics**: Log which entities are auto-ingested vs. user-provided for analysis

## Troubleshooting

**Issue**: Still seeing excessive API calls
- **Cause**: Auto-ingest calling `_embed_text()` when cache should have results
- **Fix**: Verify embedding cache is populated during main analysis (check `_embed_cache` lock in embedding_service.py)

**Issue**: Background storage not happening
- **Cause**: Identity profile not passed correctly from app.py
- **Fix**: Verify `identity_a` and `identity_b` are dicts in cache (line 137-138 of app.py)

**Issue**: Duplicate entries in database
- **Cause**: Race condition between cache check and insert
- **Fix**: `has_entity()` check at start of `store_analyzed_entity()` prevents this

**Issue**: Partial axis storage
- **Cause**: Some axis embeddings failed while others succeeded
- **Fix**: Check logs for "Failed to embed axis X" warnings; those axes are skipped

## Related Files

- `ingest_entities.py` - Core ingestion logic (reused by auto_ingest)
- `services/embedding_service.py` - Identity extraction & embedding
- `services/wikipedia_service.py` - Wikipedia fetch & disambiguation
- `services/entity_db.py` - Database operations (CRUD)
- `app.py` - Main Streamlit app (integration point)
