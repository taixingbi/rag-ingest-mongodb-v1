from typing import Any, Dict, List
from pymongo.operations import UpdateOne

try:
    from motor.motor_asyncio import AsyncIOMotorCollection
except ImportError:
    AsyncIOMotorCollection = None  # type: ignore

# Batch size for bulk_write to avoid very large single requests (e.g. 100k docs)
BULK_WRITE_BATCH = 5000


def _log_index_error(name: str, e: Exception) -> None:
    """Log index creation failure; duplicate index (e.g. code 85/86) is expected and can be ignored."""
    msg = str(e).lower()
    if "already exists" in msg or "duplicate" in msg or (getattr(e, "code", None) in (85, 86)):
        return
    print(f"Warning: index '{name}' creation failed: {e}", flush=True)


def ensure_unique_index(col) -> None:
    """
    Ensure indexes exist for efficient queries.
    Note: _id is already unique and indexed by default in MongoDB.
    """
    try:
        col.create_index("chunk_id", unique=True)
    except Exception as e:
        _log_index_error("chunk_id", e)
    try:
        col.create_index("source.source_id")
    except Exception as e:
        _log_index_error("source.source_id", e)
    try:
        col.create_index("metadata.tags")
    except Exception as e:
        _log_index_error("metadata.tags", e)


def delete_chunks_by_source(col, source_id: str) -> int:
    """
    Remove all chunks for a given source (e.g. filename).
    Returns the number of documents deleted.
    Call this before upserting when re-ingesting an updated file so old chunks don't remain.
    """
    result = col.delete_many({"source.source_id": source_id})
    return result.deleted_count


def upsert_chunks(col, docs: List[Dict[str, Any]]) -> None:
    """
    Bulk upsert chunks using stable _id.
    Uses UpdateOne with upsert=True for idempotent ingestion.
    Batches writes to BULK_WRITE_BATCH to avoid oversized requests.
    """
    for i in range(0, len(docs), BULK_WRITE_BATCH):
        batch = docs[i : i + BULK_WRITE_BATCH]
        ops = [
            UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True)
            for d in batch
        ]
        if ops:
            col.bulk_write(ops, ordered=False)


# ----------------------------
# Async (Motor) helpers
# ----------------------------

async def async_ensure_unique_index(col: "AsyncIOMotorCollection") -> None:
    """Ensure indexes exist; use with Motor collection."""
    try:
        await col.create_index("chunk_id", unique=True)
    except Exception as e:
        _log_index_error("chunk_id", e)
    try:
        await col.create_index("source.source_id")
    except Exception as e:
        _log_index_error("source.source_id", e)
    try:
        await col.create_index("metadata.tags")
    except Exception as e:
        _log_index_error("metadata.tags", e)


async def async_delete_chunks_by_source(col: "AsyncIOMotorCollection", source_id: str) -> int:
    """Remove all chunks for a given source. Returns deleted count."""
    result = await col.delete_many({"source.source_id": source_id})
    return result.deleted_count


async def async_upsert_chunks(col: "AsyncIOMotorCollection", docs: List[Dict[str, Any]]) -> None:
    """Bulk upsert chunks using stable _id (Motor). Batches to BULK_WRITE_BATCH."""
    for i in range(0, len(docs), BULK_WRITE_BATCH):
        batch = docs[i : i + BULK_WRITE_BATCH]
        ops = [
            UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True)
            for d in batch
        ]
        if ops:
            await col.bulk_write(ops, ordered=False)
