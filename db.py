from typing import Any, Dict, List
from pymongo.operations import UpdateOne

try:
    from motor.motor_asyncio import AsyncIOMotorCollection
except ImportError:
    AsyncIOMotorCollection = None  # type: ignore


def ensure_unique_index(col) -> None:
    """
    Ensure indexes exist for efficient queries.
    Note: _id is already unique and indexed by default in MongoDB.
    """
    # Index chunk_id for queries (unique to prevent duplicates)
    try:
        col.create_index("chunk_id", unique=True)
    except Exception:
        # Index might already exist, that's fine
        pass
    
    # Index source.source_id for filtering
    try:
        col.create_index("source.source_id")
    except Exception:
        pass
    
    # Index metadata.tags for filtering
    try:
        col.create_index("metadata.tags")
    except Exception:
        pass


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
    """
    ops = []
    for d in docs:
        doc_id = d["_id"]
        ops.append(
            UpdateOne(
                {"_id": doc_id},
                {"$set": d},
                upsert=True,
            )
        )
    if ops:
        col.bulk_write(ops, ordered=False)


# ----------------------------
# Async (Motor) helpers
# ----------------------------

async def async_ensure_unique_index(col: "AsyncIOMotorCollection") -> None:
    """Ensure indexes exist; use with Motor collection."""
    try:
        await col.create_index("chunk_id", unique=True)
    except Exception:
        pass
    try:
        await col.create_index("source.source_id")
    except Exception:
        pass
    try:
        await col.create_index("metadata.tags")
    except Exception:
        pass


async def async_delete_chunks_by_source(col: "AsyncIOMotorCollection", source_id: str) -> int:
    """Remove all chunks for a given source. Returns deleted count."""
    result = await col.delete_many({"source.source_id": source_id})
    return result.deleted_count


async def async_upsert_chunks(col: "AsyncIOMotorCollection", docs: List[Dict[str, Any]]) -> None:
    """Bulk upsert chunks using stable _id (Motor)."""
    ops = []
    for d in docs:
        doc_id = d["_id"]
        ops.append(
            UpdateOne(
                {"_id": doc_id},
                {"$set": d},
                upsert=True,
            )
        )
    if ops:
        await col.bulk_write(ops, ordered=False)
