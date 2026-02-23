"""
Ingest files (JSON/MD/PDF) -> chunk -> embed -> upsert into MongoDB Atlas Vector Search.

Architecture:
- Mac mini (local): Reads files, chunks, computes embeddings, upserts to Atlas
- MongoDB Atlas: Stores text + metadata + embedding vector with Vector Search index

Install:
  pip install pymongo[srv] openai python-dotenv tiktoken
  # Optional for PDF:
  pip install pdfplumber

Env (.env):
  MONGODB_URI="mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority"
  MONGODB_DB="rag"
  MONGODB_COLLECTION="rag_chunks"
  OPENAI_API_KEY="..."
  OPENAI_EMBED_MODEL="text-embedding-3-small"
  CHUNK_TOKENS=1000
  OVERLAP_TOKENS=150
  BATCH_SIZE=64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from pymongo import MongoClient
from openai import OpenAI, AsyncOpenAI

from chunk import chunk_text_tokens
from config import Settings
from db import (
    delete_chunks_by_source,
    ensure_unique_index,
    upsert_chunks,
    async_delete_chunks_by_source,
    async_ensure_unique_index,
    async_upsert_chunks,
)
from embed import (
    acquire_tokens,
    embed_texts_openai,
    embed_texts_openai_async,
    embed_texts_sentence_transformers,
    embed_texts_sentence_transformers_async,
    estimate_tokens,
)
from normalize import _extract_metadata, detect_file_type, normalize_document
from state import load_state, save_state, should_skip_file, update_file_state
from utils import (
    compute_stable_id,
    get_file_mtime,
    get_files_for_ingest,
    now_iso,
    read_lines_in_blocks,
    sha256_text,
    stable_json_text,
)

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError:
    AsyncIOMotorClient = None


# ----------------------------
# Helpers for doc building
# ----------------------------

def _tags_from_filename(filename: str) -> List[str]:
    """Infer tags from filename for metadata."""
    lower = filename.lower()
    if "profile" in lower:
        return ["profile", "resume", "candidate"]
    if "resume" in lower:
        return ["resume", "candidate"]
    if "qa" in lower:
        return ["qa", "questions"]
    return ["document"]


def _title_for_doc(file_metadata: Optional[Dict[str, Any]], filename: str) -> str:
    """Title from metadata or filename stem."""
    if file_metadata and file_metadata.get("title"):
        return file_metadata["title"]
    return os.path.splitext(filename)[0]


# ----------------------------
# Main ingestion
# ----------------------------

def build_docs_for_file(
    filepath: str,
    embed_client: Any,
    settings: Settings,
) -> List[Dict[str, Any]]:
    """
    Process a single file: normalize -> chunk -> embed -> build MongoDB documents.
    
    Returns list of document dicts matching the target schema.
    """
    filename = os.path.basename(filepath)
    source_id = filename
    file_type = detect_file_type(filepath)
    mtime = get_file_mtime(filepath)
    
    # Normalize document to text
    text, file_metadata = normalize_document(filepath)
    
    # Chunk
    chunks = chunk_text_tokens(
        text=text,
        chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.overlap_tokens,
        model=settings.embed_model,
        chunk_chars=settings.chunk_chars,
        overlap_chars=settings.overlap_chars,
    )
    
    if not chunks:
        return []
    
    # Embed in batches (OpenAI: async with limited concurrency to avoid 429)
    embeddings: List[List[float]] = []
    if settings.embed_provider == "openai":
        sem = asyncio.Semaphore(settings.embed_max_concurrent)

        async def _embed_one(client: Any, model: str, batch: List[str]) -> List[List[float]]:
            cost = sum(estimate_tokens(t) for t in batch)
            await acquire_tokens(cost)
            async with sem:
                return await embed_texts_openai_async(client, model, batch)

        async def _embed_batches() -> List[List[float]]:
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            try:
                batches = [chunks[i : i + settings.batch_size] for i in range(0, len(chunks), settings.batch_size)]
                tasks = [_embed_one(client, settings.embed_model, b) for b in batches]
                results = await asyncio.gather(*tasks)
                return [e for r in results for e in r]
            finally:
                await client.close()
        embeddings = asyncio.run(_embed_batches())
    else:
        embeddings = embed_texts_sentence_transformers(
            embed_client, chunks, batch_size=settings.embed_batch_size_local
        )
    
    # Build MongoDB documents matching target schema
    docs = _docs_from_chunks_embeddings(
        source_id, filepath, file_type, mtime, filename, file_metadata,
        chunks, embeddings, 0, settings,
    )
    return docs


async def build_docs_for_file_async(
    filepath: str,
    embed_client: Any,
    settings: Settings,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Async: normalize -> chunk -> embed -> build MongoDB documents. Returns (docs, timings).
    When progress_callback is set, embedding runs in batches of PROGRESS_CHUNK_INTERVAL and
    callback(n_done, n_total, timings) is invoked after each batch so progress streams."""
    timings: Dict[str, float] = {"load_parse": 0.0, "chunk": 0.0, "embed": 0.0, "mongo_bulk_write": 0.0, "finalize": 0.0}
    filename = os.path.basename(filepath)
    source_id = filename
    file_type = detect_file_type(filepath)
    mtime = get_file_mtime(filepath)
    t0 = time.perf_counter()
    text, file_metadata = normalize_document(filepath)
    timings["load_parse"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    chunks = chunk_text_tokens(
        text=text,
        chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.overlap_tokens,
        model=settings.embed_model,
        chunk_chars=settings.chunk_chars,
        overlap_chars=settings.overlap_chars,
    )
    timings["chunk"] = time.perf_counter() - t0
    if not chunks:
        return [], timings
    n_total = len(chunks)
    embeddings: List[List[float]] = []
    t_embed_start = time.perf_counter()
    if progress_callback is not None:
        # Stream progress: embed in batches of PROGRESS_CHUNK_INTERVAL and report after each
        next_milestone = PROGRESS_CHUNK_INTERVAL
        for start in range(0, n_total, PROGRESS_CHUNK_INTERVAL):
            batch_chunks = chunks[start : start + PROGRESS_CHUNK_INTERVAL]
            if settings.embed_provider == "openai":
                sem = asyncio.Semaphore(settings.embed_max_concurrent)

                async def _embed_one(batch: List[str]) -> List[List[float]]:
                    cost = sum(estimate_tokens(t) for t in batch)
                    await acquire_tokens(cost)
                    async with sem:
                        return await embed_texts_openai_async(
                            embed_client, settings.embed_model, batch
                        )

                api_batches = [batch_chunks[i : i + settings.batch_size] for i in range(0, len(batch_chunks), settings.batch_size)]
                results = await asyncio.gather(*[_embed_one(b) for b in api_batches])
                batch_embs = [e for r in results for e in r]
            else:
                batch_embs = await embed_texts_sentence_transformers_async(
                    embed_client, batch_chunks, batch_size=settings.embed_batch_size_local
                )
            embeddings.extend(batch_embs)
            timings["embed"] = time.perf_counter() - t_embed_start
            n_done = start + len(batch_chunks)
            while next_milestone <= n_done:
                progress_callback(next_milestone, n_total, timings)
                next_milestone += PROGRESS_CHUNK_INTERVAL
    else:
        if settings.embed_provider == "openai":
            sem = asyncio.Semaphore(settings.embed_max_concurrent)

            async def _embed_one(batch: List[str]) -> List[List[float]]:
                cost = sum(estimate_tokens(t) for t in batch)
                await acquire_tokens(cost)
                async with sem:
                    return await embed_texts_openai_async(
                        embed_client, settings.embed_model, batch
                    )

            batches = [chunks[i : i + settings.batch_size] for i in range(0, len(chunks), settings.batch_size)]
            results = await asyncio.gather(*[_embed_one(b) for b in batches])
            embeddings = [e for r in results for e in r]
        else:
            embeddings = await embed_texts_sentence_transformers_async(
                embed_client, chunks, batch_size=settings.embed_batch_size_local
            )
        timings["embed"] = time.perf_counter() - t_embed_start
    docs = _docs_from_chunks_embeddings(
        source_id, filepath, file_type, mtime, filename, file_metadata,
        chunks, embeddings, 0, settings,
    )
    return docs, timings


def _docs_from_chunks_embeddings(
    source_id: str,
    filepath: str,
    file_type: str,
    mtime: str,
    filename: str,
    file_metadata: Optional[Dict[str, Any]],
    chunks: List[str],
    embeddings: List[List[float]],
    chunk_offset: int,
    settings: Settings,
) -> List[Dict[str, Any]]:
    """Build MongoDB doc dicts from chunks and embeddings. Single source of truth for doc schema."""
    if not embeddings:
        return []
    dims = len(embeddings[0]) if embeddings else (384 if settings.embed_provider == "sentence_transformers" else 1536)
    title = _title_for_doc(file_metadata, filename)
    tags = _tags_from_filename(filename)
    ts = now_iso()
    docs: List[Dict[str, Any]] = []
    for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        global_i = chunk_offset + i
        chunk_id = f"{source_id}::chunk_{global_i:04d}"
        chunk_hash = sha256_text(chunk_text)
        doc_id = compute_stable_id(source_id, chunk_id, chunk_hash)
        doc = {
            "_id": doc_id,
            "chunk_id": chunk_id,
            "source": {"source_id": source_id, "path": filepath, "type": file_type, "mtime": mtime},
            "text": chunk_text,
            "metadata": {"title": title, "section": f"chunk_{global_i}", "tags": tags, "lang": "en"},
            "embedding": emb,
            "embedding_model": settings.embed_model,
            "dims": dims,
            "created_at": ts,
            "updated_at": ts,
        }
        docs.append(doc)
    return docs


def _build_docs_for_block(
    source_id: str,
    filepath: str,
    file_type: str,
    mtime: str,
    filename: str,
    file_metadata: Optional[Dict[str, Any]],
    chunks: List[str],
    embeddings: List[List[float]],
    chunk_offset: int,
    settings: Settings,
) -> List[Dict[str, Any]]:
    """Build MongoDB doc dicts for one block of chunks."""
    return _docs_from_chunks_embeddings(
        source_id, filepath, file_type, mtime, filename, file_metadata,
        chunks, embeddings, chunk_offset, settings,
    )


def process_ndjson_blocks(
    filepath: str,
    col: Any,
    embed_client: Any,
    settings: Settings,
    block_size: int = 10,
) -> Tuple[int, bool]:
    """
    Process NDJSON in blocks via queue: producer reads blocks, consumer does chunk->embed->mongo.
    First block is built and processed in the main thread so we don't freeze waiting for the producer.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return 0, False
    try:
        first_obj = json.loads(first_line)
    except json.JSONDecodeError:
        return 0, False

    filename = os.path.basename(filepath)
    source_id = filename
    file_type = detect_file_type(filepath)
    mtime = get_file_mtime(filepath)
    file_metadata = _extract_metadata(first_obj) if isinstance(first_obj, dict) else None

    t0 = time.perf_counter()
    delete_chunks_by_source(col, source_id)
    print(f"  [db_delete] {time.perf_counter() - t0:.3f}s")

    chunk_offset = 0
    total_docs = 0
    total_timings: Dict[str, float] = {"load_parse": 0.0, "chunk": 0.0, "embed": 0.0, "mongo_bulk_write": 0.0, "finalize": 0.0}

    async def _embed_block_batches(chunks_block: List[str], block_num: Optional[int] = None) -> List[List[float]]:
        sem = asyncio.Semaphore(settings.embed_max_concurrent)
        async def _embed_one(client: Any, model: str, batch: List[str]) -> List[List[float]]:
            cost = sum(estimate_tokens(t) for t in batch)
            await acquire_tokens(cost)
            async with sem:
                return await embed_texts_openai_async(client, model, batch)
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        try:
            batches = [chunks_block[i : i + settings.batch_size] for i in range(0, len(chunks_block), settings.batch_size)]
            n_batches = len(batches)
            done_count = 0
            done_lock = asyncio.Lock()

            async def _with_progress(batch: List[str]) -> List[List[float]]:
                nonlocal done_count
                r = await _embed_one(client, settings.embed_model, batch)
                async with done_lock:
                    done_count += 1
                    # if n_batches > 1 and block_num is not None:
                    #     print(f"  Block {block_num}: embedding batch {done_count}/{n_batches}...", flush=True)
                return r

            tasks = [_with_progress(b) for b in batches]
            results = await asyncio.gather(*tasks)
            return [e for r in results for e in r]
        finally:
            await client.close()

    def process_one_block(block_num: int, objs: List[Any]) -> None:
        nonlocal chunk_offset, total_docs, total_timings
        n_objs = len(objs) if objs else 0
        # print(f"  Block {block_num}: parsing {n_objs} objects...", flush=True)
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        text = stable_json_text(objs)
        timings["load_parse"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunks = chunk_text_tokens(
            text=text,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.overlap_tokens,
            model=settings.embed_model,
            chunk_chars=settings.chunk_chars,
            overlap_chars=settings.overlap_chars,
        )
        timings["chunk"] = time.perf_counter() - t0
        if not chunks:
            for k in total_timings:
                total_timings[k] += timings.get(k, 0.0)
            return
        # print(f"  Block {block_num}: {len(chunks)} chunks, embedding...", flush=True)
        t0 = time.perf_counter()
        if settings.embed_provider == "openai":
            embeddings = asyncio.run(_embed_block_batches(chunks, block_num=block_num))
        else:
            embeddings = embed_texts_sentence_transformers(
                embed_client, chunks, batch_size=settings.embed_batch_size_local
            )
        timings["embed"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        docs = _build_docs_for_block(
            source_id, filepath, file_type, mtime, filename, file_metadata,
            chunks, embeddings, chunk_offset, settings,
        )
        t0_upsert = time.perf_counter()
        upsert_chunks(col, docs)
        timings["mongo_bulk_write"] = time.perf_counter() - t0_upsert
        timings["finalize"] = 0.0
        chunk_offset += len(chunks)
        total_docs += len(docs)
        for k in total_timings:
            total_timings[k] += timings.get(k, 0.0)
        n_chunks = len(chunks)
        lp, ch, em, mb = timings["load_parse"], timings["chunk"], timings["embed"], timings["mongo_bulk_write"]
        print(f"  block {block_num}: summary: load_parse={lp:.3f}s chunk={ch:.3f}s embed={em:.3f}s mongo_bulk_write={mb:.3f}s chunks={n_chunks}")

    # Build first block in main thread (no wait on producer) to avoid freeze
    first_block_lines: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        f.readline()  # skip first line (already have first_obj)
        for _ in range(block_size - 1):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                first_block_lines.append(line)
    first_objs: List[Any] = [first_obj]
    for line in first_block_lines:
        try:
            first_objs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    process_one_block(1, first_objs)

    # Remaining blocks via producer/consumer
    maxsize = settings.block_queue_size if settings.block_queue_size > 0 else 0
    block_queue: "queue.Queue[Tuple[Optional[int], Optional[List[Any]]]]" = queue.Queue(maxsize=maxsize)

    def producer() -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            for _ in range(block_size):  # skip first block (already processed)
                if not f.readline():
                    break
            block_num = 1
            for block in read_lines_in_blocks(f, block_size=block_size):
                block_num += 1
                objs: List[Any] = []
                for line in block:
                    objs.append(json.loads(line))
                block_queue.put((block_num, objs))
        block_queue.put((None, None))

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    while True:
        block_num, objs = block_queue.get()
        if block_num is None:
            break
        process_one_block(block_num, objs)
    lp, ch, em, mb = total_timings["load_parse"], total_timings["chunk"], total_timings["embed"], total_timings["mongo_bulk_write"]
    print(f"  total summary: load_parse={lp:.3f}s chunk={ch:.3f}s embed={em:.3f}s mongo_bulk_write={mb:.3f}s chunks={total_docs}")
    return total_docs, True


def ingest_folder(
    folder_glob: str = "data/**/*",
    skip_unchanged: bool = True,
) -> None:
    """
    Ingest all matching files from folder.
    
    Args:
        folder_glob: Glob pattern for files to ingest (e.g., "data/**/*.json")
        skip_unchanged: If True, skip files that haven't changed since last ingest
    """
    settings = Settings()
    
    assert settings.mongodb_uri, "Missing MONGODB_URI"
    if settings.embed_provider == "openai":
        assert settings.openai_api_key, "Missing OPENAI_API_KEY"
    
    # MongoDB connection
    mongo = MongoClient(settings.mongodb_uri)
    db = mongo[settings.mongodb_db]
    col = db[settings.mongodb_collection]
    ensure_unique_index(col)
    
    # Embed client (OpenAI or SentenceTransformer)
    if settings.embed_provider == "openai":
        embed_client = OpenAI(api_key=settings.openai_api_key)
        print(f"Embed: openai ({settings.embed_model})")
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Embed: sentence_transformers ({settings.embed_model})")
        embed_client = SentenceTransformer(
            settings.embed_model,
            device=settings.embed_device or None,
        )
    
    # Load state for incremental ingestion
    state = load_state()
    start = time.perf_counter()
    files = get_files_for_ingest(folder_glob)
    print(f"Found {len(files)} files matching pattern")
    
    total_docs = 0
    skipped = 0
    last_printed_round = 0

    for filepath in files:
        # Check if file should be skipped (incremental ingestion)
        if skip_unchanged:
            try:
                text, _ = normalize_document(filepath)
                content_hash = sha256_text(text)
                mtime = get_file_mtime(filepath)
                
                if should_skip_file(filepath, content_hash, mtime, state):
                    print(f"Skipping unchanged: {filepath}")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"Warning: Could not check state for {filepath}: {e}")
        
        # Process file
        try:
            print(f"Processing: {filepath}")
            doc_start = time.perf_counter()
            if filepath.endswith(".json"):
                n, ok = process_ndjson_blocks(filepath, col, embed_client, settings, block_size=10)
                if ok:
                    total_docs += n
                    if skip_unchanged:
                        text, _ = normalize_document(filepath)
                        content_hash = sha256_text(text)
                        mtime = get_file_mtime(filepath)
                        update_file_state(filepath, content_hash, mtime, state)
                    continue
            docs = build_docs_for_file(filepath, embed_client, settings)

            if docs:
                source_id = docs[0]["source"]["source_id"]
                deleted = delete_chunks_by_source(col, source_id)
                if deleted:
                    print(f"  Deleted {deleted} old chunks for {source_id}")
                upsert_chunks(col, docs)
                total_docs += len(docs)
                while last_printed_round + 640 <= total_docs:
                    last_printed_round += 640
                    print(f"  ... Processed {last_printed_round} chunks total")
                
                # Update state
                if skip_unchanged:
                    text, _ = normalize_document(filepath)
                    content_hash = sha256_text(text)
                    mtime = get_file_mtime(filepath)
                    update_file_state(filepath, content_hash, mtime, state)
                
                elapsed = time.perf_counter() - doc_start
                print(f"  ✓ Ingested {len(docs)} chunks ({elapsed:.2f}s)")
            else:
                elapsed = time.perf_counter() - doc_start
                print(f"  ⚠ No chunks generated ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  ✗ Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save state
    if skip_unchanged:
        save_state(state)

    durable_s = time.perf_counter() - start
    print(f"\nDone.")
    print(f"  Durable time: {durable_s:.2f}s")
    print(f"  Total chunks upserted: {total_docs}")
    print(f"  Files skipped (unchanged): {skipped}")
    print(f"  MongoDB: {settings.mongodb_db}.{settings.mongodb_collection}")
    print(f"\nNext steps:")
    dim_hint = 384 if settings.embed_provider == "sentence_transformers" else 1536
    print(f"  1. Create Vector Search index in Atlas UI:")
    print(f"     - Field: embedding (knnVector, dims={dim_hint})")
    print(f"     - Optional filters: source.source_id, metadata.tags")
    print(f"  2. Optional: Add text index for hybrid search (field: text)")


PROGRESS_CHUNK_INTERVAL = 640  # Print progress every N chunks (async path: upsert in batches and print after each)


async def _process_one_file_async(
    filepath: str,
    col: Any,
    embed_client: Any,
    settings: Settings,
    state: Dict[str, Dict[str, str]],
    block_num: Optional[int] = None,
) -> tuple[int, Exception | None, Optional[Dict[str, float]]]:
    """Process one file: build_docs_async -> delete_by_source -> upsert -> update state. Returns (num_docs, error, timings)."""
    def _embed_progress(n_done: int, n_total: int, t: Dict[str, float]) -> None:
        lp, ch, em = t["load_parse"], t["chunk"], t["embed"]
        print(f"  progress {n_done} chunks: load_parse={lp:.3f}s chunk={ch:.3f}s embed={em:.3f}s mongo_bulk_write=0.000s", flush=True)
    # Always stream progress during embed (in-process async and RabbitMQ workers)
    docs, timings = await build_docs_for_file_async(filepath, embed_client, settings, progress_callback=_embed_progress)
    if not docs:
        return 0, None, timings
    source_id = docs[0]["source"]["source_id"]
    t0 = time.perf_counter()
    await async_delete_chunks_by_source(col, source_id)
    t_mongo = 0.0
    if block_num is not None:
        # Upsert in batches (no per-batch print; progress was already printed during embed)
        for start in range(0, len(docs), PROGRESS_CHUNK_INTERVAL):
            batch = docs[start : start + PROGRESS_CHUNK_INTERVAL]
            t_batch = time.perf_counter()
            await async_upsert_chunks(col, batch)
            t_mongo += time.perf_counter() - t_batch
        timings["mongo_bulk_write"] = t_mongo
    else:
        await async_upsert_chunks(col, docs)
        timings["mongo_bulk_write"] = time.perf_counter() - t0
    lp, ch, em, mb = timings["load_parse"], timings["chunk"], timings["embed"], timings["mongo_bulk_write"]
    n_chunks = len(docs)
    if block_num is not None:
        # print(f"  block {block_num}: summary: ...")
        pass
    # When block_num is None (e.g. RabbitMQ worker), caller prints ✓ and total summary
    text, _ = normalize_document(filepath)
    content_hash = sha256_text(text)
    mtime = get_file_mtime(filepath)
    update_file_state(filepath, content_hash, mtime, state)
    return len(docs), None, timings


def ingest_folder_async(
    folder_glob: str = "data/**/*",
    skip_unchanged: bool = True,
) -> None:
    """Ingest using async workers in-process (no queue). Saves state at end."""
    settings = Settings()
    assert settings.mongodb_uri, "Missing MONGODB_URI"
    if settings.embed_provider == "openai":
        assert settings.openai_api_key, "Missing OPENAI_API_KEY"
    assert AsyncIOMotorClient is not None, "Install motor: pip install motor"

    async def _run() -> None:
        start = time.perf_counter()
        mongo = AsyncIOMotorClient(settings.mongodb_uri)
        db = mongo[settings.mongodb_db]
        col = db[settings.mongodb_collection]
        await async_ensure_unique_index(col)
        if settings.embed_provider == "openai":
            embed_client = AsyncOpenAI(api_key=settings.openai_api_key)
            print(f"Embed: openai ({settings.embed_model})")
        else:
            from sentence_transformers import SentenceTransformer
            print(f"Embed: sentence_transformers ({settings.embed_model})")
            embed_client = SentenceTransformer(
                settings.embed_model,
                device=settings.embed_device or None,
            )
        state = load_state()
        files = get_files_for_ingest(folder_glob)
        to_process: List[str] = []
        skipped = 0
        for filepath in files:
            if skip_unchanged:
                try:
                    text, _ = normalize_document(filepath)
                    content_hash = sha256_text(text)
                    mtime = get_file_mtime(filepath)
                    if should_skip_file(filepath, content_hash, mtime, state):
                        print(f"Skipping unchanged: {filepath}")
                        skipped += 1
                        continue
                except Exception as e:
                    print(f"Warning: Could not check state for {filepath}: {e}")
            to_process.append(filepath)
        print(f"Found {len(files)} files; {len(to_process)} to process, {skipped} skipped (unchanged)")
        if not to_process:
            if skip_unchanged:
                save_state(state)
            print(f"Nothing to do. Durable time: {time.perf_counter() - start:.2f}s")
            return

        sem = asyncio.Semaphore(settings.max_concurrent_files)
        progress_lock = asyncio.Lock()
        total_docs = 0
        processed = 0
        errors = 0
        last_printed_round = 0
        total_timings: Dict[str, float] = {"load_parse": 0.0, "chunk": 0.0, "embed": 0.0, "mongo_bulk_write": 0.0, "finalize": 0.0}

        async def process_with_semaphore(filepath: str, block_num: int) -> None:
            nonlocal total_docs, processed, errors, last_printed_round
            async with sem:
                try:
                    doc_start = time.perf_counter()
                    n, _, timings = await _process_one_file_async(
                        filepath, col, embed_client, settings, state, block_num=block_num
                    )
                    elapsed = time.perf_counter() - doc_start
                    async with progress_lock:
                        total_docs += n
                        if timings:
                            for k in total_timings:
                                total_timings[k] += timings.get(k, 0.0)
                        while last_printed_round + 640 <= total_docs:
                            last_printed_round += 640
                            # Progress already printed from inside _process_one_file_async (per-file streaming)
                        processed += 1
                    print(f"  ✓ {filepath} -> {n} chunks ({elapsed:.2f}s)", flush=True)
                except Exception as e:
                    errors += 1
                    print(f"  ✗ {filepath}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        await asyncio.gather(*[process_with_semaphore(fp, i + 1) for i, fp in enumerate(to_process)])

        lp, ch, em, mb = total_timings["load_parse"], total_timings["chunk"], total_timings["embed"], total_timings["mongo_bulk_write"]
        print(f"  total summary: load_parse={lp:.3f}s chunk={ch:.3f}s embed={em:.3f}s mongo_bulk_write={mb:.3f}s chunks={total_docs}", flush=True)

        if skip_unchanged:
            save_state(state)
        durable_s = time.perf_counter() - start
        print(f"\nDone (async).")
        print(f"  Durable time: {durable_s:.2f}s")
        print(f"  Total chunks upserted: {total_docs}")
        print(f"  Files processed: {processed}, errors: {errors}")
        print(f"  MongoDB: {settings.mongodb_db}.{settings.mongodb_collection}")

    asyncio.run(_run())


def add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", default="dev")
    parser.add_argument("--target", default="localhost", choices=["localhost", "atlas"], help="MongoDB: localhost or atlas")
    parser.add_argument("--mode", choices=["sync", "async"], default="async")
    parser.add_argument("--queue", choices=["none", "memory", "redis", "rabbitmq"], default="memory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedder", default="sentence-transformers")
    parser.add_argument("--input-dir", default="./data", help="Directory to glob for files (default: ./data)")
    parser.add_argument("--pattern", default="**/*", help="Glob pattern under input-dir, e.g. *.json (default: **/*)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _ingest_glob(args: argparse.Namespace) -> str:
    """Build effective file glob from --input-dir and --pattern (recursive if pattern has no **)."""
    base = args.input_dir.rstrip(os.sep)
    pat = args.pattern.lstrip(os.sep).replace("\\", "/")
    if "**" in pat:
        combined = f"{base}{os.sep}{pat}"
    else:
        combined = f"{base}{os.sep}**{os.sep}{pat}"
    return combined.replace("\\", "/")


if __name__ == "__main__":
    # Drop empty/whitespace-only args (e.g. from multiline paste or docker-compose) so we don't get "unrecognized arguments"
    sys.argv = [a for a in sys.argv if a and a.strip()]

    # Line-buffer stdout so progress logs appear as they're printed (not in one shot at the end, e.g. in Docker)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    COLLECTIONS = {"dev": "collection_taixingbi_dev", "qa": "collection_taixingbi_qa", "prod": "collection_taixingbi_prod"}

    parser = argparse.ArgumentParser(description="RAG ingest: chunk, embed, upsert to MongoDB")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", help="Run ingest pipeline")
    add_ingest_args(ingest_parser)
    args = parser.parse_args()

    assert args.command == "ingest"
    folder_glob = _ingest_glob(args)

    # Env: set collection
    if args.env in COLLECTIONS:
        os.environ["MONGODB_COLLECTION"] = COLLECTIONS[args.env]
    # Target: which MongoDB
    if args.target == "localhost":
        os.environ["MONGODB_URI"] = os.environ.get("MONGODB_URI_LOCAL", "mongodb://localhost:27017")

    # Embedder -> EMBED_PROVIDER (openai | sentence_transformers)
    if args.embedder == "sentence-transformers":
        os.environ["EMBED_PROVIDER"] = "sentence_transformers"
    elif args.embedder == "openai":
        os.environ["EMBED_PROVIDER"] = "openai"
    os.environ["BATCH_SIZE"] = str(args.batch_size)

    skip_unchanged = not args.force
    use_async = args.mode == "async"
    use_rabbitmq = args.queue == "rabbitmq"

    if use_async and use_rabbitmq:
        from queue_rabbit import ingest_via_rabbitmq
        ingest_via_rabbitmq(folder_glob, skip_unchanged=skip_unchanged, num_workers=args.workers)
    elif use_async:
        ingest_folder_async(folder_glob, skip_unchanged=skip_unchanged)
    else:
        ingest_folder(folder_glob, skip_unchanged=skip_unchanged)
