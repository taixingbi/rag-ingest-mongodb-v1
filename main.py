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

import asyncio
import json
import os
import glob
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

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
    content_hash = sha256_text(text)
    
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
            embed_client, chunks, batch_size=settings.batch_size
        )
    
    # Build MongoDB documents matching target schema
    docs: List[Dict[str, Any]] = []
    dims = len(embeddings[0]) if embeddings else (384 if settings.embed_provider == "sentence_transformers" else 1536)
    
    for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{source_id}::chunk_{i:04d}"
        chunk_hash = sha256_text(chunk_text)
        
        # Compute stable _id
        doc_id = compute_stable_id(source_id, chunk_id, chunk_hash)
        
        # Extract metadata
        title = file_metadata.get("title") if file_metadata else None
        if not title:
            # Fallback: use filename without extension
            title = os.path.splitext(filename)[0]
        
        # Determine tags based on file type and name
        tags = []
        if "profile" in filename.lower():
            tags.extend(["profile", "resume", "candidate"])
        elif "resume" in filename.lower():
            tags.extend(["resume", "candidate"])
        elif "qa" in filename.lower():
            tags.extend(["qa", "questions"])
        else:
            tags.append("document")
        
        doc = {
            "_id": doc_id,
            "chunk_id": chunk_id,
            "source": {
                "source_id": source_id,
                "path": filepath,
                "type": file_type,
                "mtime": mtime,
            },
            "text": chunk_text,
            "metadata": {
                "title": title,
                "section": f"chunk_{i}",
                "tags": tags,
                "lang": "en",
            },
            "embedding": emb,
            "embedding_model": settings.embed_model,
            "dims": dims,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        docs.append(doc)
    
    return docs


async def build_docs_for_file_async(
    filepath: str,
    embed_client: Any,
    settings: Settings,
) -> List[Dict[str, Any]]:
    """Async: normalize -> chunk -> embed -> build MongoDB documents."""
    filename = os.path.basename(filepath)
    source_id = filename
    file_type = detect_file_type(filepath)
    mtime = get_file_mtime(filepath)
    text, file_metadata = normalize_document(filepath)
    content_hash = sha256_text(text)
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
    embeddings: List[List[float]] = []
    if settings.embed_provider == "openai":
        for i in range(0, len(chunks), settings.batch_size):
            batch = chunks[i : i + settings.batch_size]
            cost = sum(estimate_tokens(t) for t in batch)
            await acquire_tokens(cost)
            batch_embeddings = await embed_texts_openai_async(
                embed_client, settings.embed_model, batch
            )
            embeddings.extend(batch_embeddings)
    else:
        embeddings = await embed_texts_sentence_transformers_async(
            embed_client, chunks, batch_size=settings.batch_size
        )
    docs = []
    dims = len(embeddings[0]) if embeddings else (384 if settings.embed_provider == "sentence_transformers" else 1536)
    for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{source_id}::chunk_{i:04d}"
        chunk_hash = sha256_text(chunk_text)
        doc_id = compute_stable_id(source_id, chunk_id, chunk_hash)
        title = file_metadata.get("title") if file_metadata else None
        if not title:
            title = os.path.splitext(filename)[0]
        tags = []
        if "profile" in filename.lower():
            tags.extend(["profile", "resume", "candidate"])
        elif "resume" in filename.lower():
            tags.extend(["resume", "candidate"])
        elif "qa" in filename.lower():
            tags.extend(["qa", "questions"])
        else:
            tags.append("document")
        doc = {
            "_id": doc_id,
            "chunk_id": chunk_id,
            "source": {"source_id": source_id, "path": filepath, "type": file_type, "mtime": mtime},
            "text": chunk_text,
            "metadata": {"title": title, "section": f"chunk_{i}", "tags": tags, "lang": "en"},
            "embedding": emb,
            "embedding_model": settings.embed_model,
            "dims": dims,
            "created_at": now_iso(),
            "updated_at": now_iso(),
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
    docs: List[Dict[str, Any]] = []
    dims = len(embeddings[0]) if embeddings else (384 if settings.embed_provider == "sentence_transformers" else 1536)
    title = (file_metadata or {}).get("title") or os.path.splitext(filename)[0]
    tags = []
    if "profile" in filename.lower():
        tags.extend(["profile", "resume", "candidate"])
    elif "resume" in filename.lower():
        tags.extend(["resume", "candidate"])
    elif "qa" in filename.lower():
        tags.extend(["qa", "questions"])
    else:
        tags.append("document")
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
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        docs.append(doc)
    return docs


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
                embed_client, chunks, batch_size=settings.batch_size
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
        lp, ch, em, mb, fi = timings["load_parse"], timings["chunk"], timings["embed"], timings["mongo_bulk_write"], timings["finalize"]
        print(f"  block {block_num}: summary: load_parse={lp:.3f}s chunk={ch:.3f}s embed={em:.3f}s mongo_bulk_write={mb:.3f}s finalize={fi:.3f}s chunks={n_chunks}")

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
    lp, ch, em, mb, fi = total_timings["load_parse"], total_timings["chunk"], total_timings["embed"], total_timings["mongo_bulk_write"], total_timings["finalize"]
    print(f"  total summary: load_parse={lp:.3f}s chunk={ch:.3f}s embed={em:.3f}s mongo_bulk_write={mb:.3f}s finalize={fi:.3f}s chunks={total_docs}")
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
        embed_client = SentenceTransformer(settings.embed_model)
    
    # Load state for incremental ingestion
    state = load_state()
    start = time.perf_counter()

    # Find files (support multiple extensions)
    patterns = [
        folder_glob,
        folder_glob.replace("**/*", "**/*.json"),
        folder_glob.replace("**/*", "**/*.md"),
        folder_glob.replace("**/*", "**/*.txt"),
        folder_glob.replace("**/*", "**/*.pdf"),
    ]
    
    all_files = set()
    for pattern in patterns:
        all_files.update(glob.glob(pattern, recursive=True))
    
    # Filter out directories
    files = sorted([f for f in all_files if os.path.isfile(f)])
    
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


async def _process_one_file_async(
    filepath: str,
    col: Any,
    embed_client: Any,
    settings: Settings,
    state: Dict[str, Dict[str, str]],
) -> tuple[int, Exception | None]:
    """Process one file: build_docs_async -> delete_by_source -> upsert -> update state. Returns (num_docs, error)."""
    docs = await build_docs_for_file_async(filepath, embed_client, settings)
    if not docs:
        return 0, None
    source_id = docs[0]["source"]["source_id"]
    await async_delete_chunks_by_source(col, source_id)
    await async_upsert_chunks(col, docs)
    text, _ = normalize_document(filepath)
    content_hash = sha256_text(text)
    mtime = get_file_mtime(filepath)
    update_file_state(filepath, content_hash, mtime, state)
    return len(docs), None


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
            embed_client = SentenceTransformer(settings.embed_model)
        state = load_state()

        patterns = [
            folder_glob,
            folder_glob.replace("**/*", "**/*.json"),
            folder_glob.replace("**/*", "**/*.md"),
            folder_glob.replace("**/*", "**/*.txt"),
            folder_glob.replace("**/*", "**/*.pdf"),
        ]
        all_files = set()
        for pattern in patterns:
            all_files.update(glob.glob(pattern, recursive=True))
        files = sorted([f for f in all_files if os.path.isfile(f)])
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

        async def process_with_semaphore(filepath: str) -> None:
            nonlocal total_docs, processed, errors, last_printed_round
            async with sem:
                try:
                    doc_start = time.perf_counter()
                    n, _ = await _process_one_file_async(
                        filepath, col, embed_client, settings, state
                    )
                    elapsed = time.perf_counter() - doc_start
                    async with progress_lock:
                        total_docs += n
                        while last_printed_round + 640 <= total_docs:
                            last_printed_round += 640
                            print(f"  ... Processed {last_printed_round} chunks total")
                        processed += 1
                    print(f"  ✓ {filepath} -> {n} chunks ({elapsed:.2f}s)")
                except Exception as e:
                    errors += 1
                    print(f"  ✗ {filepath}: {e}")
                    import traceback
                    traceback.print_exc()

        await asyncio.gather(*[process_with_semaphore(fp) for fp in to_process])

        if skip_unchanged:
            save_state(state)
        durable_s = time.perf_counter() - start
        print(f"\nDone (async).")
        print(f"  Durable time: {durable_s:.2f}s")
        print(f"  Total chunks upserted: {total_docs}")
        print(f"  Files processed: {processed}, errors: {errors}")
        print(f"  MongoDB: {settings.mongodb_db}.{settings.mongodb_collection}")

    asyncio.run(_run())


if __name__ == "__main__":
    import os
    import sys

    # Parse: python main.py [dev|qa|prod] [local|remote] [pattern] [--force] [--async]
    COLLECTIONS = {"dev": "collection_taixingbi_dev", "qa": "collection_taixingbi_qa", "prod": "collection_taixingbi_prod"}
    TARGETS = ("local", "remote")
    use_async = "--async" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--force", "--async")]
    skip_unchanged = "--force" not in sys.argv

    # 1) env: dev | qa | prod
    env_arg = args[0] if args and args[0] in COLLECTIONS else None
    if env_arg:
        os.environ["MONGODB_COLLECTION"] = COLLECTIONS[env_arg]
        args = args[1:]

    # 2) target: local | remote (pick which MongoDB)
    target_arg = args[0] if args and args[0] in TARGETS else None
    if target_arg == "local":
        os.environ["MONGODB_URI"] = os.environ.get("MONGODB_URI_LOCAL", "mongodb://localhost:27017")
        args = args[1:]
    elif target_arg == "remote":
        args = args[1:]

    pattern = args[0] if args else "data/**/*"

    if use_async:
        ingest_folder_async(pattern, skip_unchanged=skip_unchanged)
    else:
        ingest_folder(pattern, skip_unchanged=skip_unchanged)
