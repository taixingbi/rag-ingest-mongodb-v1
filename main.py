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
import os
import glob
import time
from typing import Any, Dict, List

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
from embed import embed_texts_openai, embed_texts_openai_async
from normalize import detect_file_type, normalize_document
from state import load_state, save_state, should_skip_file, update_file_state
from utils import (
    compute_stable_id,
    get_file_mtime,
    now_iso,
    sha256_text,
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
    embed_client: OpenAI,
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
    
    # Embed in batches
    embeddings: List[List[float]] = []
    for i in range(0, len(chunks), settings.batch_size):
        batch = chunks[i : i + settings.batch_size]
        batch_embeddings = embed_texts_openai(embed_client, settings.embed_model, batch)
        embeddings.extend(batch_embeddings)
    
    # Build MongoDB documents matching target schema
    docs: List[Dict[str, Any]] = []
    dims = len(embeddings[0]) if embeddings else 1536
    
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
    embed_client: AsyncOpenAI,
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
    for i in range(0, len(chunks), settings.batch_size):
        batch = chunks[i : i + settings.batch_size]
        batch_embeddings = await embed_texts_openai_async(
            embed_client, settings.embed_model, batch
        )
        embeddings.extend(batch_embeddings)
    docs = []
    dims = len(embeddings[0]) if embeddings else 1536
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
    assert settings.openai_api_key, "Missing OPENAI_API_KEY"
    
    # MongoDB connection
    mongo = MongoClient(settings.mongodb_uri)
    db = mongo[settings.mongodb_db]
    col = db[settings.mongodb_collection]
    ensure_unique_index(col)
    
    # OpenAI client
    embed_client = OpenAI(api_key=settings.openai_api_key)
    
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
            docs = build_docs_for_file(filepath, embed_client, settings)
            
            if docs:
                source_id = docs[0]["source"]["source_id"]
                deleted = delete_chunks_by_source(col, source_id)
                if deleted:
                    print(f"  Deleted {deleted} old chunks for {source_id}")
                upsert_chunks(col, docs)
                total_docs += len(docs)
                
                # Update state
                if skip_unchanged:
                    text, _ = normalize_document(filepath)
                    content_hash = sha256_text(text)
                    mtime = get_file_mtime(filepath)
                    update_file_state(filepath, content_hash, mtime, state)
                
                print(f"  ✓ Ingested {len(docs)} chunks")
            else:
                print(f"  ⚠ No chunks generated")
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
    print(f"  1. Create Vector Search index in Atlas UI:")
    print(f"     - Field: embedding (knnVector, dims={settings.embed_model.split('-')[-1] if 'small' in settings.embed_model else '1536'})")
    print(f"     - Optional filters: source.source_id, metadata.tags")
    print(f"  2. Optional: Add text index for hybrid search (field: text)")


async def _process_one_file_async(
    filepath: str,
    col: Any,
    embed_client: AsyncOpenAI,
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
    assert settings.openai_api_key, "Missing OPENAI_API_KEY"
    assert AsyncIOMotorClient is not None, "Install motor: pip install motor"

    async def _run() -> None:
        start = time.perf_counter()
        mongo = AsyncIOMotorClient(settings.mongodb_uri)
        db = mongo[settings.mongodb_db]
        col = db[settings.mongodb_collection]
        await async_ensure_unique_index(col)
        embed_client = AsyncOpenAI(api_key=settings.openai_api_key)
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
        total_docs = 0
        processed = 0
        errors = 0

        async def process_with_semaphore(filepath: str) -> None:
            nonlocal total_docs, processed, errors
            async with sem:
                try:
                    n, _ = await _process_one_file_async(
                        filepath, col, embed_client, settings, state
                    )
                    total_docs += n
                    processed += 1
                    print(f"  ✓ {filepath} -> {n} chunks")
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
