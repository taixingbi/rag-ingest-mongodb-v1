"""
RabbitMQ queue for RAG ingest: producer publishes file paths, workers consume and process.
Use with: python main.py ingest --env dev --target localhost --mode async --queue rabbitmq
Requires: AMQP_URL, optional RABBITMQ_QUEUE (default rag.ingest.tasks).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Callable, List

import aio_pika
from aio_pika import DeliveryMode, Message

SENTINEL_TYPE = "done"


async def ensure_queue(channel: aio_pika.Channel, queue_name: str) -> aio_pika.Queue:
    """Declare durable queue; idempotent."""
    return await channel.declare_queue(queue_name, durable=True)


async def publish_tasks(
    channel: aio_pika.Channel,
    queue_name: str,
    filepaths: List[str],
    send_sentinel: bool = True,
    sentinel_count: int = 1,
) -> None:
    """Publish filepath messages and optional sentinels (one per worker so all workers exit)."""
    for filepath in filepaths:
        body = json.dumps({"filepath": filepath}).encode()
        message = Message(body=body, delivery_mode=DeliveryMode.PERSISTENT)
        await channel.default_exchange.publish(message, routing_key=queue_name)
    if send_sentinel:
        sentinel_body = json.dumps({"type": SENTINEL_TYPE}).encode()
        for _ in range(sentinel_count):
            message = Message(body=sentinel_body, delivery_mode=DeliveryMode.PERSISTENT)
            await channel.default_exchange.publish(message, routing_key=queue_name)


async def run_worker(
    connection: aio_pika.Connection,
    queue_name: str,
    worker_id: str,
    process_fn: Callable[[str], Any],
    done_flag: asyncio.Event,
) -> None:
    """
    Consume messages; for each filepath call process_fn(filepath); on sentinel set done_flag and exit.
    process_fn is async (filepath: str) -> None.
    """
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await ensure_queue(channel, queue_name)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process(ignore_processed=True):
                try:
                    body = json.loads(message.body.decode())
                except Exception:
                    continue
                if body.get("type") == SENTINEL_TYPE:
                    done_flag.set()
                    return
                filepath = body.get("filepath")
                if not filepath or not isinstance(filepath, str):
                    continue
                try:
                    await process_fn(filepath)
                except Exception as e:
                    print(f"  [worker-{worker_id}] ✗ {filepath}: {e}")
                    # still ack so we don't block the queue; optional: publish to DLQ
                    raise


def ingest_via_rabbitmq(
    folder_glob: str = "data/**/*",
    skip_unchanged: bool = True,
    num_workers: int = 3,
) -> None:
    """
    Ingest using RabbitMQ: producer publishes file paths, N workers consume and process.
    Reuses async embed + Motor; saves state when all work is done.
    """
    from config import Settings
    from state import load_state, save_state, should_skip_file, update_file_state
    from utils import get_files_for_ingest, get_file_mtime, sha256_text
    from normalize import normalize_document

    settings = Settings()
    assert settings.mongodb_uri, "Missing MONGODB_URI"
    if settings.embed_provider == "openai":
        assert settings.openai_api_key, "Missing OPENAI_API_KEY"

    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from openai import AsyncOpenAI
    except ImportError as e:
        raise RuntimeError("Need motor and openai for RabbitMQ ingest") from e

    async def _run() -> None:
        start = time.perf_counter()
        connection = await aio_pika.connect_robust(settings.amqp_url)
        channel = await connection.channel()
        await ensure_queue(channel, settings.rabbitmq_queue)
        await channel.close()

        mongo = AsyncIOMotorClient(settings.mongodb_uri)
        db = mongo[settings.mongodb_db]
        col = db[settings.mongodb_collection]

        from db import async_ensure_unique_index
        await async_ensure_unique_index(col)

        if settings.embed_provider == "openai":
            embed_client = AsyncOpenAI(api_key=settings.openai_api_key)
            print(f"Embed: openai ({settings.embed_model})")
        else:
            from sentence_transformers import SentenceTransformer
            embed_client = SentenceTransformer(
                settings.embed_model,
                device=settings.embed_device or None,
            )
            print(f"Embed: sentence_transformers ({settings.embed_model})")

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
        print("Streaming progress: printing every 640 chunks during embedding (not only after mongo batches).", flush=True)
        if not to_process:
            if skip_unchanged:
                save_state(state)
            print(f"Nothing to do.")
            await connection.close()
            return

        from main import _process_one_file_async

        total_docs = 0
        processed = 0
        errors = 0
        lock = asyncio.Lock()

        async def process_file(filepath: str) -> None:
            nonlocal total_docs, processed, errors
            try:
                t0 = time.perf_counter()
                n, _, timings = await _process_one_file_async(
                    filepath, col, embed_client, settings, state
                )
                elapsed = time.perf_counter() - t0
                async with lock:
                    total_docs += n
                    processed += 1
                print(f"  ✓ {filepath} -> {n} chunks ({elapsed:.2f}s)", flush=True)
                if timings:
                    lp, ch, em, mb = timings["load_parse"], timings["chunk"], timings["embed"], timings["mongo_bulk_write"]
                    print(f"  total summary: load_parse={lp:.3f}s chunk={ch:.3f}s embed={em:.3f}s mongo_bulk_write={mb:.3f}s chunks={n}", flush=True)
            except Exception as e:
                async with lock:
                    errors += 1
                print(f"  ✗ {filepath}: {e}")
                # ack so queue keeps moving; optional: publish to DLQ

        done_flag = asyncio.Event()
        n_workers = min(num_workers, settings.max_concurrent_files)
        workers = [
            asyncio.create_task(
                run_worker(
                    connection,
                    settings.rabbitmq_queue,
                    str(i),
                    process_file,
                    done_flag,
                )
            )
            for i in range(n_workers)
        ]

        channel = await connection.channel()
        await publish_tasks(
            channel,
            settings.rabbitmq_queue,
            to_process,
            send_sentinel=True,
            sentinel_count=n_workers,
        )
        await channel.close()

        await asyncio.gather(*workers)

        if skip_unchanged:
            save_state(state)

        await connection.close()
        durable_s = time.perf_counter() - start
        print(f"\nDone (RabbitMQ).")
        print(f"  Durable time: {durable_s:.2f}s")
        print(f"  Total chunks: {total_docs}, processed: {processed}, errors: {errors}")
        print(f"  MongoDB: {settings.mongodb_db}.{settings.mongodb_collection}")

    asyncio.run(_run())
