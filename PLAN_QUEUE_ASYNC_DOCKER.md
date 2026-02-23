ere’s a plan for queue (Redis Streams) + async for your RAG ingest.
Plan: Queue (Redis Streams) + Async
1. Why Redis Streams
Durable: Messages stay in Redis (optional TTL). Survives restarts if Redis is persistent.
Consumer groups: Multiple workers share the stream; each message is handled by one consumer; automatic tracking of pending/ack (XACK).
At-least-once: Unacked messages can be re-read (e.g. XPENDING + XCLAIM or XREADGROUP from last id). Good for retries.
Single broker: Same Redis you might use for cache/session; no separate queue server like RabbitMQ.
2. Dependencies
requirements.txt: add redis (has async support: redis.asyncio or from redis.asyncio import Redis in 4.2+). Or aioredis if you prefer (older async-first client). Prefer redis>=4.5 with redis.asyncio.
Keep: motor, openai (AsyncOpenAI).
3. Stream design
Stream name: e.g. rag:ingest:tasks.
Message: one field per task, e.g. {"filepath": "/app/data/foo.json"} or filepath + optional source_id. Workers need the path (and shared volume in Docker so path is valid in worker).
Consumer group: e.g. rag-workers, created once (XGROUP CREATE stream rag-workers 0 MKSTREAM). Consumers use XREADGROUP GROUP rag-workers worker-1 (worker-2, …) to read.
Ack: XACK after successful process. Pending entries can be reclaimed after a timeout for retry (optional).
Finish: Producer adds a “sentinel” message (e.g. type=done) or tracks “files enqueued” and workers exit when stream is empty and no pending (or after sentinel). Alternatively: producer uses stream length / consumer lag to decide when to call save_state and exit.
4. Components
Component	Responsibility
Producer	Connect to Redis, ensure stream + consumer group exist. Glob + skip unchanged (using state). For each file to process: XADD rag:ingest:tasks * filepath <path>. Optionally send sentinel: XADD rag:ingest:tasks * type done. Then either exit (workers run separately) or wait for completion (see below).
Worker(s)	Async loop: XREADGROUP GROUP rag-workers worker-<id> BLOCK 5000 STREAMS rag:ingest:tasks >. For each message: if type==done, XACK and optionally signal “all producers done”; else parse filepath, run normalize → chunk → embed_async → delete_by_source + upsert_async, update in-memory state, XACK. Run N workers (same process or multiple containers).
State	Same idea as before: workers update shared in-memory state; one place calls save_state(state) at the end. Single-process: one program starts producer then N worker coroutines; after producer finishes, wait until stream is empty (or sentinel consumed) and no pending, then save_state(state) and exit. Multi-process: need a way to know “all work done” (e.g. sentinel + pending count in Redis) and have one process write state.
5. Redis connection
URL: REDIS_URL=redis://localhost:6379/0 or redis://:password@host:6379/0.
Config: config.py add redis_url, redis_stream (e.g. rag:ingest:tasks), redis_consumer_group (e.g. rag-workers), max_concurrent_files (worker concurrency).
6. Docker
Services:
redis: official redis:7-alpine (or 6). Optional: persistence with a volume for /data. Expose 6379.
rag-ingest: your app; connect to redis://redis:6379/0.
Single-process run: one command, e.g. python main.py ingest --env dev --target atlas --mode async. Process: create stream/group if needed, start N worker tasks (XREADGROUP loop), run producer (glob → XADD), then wait until stream empty + no pending (or sentinel consumed), then save_state(state) and exit.
Compose: docker-compose.yml with redis + rag-ingest; rag-ingest depends_on redis, env REDIS_URL=redis://redis:6379/0. Volumes for data and state as before.
7. Error handling and retry
Worker crashes before XACK: message stays in “pending”. Use XPENDING to list, then XCLAIM by another worker after idle time (e.g. 60s) to retry. Optionally cap retries with a retries field in the message and DLQ or delete after N claims.
On fatal error (e.g. file missing): XACK and skip (or XADD to a “dead letter” stream) so the stream keeps moving.
8. Order of work
Add redis (async) to requirements; add Redis to Compose and REDIS_* to config.
Implement: ensure stream + consumer group (XGROUP CREATE … MKSTREAM); producer (XADD); worker loop (XREADGROUP → process → XACK). Reuse existing async embed + Motor DB inside worker.
Wire “async” path in main: when --async, use Redis Streams producer + workers; after producer done, wait for completion then save_state(state).
Optional: pending reclaim (XCLAIM) for retries; sentinel message for clean shutdown.
9. Files to add/change (summary)
PLAN_QUEUE_ASYNC_DOCKER.md: switch “asyncio.Queue” / RabbitMQ to “Redis Streams”; add stream name, consumer group, producer/worker roles; add Redis service and REDIS_* env; note single-process vs multi-process.
requirements.txt: add redis>=4.5 (async).
config.py: add redis_url, redis_stream, redis_consumer_group.
main.py (or queue_redis.py): producer (XADD file paths), workers (XREADGROUP, process, XACK), combined “run producer then workers and save_state when done” for single-process.
docker-compose.yml: add redis service; set REDIS_URL for rag-ingest.
Summary: Use one Redis stream (rag:ingest:tasks) and one consumer group (rag-workers). Producer XADDs file paths; N async workers XREADGROUP, process with existing async embed + Motor, XACK, and update in-memory state; one place calls save_state when all work is done. Same async pipeline as before, only the queue is Redis Streams instead of in-memory or RabbitMQ.
If you want this written into PLAN_QUEUE_ASYNC_DOCKER.md (replacing or alongside the current queue section), switch to Agent mode and ask to update the plan to use Redis Streams