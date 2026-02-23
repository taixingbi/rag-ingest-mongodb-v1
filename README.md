# RAG Ingest - MongoDB Atlas Vector Search

Local ingestion pipeline that reads files (JSON/MD/PDF), chunks them, computes embeddings, and upserts into MongoDB Atlas Vector Search.

## Architecture

- **Mac mini (local)**: Reads files → chunks → computes embeddings (OpenAI or sentence_transformers) → upserts to Atlas
- **MongoDB Atlas**: Stores text + metadata + embedding vector with Vector Search index


## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt

# Optional: For PDF support
pip install pdfplumber
```

## Configuration (.env)

```bash
MONGODB_URI="mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority"
MONGODB_DB="rag"
MONGODB_COLLECTION="rag_chunks"

# Embed: openai (TPM-limited) or sentence_transformers (local, no TPM limit)
EMBED_PROVIDER=openai
OPENAI_API_KEY="sk-..."
OPENAI_EMBED_MODEL="text-embedding-3-small"  # or text-embedding-3-large

# For local embeddings (no TPM limit, lower latency): EMBED_PROVIDER=sentence_transformers, EMBED_MODEL=BAAI/bge-small-en-v1.5
# EMBED_BATCH_SIZE_LOCAL=256   # larger = faster encode (sentence_transformers only)
# EMBED_DEVICE=cuda            # or mps, cpu (sentence_transformers only; default auto)

CHUNK_TOKENS=1000
OVERLAP_TOKENS=150
BATCH_SIZE=128
EMBED_MAX_CONCURRENT=8
MAX_CONCURRENT_FILES=6
# Token bucket (OpenAI only; ignored when EMBED_PROVIDER=sentence_transformers)
EMBED_TPM_SAFETY=0.9
# OPENAI_TPM_LIMIT=1000000   # set higher if your account has more TPM
```

## Docker (docker-compose)

Requires a `.env` in the project root (see Configuration). Put input files under `./data` on the host; they are mounted at `/data` in the container.

**MongoDB from Docker:** The default command uses `--target atlas`, so `MONGODB_URI` in `.env` must be your **Atlas** connection string (`mongodb+srv://...`). If it is `mongodb://localhost:27017`, the container will try to reach MongoDB inside the container and get "Connection refused". To use MongoDB running on your host Mac from inside the container, use `--target localhost` and set in `.env`: `MONGODB_URI_LOCAL=mongodb://host.docker.internal:27017`.

**One-off ingest (recommended)** — uses default command with `/data`, `--force`, RabbitMQ:

```bash
docker-compose run --rm rag-ingest
```

Run this **exact** one-liner (nothing after `rag-ingest`). Pasting a multi-line command without `\` at the end of each line will make zsh run the next line as a new command (`command not found: --target`). Adding flags like `--force` after `rag-ingest` replaces the whole command and can cause "unrecognized arguments".

**Custom command** — if you override, use `--input-dir /data` (path inside the container):

```bash
# With RabbitMQ queue (async workers)
docker-compose run --rm rag-ingest python main.py ingest \
  --input-dir /data --pattern "**/*" \
  --env dev --target localhost --mode async --queue rabbitmq --force

# Without queue (sync or in-process async)
docker-compose run --rm rag-ingest python main.py ingest \
  --input-dir /data --env dev --target atlas --force
```

**Start RabbitMQ only** (run the app on the host, use container only for the broker)

```bash
docker-compose up -d rabbitmq
```

- AMQP: `localhost:5672`
- Management UI: http://localhost:15672 (login `guest` / `guest`)

## Usage
Options:
`--env` (dev|qa|prod), default: dev.
`--target` (localhost|atlas), which MongoDB to write to.
`--input-dir` (default: ./data), directory to glob for files.
`--pattern` (default: **/*), glob under input-dir, e.g. *.json.
`--mode` (sync|async), default: async.
`--queue` (none|memory|redis|rabbitmq), default: memory (in-process; rabbitmq = external broker).
`--workers` (default: 4), number of workers when queue=rabbitmq.
`--max-inflight` (default: 128), max in-flight tasks.
`--batch-size` (default: 64), embedding batch size.
`--embedder` (default: sentence-transformers), openai or sentence-transformers.
`--force`, re-ingest all files (ignore state).
`--resume`, resume from state (if supported).
`--dry-run`, don't write to DB (if supported).

python main.py ingest \
  --env dev \
  --target atlas \
  --mode async \
  --queue memory \
  --workers 4 \
  --batch-size 64 \
  --embedder sentence-transformers \
  --input-dir ./data \
  --pattern "*.json" \
  --force 

python main.py ingest \
  --env dev \
  --target localhost \
  --mode async \
  --queue redis \
  --workers 4 \
  --batch-size 64 \
  --embedder sentence-transformers \
  --input-dir ./data \
  --pattern "*.json" \
  --force 

docker-compose run --rm rag-ingest python main.py ingest \
  --env dev \
  --target atlas \
  --mode async \
  --queue redis \
  --workers 4 \
  --batch-size 64 \
  --embedder sentence-transformers \
  --input-dir ./data \
  --pattern "*.json" \
  --force 

docker-compose run --rm rag-ingest python main.py ingest \
  --env dev \
  --target atlas \
  --mode async \
  --queue rabbitmq \
  --workers 4 \
  --batch-size 64 \
  --embedder sentence-transformers \
  --input-dir ./data \
  --pattern "*.json" \
  --force

docker-compose run --rm rag-ingest python main.py ingest --env dev --target atlas --mode async --queue redis --workers 4 --batch-size 64 --embedder sentence-transformers --input-dir /data --pattern "*.json" --force

docker-compose run --rm rag-ingest python main.py ingest --env dev --target atlas --mode async --queue rabbitmq --workers 4 --batch-size 64 --embedder sentence-transformers --input-dir /data --pattern "*.json" --force

docker-compose run python main.py ingest --env dev --target atlas --mode async --queue rabbitmq --workers 4 --batch-size 64 --embedder sentence-transformers --input-dir /data --pattern "*.json" --force
## Data Model

Collection: `rag.rag_chunks`

Each chunk document:
```json
{
  "_id": "sha256(source_id + chunk_id + content_hash)",
  "chunk_id": "profile.json::chunk_0003",
  "source": {
    "source_id": "profile.json",
    "path": "data/profile.json",
    "type": "json",
    "mtime": "2026-02-19T12:00:00Z"
  },
  "text": "chunk text...",
  "metadata": {
    "title": "Profile",
    "section": "chunk_0",
    "tags": ["profile", "resume", "candidate"],
    "lang": "en"
  },
  "embedding": [0.0123, ...],
  "embedding_model": "text-embedding-3-small",
  "dims": 1536,
  "created_at": "2026-02-19T12:01:00Z",
  "updated_at": "2026-02-19T12:01:00Z"
}
```

## MongoDB Atlas Vector Search Index

After ingestion, create a Vector Search index in Atlas UI:

1. Go to Atlas → Search → Create Search Index
2. Select "JSON Editor"
3. Configure:
```json
{
  "fields": [
    {
      "type": "knnVector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "string",
      "path": "source.source_id"
    },
    {
      "type": "string",
      "path": "metadata.tags"
    },
    {
      "type": "string",
      "path": "text"
    }
  ]
}
```

## Incremental Ingestion

The pipeline uses `state.json` to track file hashes and modification times. Files that haven't changed are automatically skipped. Delete `state.json` to reset or use `--force` flag to re-ingest everything.

## Supported File Types

- **JSON**: Automatically normalized (sorted keys, stable format)
- **Markdown (.md)**: Text extracted as-is
- **Text (.txt)**: Plain text files
- **PDF**: Requires `pdfplumber` package

## MongoDB Collections

Update collection name in `.env`:
```
MONGODB_COLLECTION=collection_taixingbi_dev
MONGODB_COLLECTION=collection_taixingbi_qa
MONGODB_COLLECTION=collection_taixingbi_prod
```

## Links

- [MongoDB Atlas Dashboard](https://cloud.mongodb.com/v2/5f8d901d427b1f41a5daf2c0#/explorer/6994e45919851ad449223e8a/db_hunt/collection_taixingbi_dev/find)
