# RAG Ingest - MongoDB Atlas Vector Search

Local ingestion pipeline that reads files (JSON/MD/PDF), chunks them, computes embeddings, and upserts into MongoDB Atlas Vector Search.

## Architecture

- **Mac mini (local)**: Reads files → chunks → computes embeddings (OpenAI) → upserts to Atlas
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
OPENAI_API_KEY="sk-..."
OPENAI_EMBED_MODEL="text-embedding-3-small"  # or text-embedding-3-large
CHUNK_TOKENS=1000
OVERLAP_TOKENS=150
BATCH_SIZE=64
```

## Usage

```bash
# Ingest to dev collection on remote (Atlas)
python main.py dev remote

# Ingest to dev collection on local MongoDB (Mac)
python main.py dev local

# Other envs: qa, prod
python main.py qa remote
python main.py prod local

# Optional: file pattern and --force
python main.py dev remote "data/**/*.json"
python main.py dev local --force
```

Order: `[dev|qa|prod] [local|remote] [pattern] [--force]`

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
