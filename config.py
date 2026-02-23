"""
Configuration loaded from environment (.env).
Values are read when Settings() is created so CLI args (--env, --target localhost|atlas) take effect.
"""

from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: str) -> int:
    return int(os.environ.get(key, default))


@dataclass
class Settings:
    mongodb_uri: str = field(default_factory=lambda: _env("MONGODB_URI", ""))
    mongodb_db: str = field(default_factory=lambda: _env("MONGODB_DB", "rag"))
    mongodb_collection: str = field(default_factory=lambda: _env("MONGODB_COLLECTION", "rag_chunks"))

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    # Embed: "openai" or "sentence_transformers"
    embed_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "openai"))
    embed_model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "")
        or ("BAAI/bge-small-en-v1.5" if _env("EMBED_PROVIDER", "openai") == "sentence_transformers" else "text-embedding-3-small")
    )

    # Chunking (token-based preferred, char-based fallback)
    chunk_tokens: int = field(default_factory=lambda: _env_int("CHUNK_TOKENS", "1000"))
    overlap_tokens: int = field(default_factory=lambda: _env_int("OVERLAP_TOKENS", "150"))
    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", "5000"))
    overlap_chars: int = field(default_factory=lambda: _env_int("OVERLAP_CHARS", "800"))

    # Ingest (batch size for embeddings API; max throughput defaults)
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", "128"))
    # Local (sentence_transformers) only: larger batches = lower latency (no API limit). Default 256.
    embed_batch_size_local: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE_LOCAL", "256"))
    max_concurrent_files: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_FILES", "6"))
    block_queue_size: int = field(default_factory=lambda: _env_int("BLOCK_QUEUE_SIZE", "4"))
    embed_max_concurrent: int = field(default_factory=lambda: _env_int("EMBED_MAX_CONCURRENT", "8"))

    # Device for sentence_transformers: "cuda", "mps", "cpu" (default auto)
    embed_device: str = field(default_factory=lambda: _env("EMBED_DEVICE", ""))

    # RabbitMQ (optional: use with --async --queue)
    amqp_url: str = field(default_factory=lambda: _env("AMQP_URL", "amqp://guest:guest@localhost:5672/"))
    rabbitmq_queue: str = field(default_factory=lambda: _env("RABBITMQ_QUEUE", "rag.ingest.tasks"))
