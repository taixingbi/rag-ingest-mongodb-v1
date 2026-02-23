import asyncio
import concurrent.futures
import os
import time
from typing import Any, List
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIError

# ----------------------------
# Token bucket (TPM rate limiting) — OpenAI only
# ----------------------------
# Skipped when EMBED_PROVIDER=sentence_transformers (no TPM limit). EMBED_TPM_SAFETY is only
# read when using OpenAI. RATE = refill (tokens/sec); CAPACITY = max tokens in bucket.

_bucket_initialized = False
RATE = 0.0
CAPACITY = 0.0
token_budget = 0.0
last_refill = 0.0
_lock: asyncio.Lock = None  # type: ignore[assignment]


def _init_bucket() -> None:
    """Initialize token bucket from env; only when using OpenAI (first acquire_tokens call)."""
    global RATE, CAPACITY, token_budget, last_refill, _lock, _bucket_initialized
    if _bucket_initialized:
        return
    _bucket_initialized = True
    tpm_limit = int(os.environ.get("OPENAI_TPM_LIMIT", "1000000"))
    tpm_safety = float(os.environ.get("EMBED_TPM_SAFETY", "0.9"))
    RATE = (tpm_limit * tpm_safety) / 60
    CAPACITY = RATE * 25
    token_budget = CAPACITY
    last_refill = time.monotonic()
    _lock = asyncio.Lock()


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, int(len(text) / 4))


async def acquire_tokens(cost: int) -> None:
    """Wait until the token bucket has at least `cost` tokens, then spend them.
    No-op when EMBED_PROVIDER=sentence_transformers (no TPM limit); EMBED_TPM_SAFETY is not used.
    """
    if os.environ.get("EMBED_PROVIDER", "openai").lower() == "sentence_transformers":
        return
    global token_budget, last_refill
    _init_bucket()
    if cost <= 0:
        return
    if cost > CAPACITY:
        raise ValueError(
            f"acquire_tokens(cost={cost}) exceeds bucket CAPACITY={CAPACITY}; "
            "increase CAPACITY or use smaller batches."
        )

    assert _lock is not None
    while True:
        async with _lock:
            now = time.monotonic()
            elapsed = now - last_refill
            token_budget = min(CAPACITY, token_budget + elapsed * RATE)
            last_refill = now

            if token_budget >= cost:
                token_budget -= cost
                return
            need = cost - token_budget
            sleep_sec = min(2.0, max(0.05, need / RATE))

        await asyncio.sleep(sleep_sec)


# ----------------------------
# Embeddings
# ----------------------------

def embed_texts_openai(
    client: OpenAI,
    model: str,
    texts: List[str],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> List[List[float]]:
    """
    Batch embed with exponential backoff retry logic.
    Keep texts reasonably sized (32-128 chunks per request).
    """
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except (RateLimitError, APIError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Embedding API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            # For other errors, don't retry
            raise
    
    raise RuntimeError("Failed to embed texts after retries")


async def embed_texts_openai_async(
    client: AsyncOpenAI,
    model: str,
    texts: List[str],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> List[List[float]]:
    """Async batch embed with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = await client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except (RateLimitError, APIError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Embedding API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
        except Exception as e:
            raise
    raise RuntimeError("Failed to embed texts after retries")


# ----------------------------
# SentenceTransformer (local)
# ----------------------------
# Dedicated executor for async encode so parallel file embeds don't starve the default pool.
_ST_EXECUTOR: concurrent.futures.Executor = None  # type: ignore[assignment]


def _get_st_executor() -> concurrent.futures.Executor:
    global _ST_EXECUTOR
    if _ST_EXECUTOR is None:
        _ST_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(2, (os.cpu_count() or 4)),
            thread_name_prefix="st_embed",
        )
    return _ST_EXECUTOR


def embed_texts_sentence_transformers(
    model: Any, texts: List[str], batch_size: int = 256
) -> List[List[float]]:
    """
    Embed texts using a local SentenceTransformer model (e.g. BAAI/bge-small-en-v1.5).
    Single encode() with internal batching. Use batch_size 256+ for best throughput (no API limit).
    """
    if not texts:
        return []
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [emb.tolist() for emb in embeddings]


async def embed_texts_sentence_transformers_async(
    model: Any, texts: List[str], batch_size: int = 256
) -> List[List[float]]:
    """Run SentenceTransformer encode in dedicated thread pool (model is sync-only)."""
    if not texts:
        return []
    loop = asyncio.get_running_loop()
    executor = _get_st_executor()
    return await loop.run_in_executor(
        executor,
        lambda: embed_texts_sentence_transformers(model, texts, batch_size),
    )