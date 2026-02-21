import asyncio
import time
from typing import Any, List
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIError

# ----------------------------
# Token bucket (TPM rate limiting) — required to avoid 429 tokens-per-minute
# ----------------------------
# RATE = refill (tokens/sec). CAPACITY = max tokens in bucket; must be >= largest
# single request so we never deadlock. We cap at CAPACITY, refill at RATE.

TPM_LIMIT = 1_000_000
RATE = (TPM_LIMIT * 0.7) / 60  # tokens per second, stay below TPM limit
CAPACITY = RATE * 10  # burst up to ~10 sec worth; must be >= max_request_tokens

token_budget = CAPACITY
last_refill = time.monotonic()
_lock = asyncio.Lock()


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, int(len(text) / 4))


async def acquire_tokens(cost: int) -> None:
    """Wait until the token bucket has at least `cost` tokens, then spend them.
    Refill rate is RATE; bucket capacity is CAPACITY (>= cost for any single request).
    """
    global token_budget, last_refill
    if cost <= 0:
        return
    if cost > CAPACITY:
        raise ValueError(
            f"acquire_tokens(cost={cost}) exceeds bucket CAPACITY={CAPACITY}; "
            "increase CAPACITY or use smaller batches."
        )

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

def embed_texts_sentence_transformers(
    model: Any, texts: List[str], batch_size: int = 64
) -> List[List[float]]:
    """
    Embed texts using a local SentenceTransformer model (e.g. BAAI/bge-small-en-v1.5).
    Single encode() call with internal batching for better GPU/CPU utilization.
    Install: pip install sentence-transformers
    """
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [emb.tolist() for emb in embeddings]


async def embed_texts_sentence_transformers_async(
    model: Any, texts: List[str], batch_size: int = 64
) -> List[List[float]]:
    """Run SentenceTransformer encode in thread pool (model is sync-only)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: embed_texts_sentence_transformers(model, texts, batch_size),
    )