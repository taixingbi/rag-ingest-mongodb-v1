import asyncio
import time
from typing import List
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIError

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