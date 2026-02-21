Producer → asyncio.Queue (block_queue_size)
              ↓
Workers (many)
              ↓
TokenLimiter (TPM shaping)   ← REQUIRED (avoids 429 TPM; token bucket in embed.py)
              ↓
Semaphore (connection cap)
              ↓
OpenAI Embedding API