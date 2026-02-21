"""Text chunking: token-based (tiktoken) with char fallback."""

from typing import List

try:
    import tiktoken  # optional
except Exception:
    tiktoken = None


def chunk_text_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    model: str,
    chunk_chars: int = 5000,
    overlap_chars: int = 800,
) -> List[str]:
    """
    Token-based chunking with tiktoken. Falls back to char chunking if tiktoken is missing.
    """
    if tiktoken is None:
        return chunk_text_chars(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    toks = enc.encode(text)
    if not toks:
        return []

    chunks: List[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(toks), step):
        end = min(len(toks), start + chunk_tokens)
        chunk = enc.decode(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(toks):
            break
    return chunks


def chunk_text_chars(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    step = max(1, chunk_chars - overlap_chars)
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks
