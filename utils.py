"""Shared utilities for ingest."""

import hashlib
import io
import json
import os
import time
from typing import Any, Iterator, List, Union


def read_lines_in_blocks(
    filepath_or_file: Union[str, io.TextIOWrapper],
    block_size: int = 64,
    encoding: str = "utf-8",
) -> Iterator[List[str]]:
    """Read a file in blocks of lines. Yields lists of non-empty stripped lines."""
    if isinstance(filepath_or_file, str):
        f = open(filepath_or_file, "r", encoding=encoding)
        should_close = True
    else:
        f = filepath_or_file
        should_close = False
    try:
        block: List[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            block.append(line)
            if len(block) >= block_size:
                yield block
                block = []
        if block:
            yield block
    finally:
        if should_close:
            f.close()


def stable_json_text(obj: Any) -> str:
    """Stable deterministic JSON -> text for embedding + BM25."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)


def sha256_text(s: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """Get current UTC time in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_file_mtime(filepath: str) -> str:
    """Get file modification time in ISO format."""
    mtime = os.path.getmtime(filepath)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime))


def compute_stable_id(source_id: str, chunk_id: str, content_hash: str) -> str:
    """Compute stable _id for MongoDB document."""
    combined = f"{source_id}::{chunk_id}::{content_hash}"
    return sha256_text(combined)
