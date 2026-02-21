"""Shared utilities for ingest."""

import hashlib
import json
import os
import time
from typing import Any


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
