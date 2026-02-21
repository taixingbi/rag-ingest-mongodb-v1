"""Document normalization: convert various file formats to stable text."""

import json
import os
from typing import Any, Dict, Optional, Tuple

from utils import stable_json_text


def detect_file_type(filepath: str) -> str:
    """Detect file type from extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        return "json"
    elif ext in [".md", ".markdown"]:
        return "md"
    elif ext == ".txt":
        return "txt"
    elif ext == ".pdf":
        return "pdf"
    else:
        return "unknown"


def normalize_json(filepath: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Normalize JSON file to stable text.
    Returns: (normalized_text, metadata_dict)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        obj = json.load(f)
    
    text = stable_json_text(obj)
    
    # Extract metadata if available
    metadata = {}
    if isinstance(obj, dict):
        if "metadata" in obj:
            metadata = obj["metadata"]
        elif "profile" in obj:
            # Try to extract useful metadata from profile structure
            profile = obj.get("profile", {})
            if isinstance(profile, dict):
                metadata = {
                    "title": profile.get("name") or profile.get("title", "Profile"),
                }
    
    return text, metadata if metadata else None


def normalize_markdown(filepath: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Normalize Markdown file to text.
    Returns: (normalized_text, metadata_dict)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Extract title from first heading if available
    metadata = {}
    lines = text.split("\n")
    for line in lines[:10]:  # Check first 10 lines
        if line.startswith("# "):
            metadata["title"] = line[2:].strip()
            break
    
    return text, metadata if metadata else None


def normalize_text(filepath: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Normalize plain text file.
    Returns: (normalized_text, metadata_dict)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text, None


def normalize_pdf(filepath: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Normalize PDF file to text.
    Requires: pip install pdfplumber or PyPDF2
    
    Returns: (normalized_text, metadata_dict)
    """
    text_parts = []
    metadata = {}
    
    # Try pdfplumber first (better text extraction)
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
            if pdf.metadata and pdf.metadata.get("Title"):
                metadata["title"] = pdf.metadata["Title"]
    except ImportError:
        # Fallback to PyPDF2
        try:
            import PyPDF2
            with open(filepath, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text_parts.append(page.extract_text() or "")
                if pdf_reader.metadata and pdf_reader.metadata.get("/Title"):
                    metadata["title"] = pdf_reader.metadata["/Title"]
        except ImportError:
            raise ImportError(
                "PDF support requires either 'pdfplumber' or 'PyPDF2'. "
                "Install with: pip install pdfplumber"
            )
    
    text = "\n\n".join(text_parts)
    return text, metadata if metadata else None


def normalize_document(filepath: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Normalize document based on file type.
    Returns: (normalized_text, metadata_dict)
    """
    file_type = detect_file_type(filepath)
    
    if file_type == "json":
        return normalize_json(filepath)
    elif file_type == "md":
        return normalize_markdown(filepath)
    elif file_type == "txt":
        return normalize_text(filepath)
    elif file_type == "pdf":
        return normalize_pdf(filepath)
    else:
        raise ValueError(f"Unsupported file type: {file_type} ({filepath})")
