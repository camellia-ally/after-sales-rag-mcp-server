"""Plain text and Markdown loader for the ingestion pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """Load UTF-8 ``.txt`` and ``.md`` files as standardized documents."""

    SUPPORTED_SUFFIXES = {".txt", ".md"}

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported text file type: {suffix}")

        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        if not text.strip():
            raise ValueError("Document content is empty")

        digest = hashlib.sha256(raw).hexdigest()
        title = path.stem
        for line in text.splitlines():
            candidate = line.strip().lstrip("#").strip()
            if candidate:
                title = candidate[:200]
                break

        return Document(
            id=f"doc_{digest[:16]}",
            text=text,
            metadata={
                "source_path": str(path),
                "doc_type": "markdown" if suffix == ".md" else "text",
                "doc_hash": digest,
                "title": title,
            },
        )
