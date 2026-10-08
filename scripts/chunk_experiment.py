"""Compare chunking configurations on the same PDF document."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import replace
from pathlib import Path

from src.core.settings import load_settings
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.loader.pdf_loader import PdfLoader


CASES = ((500, 100), (1000, 200), (1500, 300))


def compact_preview(text: str, limit: int = 100) -> str:
    """Collapse whitespace so chunk boundaries are easy to inspect."""
    return " ".join(text.split())[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="?",
        default="tests/fixtures/sample_documents/chinese_long_doc.pdf",
        help="PDF path to evaluate",
    )
    args = parser.parse_args()

    document = PdfLoader(extract_images=False).load(Path(args.pdf))
    settings = load_settings()
    if settings.ingestion is None:
        raise RuntimeError("Missing ingestion settings")

    print(f"Document: {document.metadata.get('title', document.id)}")
    print(f"Characters: {len(document.text)}")
    print()
    print("size/overlap | chunks | min | avg | median | max")
    print("-------------|--------|-----|-----|--------|----")

    for chunk_size, chunk_overlap in CASES:
        ingestion = replace(
            settings.ingestion,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        case_settings = replace(settings, ingestion=ingestion)
        chunks = DocumentChunker(case_settings).split_document(document)
        lengths = [len(chunk.text) for chunk in chunks]
        shortest_index = min(range(len(chunks)), key=lambda index: lengths[index])

        print(
            f"{chunk_size:>4}/{chunk_overlap:<7} | "
            f"{len(chunks):>6} | "
            f"{min(lengths):>3} | "
            f"{statistics.mean(lengths):>3.0f} | "
            f"{statistics.median(lengths):>6.0f} | "
            f"{max(lengths):>3}"
        )
        print(
            f"  shortest #{shortest_index}: "
            f"{compact_preview(chunks[shortest_index].text)}"
        )


if __name__ == "__main__":
    main()
