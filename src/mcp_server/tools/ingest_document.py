"""MCP tool that sends uploaded documents through the native RAG pipeline."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from mcp import types

logger = logging.getLogger(__name__)

TOOL_NAME = "ingest_document"
TOOL_DESCRIPTION = (
    "Ingest a PDF, UTF-8 text, or Markdown document into a named collection. "
    "Document parsing, chunking, embedding, and storage are performed by MODULAR-RAG."
)
TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {"type": "string", "description": "Original filename, including extension"},
        "content_base64": {"type": "string", "description": "Base64-encoded document bytes"},
        "collection": {"type": "string", "description": "Destination collection name"},
        "force": {"type": "boolean", "description": "Reprocess a previously ingested file"},
    },
    "required": ["filename", "content_base64", "collection"],
    "additionalProperties": False,
}
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _run_pipeline(file_path: Path, collection: str, force: bool) -> dict[str, Any]:
    """Run the synchronous pipeline outside the MCP event loop."""
    from src.core.settings import load_settings
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(load_settings(), collection=collection, force=force)
    try:
        result = pipeline.run(str(file_path))
        return result.to_dict()
    finally:
        pipeline.close()


async def ingest_document_handler(
    filename: str,
    content_base64: str,
    collection: str,
    force: bool = False,
) -> types.CallToolResult:
    """Validate and process an uploaded document through the RAG pipeline."""
    try:
        if not filename or not filename.strip():
            raise ValueError("filename must not be empty")
        if not collection or not collection.strip():
            raise ValueError("collection must not be empty")
        safe_name = PurePosixPath(filename.replace("\\", "/")).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("Supported document types are .pdf, .txt, and .md")
        # Bound encoded data before allocating its decoded copy.
        if len(content_base64) > ((MAX_DOCUMENT_BYTES + 2) // 3) * 4 + 8:
            raise ValueError("Document exceeds the 20 MB limit")
        try:
            content = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("content_base64 is invalid") from exc
        if not content:
            raise ValueError("Document content is empty")
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("Document exceeds the 20 MB limit")

        # Keep the temporary source under the RAG repository so relative paths
        # and the pipeline's own data roots remain consistent.
        staging_dir = Path(__file__).resolve().parents[3] / "data" / "tmp" / "mcp-ingestion"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staging_dir / f"{uuid.uuid4().hex}_{safe_name}"
        try:
            staged_path.write_bytes(content)
            result = await asyncio.to_thread(
                _run_pipeline, staged_path, collection.strip(), force,
            )
        finally:
            staged_path.unlink(missing_ok=True)

        result.update({"filename": safe_name, "collection": collection.strip()})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
            structuredContent=result,
            isError=not bool(result.get("success")),
        )
    except ValueError as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=str(exc))],
            isError=True,
        )
    except Exception:
        logger.exception("ingest_document failed")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="内部错误：文档导入失败，请检查 RAG 服务日志")],
            isError=True,
        )


def register_tool(protocol_handler: Any) -> None:
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=ingest_document_handler,
    )
