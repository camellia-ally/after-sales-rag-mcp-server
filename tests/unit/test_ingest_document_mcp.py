"""Unit tests for document ingestion through the MCP boundary."""

import asyncio
import base64

from src.mcp_server.protocol_handler import create_mcp_server, get_protocol_handler
from src.mcp_server.tools import ingest_document
from src.mcp_server.tools.list_collections import CollectionInfo, ListCollectionsTool


def test_ingest_tool_is_registered_with_named_collection_schema():
    server = create_mcp_server("test-server", "1.0.0")
    schemas = {tool.name: tool for tool in get_protocol_handler(server).get_tool_schemas()}

    assert "ingest_document" in schemas
    assert set(schemas["ingest_document"].inputSchema["required"]) == {
        "filename", "content_base64", "collection",
    }


def test_ingest_document_decodes_bytes_and_calls_pipeline(monkeypatch):
    calls = []

    def fake_pipeline(path, collection, force):
        calls.append((path, collection, force, path.read_bytes()))
        return {"success": True, "chunk_count": 3}

    monkeypatch.setattr(ingest_document, "_run_pipeline", fake_pipeline)
    result = asyncio.run(ingest_document.ingest_document_handler(
        filename="rules.md",
        content_base64=base64.b64encode(b"# Rules\nRefund terms").decode("ascii"),
        collection="policies",
        force=True,
    ))

    assert result.isError is False
    assert result.structuredContent["collection"] == "policies"
    assert result.structuredContent["chunk_count"] == 3
    assert calls[0][1:] == ("policies", True, b"# Rules\nRefund terms")
    assert calls[0][0].exists() is False


def test_ingest_document_rejects_invalid_base64_without_pipeline(monkeypatch):
    monkeypatch.setattr(
        ingest_document, "_run_pipeline",
        lambda *_args: (_ for _ in ()).throw(AssertionError("pipeline must not run")),
    )
    result = asyncio.run(ingest_document.ingest_document_handler(
        filename="rules.txt", content_base64="%%%", collection="policies",
    ))

    assert result.isError is True
    assert "base64" in result.content[0].text


def test_ingest_document_rejects_unsupported_format():
    result = asyncio.run(ingest_document.ingest_document_handler(
        filename="rules.docx", content_base64=base64.b64encode(b"doc").decode(),
        collection="policies",
    ))

    assert result.isError is True
    assert "Supported document types" in result.content[0].text


def test_list_collections_returns_machine_readable_counts(monkeypatch):
    tool = ListCollectionsTool()
    monkeypatch.setattr(
        tool, "list_collections",
        lambda include_stats=True: [CollectionInfo(name="portfolio", count=12)],
    )

    result = asyncio.run(tool.execute(include_stats=True))

    assert result.isError is False
    assert result.structuredContent == {
        "collections": [{"name": "portfolio", "count": 12}],
        "collection_names": ["portfolio"],
    }
