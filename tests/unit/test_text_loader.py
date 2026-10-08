"""Tests for UTF-8 text/Markdown support in the ingestion pipeline."""

from src.libs.loader.text_loader import TextLoader


def test_text_loader_extracts_content_and_heading_title(tmp_path):
    path = tmp_path / "uploaded.md"
    path.write_text("# 退款政策\n\n七天内可以申请退款。", encoding="utf-8")

    document = TextLoader().load(path)

    assert document.text.startswith("# 退款政策")
    assert document.metadata["title"] == "退款政策"
    assert document.metadata["doc_type"] == "markdown"
    assert document.metadata["source_path"] == str(path.resolve())
