"""Core contract tests for the DashScope rerank provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.libs.reranker.dashscope_reranker import (
    DashScopeRerankError,
    DashScopeReranker,
)


def _settings():
    return SimpleNamespace(
        rerank=SimpleNamespace(
            model="qwen3.7-text-rerank",
            api_key=None,
            base_url="https://example.test/rerank",
            timeout=12,
        ),
        embedding=SimpleNamespace(api_key="test-key"),
        llm=SimpleNamespace(api_key=None),
    )


def test_dashscope_reranker_maps_api_ranking_back_to_candidates():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.35},
            ]
        }
    }
    client = MagicMock()
    client.post.return_value = response
    reranker = DashScopeReranker(_settings(), client=client)

    result = reranker.rerank(
        "什么是混合检索？",
        [
            {"id": "a", "text": "不相关内容", "score": 0.9},
            {"id": "b", "text": "混合检索结合稠密和稀疏检索", "score": 0.8},
        ],
        top_k=2,
    )

    assert [item["id"] for item in result] == ["b", "a"]
    assert [item["rerank_score"] for item in result] == [0.92, 0.35]
    payload = client.post.call_args.kwargs["json"]
    assert payload["input"]["documents"] == ["不相关内容", "混合检索结合稠密和稀疏检索"]
    assert payload["parameters"]["top_n"] == 2


def test_dashscope_reranker_rejects_invalid_response_index():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "output": {"results": [{"index": 9, "relevance_score": 0.8}]}
    }
    client = MagicMock()
    client.post.return_value = response
    reranker = DashScopeReranker(_settings(), client=client)

    with pytest.raises(DashScopeRerankError, match="Invalid candidate index"):
        reranker.rerank("query", [{"id": "a", "text": "passage"}])
