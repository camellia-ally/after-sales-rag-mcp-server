"""Behavior tests for the minimal internal RAG answer service."""

from types import SimpleNamespace
from unittest.mock import Mock

from src.core.answering.rag_answer_service import RAGAnswerService
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, ChatResponse


def _result(chunk_id: str, text: str, chunk_index: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.9,
        text=text,
        metadata={
            "source_path": "docs/rag-guide.pdf",
            "chunk_index": chunk_index,
            "title": "RAG Guide",
        },
    )


def test_answer_passes_numbered_retrieval_context_to_llm() -> None:
    hybrid_search = Mock()
    hybrid_search.search.return_value = [
        _result("chunk-1", "稠密检索擅长语义匹配。", 1),
        _result("chunk-2", "BM25擅长关键词匹配。", 2),
    ]
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = ChatResponse(
        content="两种检索方式具有互补性。[1][2]",
        model="test-model",
        usage={"total_tokens": 42},
    )
    service = RAGAnswerService(hybrid_search=hybrid_search, llm=llm)

    response = service.answer("稠密检索和BM25有什么区别？", top_k=2)

    assert response.status == "answered"
    assert response.answer == "两种检索方式具有互补性。[1][2]"
    assert len(response.citations) == 2
    assert response.metadata["model"] == "test-model"
    messages = llm.chat.call_args.args[0]
    supplied_context = "\n".join(message.content for message in messages)
    assert "[1]" in supplied_context
    assert "chunk-1" in supplied_context
    assert "稠密检索擅长语义匹配" in supplied_context
    assert "[2]" in supplied_context
    assert "BM25擅长关键词匹配" in supplied_context


def test_empty_retrieval_refuses_without_calling_llm() -> None:
    hybrid_search = Mock()
    hybrid_search.search.return_value = []
    llm = Mock(spec=BaseLLM)
    service = RAGAnswerService(hybrid_search=hybrid_search, llm=llm)

    response = service.answer("知识库中不存在的问题")

    assert response.status == "refused"
    assert response.citations == []
    llm.chat.assert_not_called()


def test_retrieval_failure_is_not_reported_as_missing_knowledge() -> None:
    hybrid_search = Mock()
    hybrid_search.search.side_effect = RuntimeError("vector store unavailable")
    llm = Mock(spec=BaseLLM)
    service = RAGAnswerService(hybrid_search=hybrid_search, llm=llm)

    response = service.answer("什么是RAG？")

    assert response.status == "error"
    assert response.metadata["error_stage"] == "retrieval"
    assert "vector store unavailable" in response.metadata["error"]
    llm.chat.assert_not_called()


def test_only_citations_used_by_the_answer_are_returned() -> None:
    hybrid_search = Mock()
    hybrid_search.search.return_value = [
        _result("chunk-1", "第一条资料。", 1),
        _result("chunk-2", "第二条资料支持最终结论。", 2),
    ]
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = ChatResponse(
        content="最终结论来自第二条资料。[2]",
        model="test-model",
    )
    service = RAGAnswerService(hybrid_search=hybrid_search, llm=llm)

    response = service.answer("哪条资料支持结论？", top_k=2)

    assert response.status == "answered"
    assert len(response.citations) == 1
    assert response.citations[0].index == 2
    assert response.citations[0].chunk_id == "chunk-2"


def test_unknown_citation_number_is_rejected() -> None:
    hybrid_search = Mock()
    hybrid_search.search.return_value = [
        _result("chunk-1", "第一条资料。", 1),
        _result("chunk-2", "第二条资料。", 2),
    ]
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = ChatResponse(
        content="模型错误地引用了不存在的资料。[9]",
        model="test-model",
    )
    service = RAGAnswerService(hybrid_search=hybrid_search, llm=llm)

    response = service.answer("测试非法引用", top_k=2)

    assert response.status == "error"
    assert response.citations == []
    assert response.metadata["error_stage"] == "citation_validation"
    assert "9" in response.metadata["error"]


def test_insufficient_evidence_marker_becomes_fixed_refusal() -> None:
    hybrid_search = Mock()
    hybrid_search.search.return_value = [
        _result("chunk-1", "资料只提到了 Azure Embedding。", 1),
    ]
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = ChatResponse(
        content="INSUFFICIENT_EVIDENCE",
        model="test-model",
    )
    service = RAGAnswerService(hybrid_search=hybrid_search, llm=llm)

    response = service.answer("什么是Azure OpenAI？")

    assert response.status == "refused"
    assert response.citations == []
    assert response.metadata["stop_reason"] == "insufficient_evidence"
    assert llm.chat.call_count == 1


def test_answer_uses_reranked_results_as_llm_context() -> None:
    hybrid_search = Mock()
    first = _result("chunk-1", "粗排第一。", 1)
    second = _result("chunk-2", "精排后第一。", 2)
    hybrid_search.search.return_value = [first, second]
    reranker = Mock()
    reranker.is_enabled = True
    reranker.rerank.return_value = SimpleNamespace(
        results=[second],
        used_fallback=False,
        fallback_reason=None,
    )
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = ChatResponse(
        content="结论来自精排后的第一条。[1]",
        model="test-model",
    )
    service = RAGAnswerService(
        hybrid_search=hybrid_search,
        llm=llm,
        reranker=reranker,
    )

    response = service.answer("测试精排", top_k=1)

    assert response.status == "answered"
    hybrid_search.search.assert_called_once_with(
        query="测试精排",
        top_k=2,
        trace=None,
        return_details=False,
    )
    reranker.rerank.assert_called_once_with(
        query="测试精排",
        results=[first, second],
        top_k=1,
        trace=None,
    )
    messages = llm.chat.call_args.args[0]
    assert "[1] chunk_id=chunk-2" in messages[1].content
    assert response.citations[0].chunk_id == "chunk-2"
    assert response.metadata["retrieved_chunk_ids"] == ["chunk-2"]


def test_reranker_fallback_is_reported_as_system_error() -> None:
    hybrid_search = Mock()
    results = [
        _result("chunk-1", "第一条。", 1),
        _result("chunk-2", "第二条。", 2),
    ]
    hybrid_search.search.return_value = results
    reranker = Mock()
    reranker.is_enabled = True
    reranker.rerank.return_value = SimpleNamespace(
        results=results,
        used_fallback=True,
        fallback_reason="provider timeout",
    )
    llm = Mock(spec=BaseLLM)
    service = RAGAnswerService(
        hybrid_search=hybrid_search,
        llm=llm,
        reranker=reranker,
    )

    response = service.answer("测试精排失败", top_k=1)

    assert response.status == "error"
    assert response.metadata["error_stage"] == "rerank"
    assert "provider timeout" in response.metadata["error"]
    llm.chat.assert_not_called()


def test_low_rerank_relevance_refuses_before_generation() -> None:
    hybrid_search = Mock()
    hybrid_search.search.return_value = [_result("chunk-1", "无关资料。", 1)]
    reranker = Mock(is_enabled=True)
    candidate = _result("chunk-1", "无关资料。", 1)
    candidate.metadata["rerank_score"] = 0.2
    reranker.rerank.return_value = SimpleNamespace(
        results=[candidate], used_fallback=False, fallback_reason=None,
    )
    llm = Mock(spec=BaseLLM)
    service = RAGAnswerService(hybrid_search, llm, reranker=reranker,
                               min_rerank_score=0.5)

    response = service.answer("无关问题")

    assert response.status == "refused"
    assert response.metadata["stop_reason"] == "low_rerank_relevance"
    assert response.metadata["max_rerank_score"] == 0.2
    llm.chat.assert_not_called()


def test_high_rerank_relevance_keeps_llm_evidence_check() -> None:
    hybrid_search = Mock()
    hybrid_search.search.return_value = [_result("chunk-1", "有关资料。", 1)]
    reranker = Mock(is_enabled=True)
    candidate = _result("chunk-1", "有关资料。", 1)
    candidate.metadata["rerank_score"] = 0.8
    reranker.rerank.return_value = SimpleNamespace(
        results=[candidate], used_fallback=False, fallback_reason=None,
    )
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = ChatResponse(content="INSUFFICIENT_EVIDENCE", model="test")
    service = RAGAnswerService(hybrid_search, llm, reranker=reranker,
                               min_rerank_score=0.5)

    response = service.answer("证据仍然不充分的问题")

    assert response.status == "refused"
    assert response.metadata["stop_reason"] == "insufficient_evidence"
    assert response.metadata["max_rerank_score"] == 0.8
    llm.chat.assert_called_once()


def test_threshold_does_not_treat_missing_rerank_score_as_low_relevance() -> None:
    hybrid_search = Mock()
    hybrid_search.search.return_value = [_result("chunk-1", "资料。", 1)]
    reranker = Mock(is_enabled=True)
    reranker.rerank.return_value = SimpleNamespace(
        results=[_result("chunk-1", "资料。", 1)],
        used_fallback=False, fallback_reason=None,
    )
    llm = Mock(spec=BaseLLM)
    service = RAGAnswerService(hybrid_search, llm, reranker=reranker,
                               min_rerank_score=0.5)

    response = service.answer("问题")

    assert response.status == "error"
    assert response.metadata["error_stage"] == "relevance_gate"
    llm.chat.assert_not_called()
