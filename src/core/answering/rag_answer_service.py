"""Minimal orchestration service for retrieval-augmented answers."""

from __future__ import annotations

import math
import re
import time
from typing import Any

from src.core.answering.types import RAGAnswerResponse
from src.core.response.citation_generator import CitationGenerator
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message

_REFUSAL_MESSAGE = "当前知识库中没有足够的相关资料，无法可靠回答这个问题。"
_INSUFFICIENT_EVIDENCE_MARKER = "INSUFFICIENT_EVIDENCE"
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class RAGAnswerService:
    """Connect HybridSearch results to the project's existing LLM interface."""

    def __init__(
        self,
        hybrid_search: Any,
        llm: BaseLLM,
        citation_generator: CitationGenerator | None = None,
        reranker: Any = None,
        min_rerank_score: float | None = None,
    ) -> None:
        if min_rerank_score is not None:
            if (
                isinstance(min_rerank_score, bool)
                or not isinstance(min_rerank_score, (int, float))
                or not math.isfinite(min_rerank_score)
                or not 0 <= min_rerank_score <= 1
            ):
                raise ValueError("min_rerank_score must be a finite number between 0 and 1")
            if reranker is None or not reranker.is_enabled:
                raise ValueError("min_rerank_score requires an enabled reranker")
        self.hybrid_search = hybrid_search
        self.llm = llm
        self.citation_generator = citation_generator or CitationGenerator()
        self.reranker = reranker
        self.min_rerank_score = min_rerank_score

    def answer(
        self,
        query: str,
        top_k: int = 5,
        trace: Any | None = None,
    ) -> RAGAnswerResponse:
        """Retrieve evidence, build context, and generate one grounded answer."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            has_reranker = self.reranker is not None and self.reranker.is_enabled
            retrieval_top_k = top_k * 2 if has_reranker else top_k
            search_output = self.hybrid_search.search(
                query=query,
                top_k=retrieval_top_k,
                trace=trace,
                return_details=False,
            )
            results = self._extract_results(search_output)
        except Exception as exc:
            return self._error_response(query, "retrieval", exc)

        if has_reranker and results:
            try:
                rerank_result = self.reranker.rerank(
                    query=query,
                    results=results,
                    top_k=top_k,
                    trace=trace,
                )
                if rerank_result.used_fallback:
                    raise RuntimeError(
                        f"Reranker used fallback: {rerank_result.fallback_reason}"
                    )
                results = rerank_result.results
            except Exception as exc:
                return self._error_response(query, "rerank", exc)

        if not results:
            return RAGAnswerResponse(
                status="refused",
                answer=_REFUSAL_MESSAGE,
                citations=[],
                metadata={
                    "query": query,
                    "retrieved_count": 0,
                    "retrieved_chunk_ids": [],
                    "stop_reason": "no_retrieval_results",
                },
            )

        rerank_scores = [result.metadata.get("rerank_score") for result in results]
        valid_scores = [
            float(score) for score in rerank_scores
            if isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
        ]
        max_rerank_score = max(valid_scores) if valid_scores else None
        if self.min_rerank_score is not None:
            if max_rerank_score is None:
                return self._error_response(
                    query, "relevance_gate", ValueError("Reranker returned no relevance scores")
                )
            if max_rerank_score < self.min_rerank_score:
                if trace is not None:
                    trace.record_stage("relevance_gate", {
                        "max_rerank_score": max_rerank_score,
                        "min_rerank_score": self.min_rerank_score,
                        "passed": False,
                    })
                return RAGAnswerResponse(
                    status="refused",
                    answer=_REFUSAL_MESSAGE,
                    citations=[],
                    metadata={
                        "query": query,
                        "retrieved_count": len(results),
                        "retrieved_chunk_ids": [result.chunk_id for result in results],
                        "max_rerank_score": max_rerank_score,
                        "min_rerank_score": self.min_rerank_score,
                        "stop_reason": "low_rerank_relevance",
                    },
                )
            if trace is not None:
                trace.record_stage("relevance_gate", {
                    "max_rerank_score": max_rerank_score,
                    "min_rerank_score": self.min_rerank_score,
                    "passed": True,
                })

        context_started = time.monotonic()
        context = self._build_numbered_context(results)
        if trace is not None:
            trace.record_stage(
                "context_building",
                {
                    "retrieved_count": len(results),
                    "chunk_ids": [result.chunk_id for result in results],
                },
                elapsed_ms=(time.monotonic() - context_started) * 1000.0,
            )

        messages = self._build_messages(query, context)

        generation_started = time.monotonic()
        try:
            llm_response = self.llm.chat(messages, trace=trace)
        except Exception as exc:
            return self._error_response(query, "answer_generation", exc)

        answer, model, usage = self._normalize_llm_response(llm_response)
        if trace is not None:
            trace.record_stage(
                "answer_generation",
                {
                    "model": model,
                    "usage": usage,
                },
                elapsed_ms=(time.monotonic() - generation_started) * 1000.0,
            )

        if answer.strip() == _INSUFFICIENT_EVIDENCE_MARKER:
            return RAGAnswerResponse(
                status="refused",
                answer=_REFUSAL_MESSAGE,
                citations=[],
                metadata={
                    "query": query,
                    "retrieved_count": len(results),
                    "retrieved_chunk_ids": [result.chunk_id for result in results],
                    "model": model,
                    "usage": usage,
                    "stop_reason": "insufficient_evidence",
                    "max_rerank_score": max_rerank_score,
                    "min_rerank_score": self.min_rerank_score,
                },
            )

        try:
            citations = self._select_used_citations(answer, results)
        except ValueError as exc:
            if trace is not None:
                trace.record_stage(
                    "citation_validation",
                    {"valid": False, "error": str(exc)},
                )
            return self._error_response(query, "citation_validation", exc)

        if trace is not None:
            trace.record_stage(
                "citation_validation",
                {
                    "valid": True,
                    "used_indices": [citation.index for citation in citations],
                },
            )

        return RAGAnswerResponse(
            status="answered",
            answer=answer,
            citations=citations,
            metadata={
                "query": query,
                "retrieved_count": len(results),
                "retrieved_chunk_ids": [result.chunk_id for result in results],
                "model": model,
                "usage": usage,
                "stop_reason": "answered",
                "max_rerank_score": max_rerank_score,
                "min_rerank_score": self.min_rerank_score,
            },
        )

    @staticmethod
    def _extract_results(search_output: Any) -> list[RetrievalResult]:
        if isinstance(search_output, list):
            return search_output
        results = getattr(search_output, "results", None)
        if isinstance(results, list):
            return results
        raise TypeError("HybridSearch returned an unsupported result type")

    @staticmethod
    def _build_numbered_context(results: list[RetrievalResult]) -> str:
        sections = []
        for index, result in enumerate(results, start=1):
            source = result.metadata.get("source_path", "unknown")
            sections.append(
                f"[{index}] chunk_id={result.chunk_id}\n"
                f"source={source}\n"
                f"text={result.text}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _build_messages(query: str, context: str) -> list[Message]:
        return [
            Message(
                role="system",
                content=(
                    "你是知识库问答助手，只能根据提供的检索资料回答，"
                    "不得使用自身知识补充资料中不存在的事实。"
                    "如果资料没有直接包含回答核心问题所需的信息，"
                    "只输出INSUFFICIENT_EVIDENCE，不要输出其他内容。"
                    "如果资料足够，主要结论必须使用资料编号如[1]进行引用。"
                ),
            ),
            Message(
                role="user",
                content=f"问题：{query}\n\n检索资料：\n{context}",
            ),
        ]

    @staticmethod
    def _normalize_llm_response(response: Any) -> tuple[str, str, Any]:
        if isinstance(response, ChatResponse):
            return response.content, response.model, response.usage
        if isinstance(response, str):
            return response, "unknown", None
        raise TypeError("LLM returned an unsupported response type")

    def _select_used_citations(
        self,
        answer: str,
        results: list[RetrievalResult],
    ) -> list[Any]:
        """Return real citations referenced by valid ``[n]`` markers."""
        used_indices = list(
            dict.fromkeys(int(match) for match in _CITATION_PATTERN.findall(answer))
        )
        all_citations = self.citation_generator.generate(results)

        invalid_indices = [
            index for index in used_indices if index < 1 or index > len(all_citations)
        ]
        if invalid_indices:
            invalid_text = ", ".join(str(index) for index in invalid_indices)
            raise ValueError(f"Answer contains unknown citation indices: {invalid_text}")

        return [all_citations[index - 1] for index in used_indices]

    @staticmethod
    def _error_response(
        query: str,
        stage: str,
        error: Exception,
    ) -> RAGAnswerResponse:
        return RAGAnswerResponse(
            status="error",
            answer="系统暂时无法完成回答，请稍后重试。",
            citations=[],
            metadata={
                "query": query,
                "error_stage": stage,
                "error": str(error),
                "stop_reason": "system_error",
            },
        )
