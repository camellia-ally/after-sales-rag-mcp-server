"""DashScope text rerank provider."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class DashScopeRerankError(RuntimeError):
    """Raised when the DashScope rerank request or response is invalid."""


class DashScopeReranker(BaseReranker):
    """Rerank passages with Alibaba Cloud Model Studio's text rerank API."""

    def __init__(self, settings: Any, client: Optional[Any] = None, **kwargs: Any) -> None:
        self.settings = settings
        self.model = settings.rerank.model
        self.api_key = self._resolve_api_key(settings)
        self.endpoint = getattr(settings.rerank, "base_url", None)
        self.timeout = float(getattr(settings.rerank, "timeout", 30.0))
        self._client = client
        self.kwargs = kwargs

        if not self.model:
            raise ValueError("rerank.model must be configured for DashScope reranking")
        if not self.endpoint:
            raise ValueError("rerank.base_url must be configured for DashScope reranking")
        if "{WorkspaceId}" in self.endpoint:
            raise ValueError(
                "Replace {WorkspaceId} in rerank.base_url with your Alibaba Cloud "
                "Model Studio workspace ID."
            )
        if not self.api_key:
            raise ValueError(
                "DashScope API key is missing. Configure rerank.api_key, "
                "embedding.api_key, or llm.api_key."
            )

    @staticmethod
    def _resolve_api_key(settings: Any) -> Optional[str]:
        """Use a rerank-specific key first, then reuse the existing DashScope key."""
        rerank_key = getattr(settings.rerank, "api_key", None)
        embedding_key = getattr(getattr(settings, "embedding", None), "api_key", None)
        llm_key = getattr(getattr(settings, "llm", None), "api_key", None)
        return rerank_key or embedding_key or llm_key

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)

        top_k = kwargs.get("top_k", len(candidates))
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        top_k = min(top_k, len(candidates))

        documents = []
        for index, candidate in enumerate(candidates):
            text = candidate.get("text") or candidate.get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Candidate at index {index} has no non-empty text")
            documents.append(text)

        payload = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_k, "return_documents": False},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response_data = self._post(payload, headers)
            return self._map_results(response_data, candidates)
        except DashScopeRerankError:
            raise
        except Exception as exc:
            raise DashScopeRerankError(f"DashScope rerank failed: {exc}") from exc

    def _post(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        if self._client is not None:
            response = self._client.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required for DashScope reranking") from exc

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise DashScopeRerankError(
                f"DashScope API error (HTTP {exc.response.status_code}): {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise DashScopeRerankError(f"DashScope request failed: {exc}") from exc

    def _map_results(
        self,
        response_data: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = response_data.get("output", {}).get("results")
        if not isinstance(results, list):
            raise DashScopeRerankError("DashScope response is missing output.results")

        reranked: List[Dict[str, Any]] = []
        seen_indices = set()
        for item in results:
            index = item.get("index") if isinstance(item, dict) else None
            score = item.get("relevance_score") if isinstance(item, dict) else None
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise DashScopeRerankError(f"Invalid candidate index in response: {index}")
            if index in seen_indices:
                raise DashScopeRerankError(f"Duplicate candidate index in response: {index}")
            if not isinstance(score, (int, float)):
                raise DashScopeRerankError(
                    f"Invalid relevance_score for candidate index {index}: {score}"
                )

            candidate = candidates[index].copy()
            candidate["rerank_score"] = float(score)
            reranked.append(candidate)
            seen_indices.add(index)

        if not reranked:
            raise DashScopeRerankError("DashScope returned no rerank results")

        return reranked
