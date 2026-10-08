"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Config-Driven: LLM/Embedding backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from collections.abc import Sequence
from typing import Any

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _ensure_legacy_langchain_compat() -> None:
    """Provide the removed VertexAI type names imported by Ragas 0.2.x.

    Ragas imports these classes unconditionally when its LLM module loads,
    although the OpenAI-compatible evaluation path never instantiates them.
    LangChain Community 0.4 removed the old module, so expose inert type names
    instead of downgrading the project's entire LangChain dependency graph.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name not in sys.modules:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            module = types.ModuleType(module_name)
            module.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[module_name] = module

    try:
        import langchain_community.llms as community_llms
    except ImportError:
        community_llms = None

    if community_llms is not None and not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = type("VertexAI", (), {})


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    _ensure_legacy_langchain_compat()
    import nest_asyncio

    original_apply = nest_asyncio.apply
    nest_asyncio.apply = lambda loop=None: None
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc
    finally:
        nest_asyncio.apply = original_apply


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge
    the quality of the generated answer against the retrieved context.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[Any],
        generated_answer: str | None = None,
        ground_truth: Any | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: The generated answer text. Required for Ragas.
            ground_truth: Ignored by Ragas (not needed for LLM-as-Judge).
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 – 1.0).

        Raises:
            ValueError: If query/chunks are invalid or generated_answer is missing.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── private helpers ───────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: list[str],
        answer: str,
    ) -> dict[str, float]:
        """Execute Ragas 0.2 single-turn metrics and return normalised scores."""
        from ragas import SingleTurnSample
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
        )

        needs_embeddings = ANSWER_RELEVANCY in self._metric_names
        llm, embeddings = self._build_wrappers(needs_embeddings=needs_embeddings)
        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
        )

        scores: dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == FAITHFULNESS:
                metric = Faithfulness(llm=llm)
            elif metric_name == ANSWER_RELEVANCY:
                metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
            elif metric_name == CONTEXT_PRECISION:
                metric = LLMContextPrecisionWithoutReference(llm=llm)
            else:
                continue

            scores[metric_name] = float(
                asyncio.run(metric.single_turn_ascore(sample))
            )

        return scores

    def _build_wrappers(self, needs_embeddings: bool = False) -> tuple[Any, Any | None]:
        """Build Ragas LLM and Embedding wrappers from project settings.

        Uses Ragas 0.2's LangChain wrappers while preserving the project's
        OpenAI-compatible ``base_url`` configuration for DashScope.

        Returns:
            Tuple of (llm_wrapper, embeddings_wrapper).
        """
        from langchain_openai import (
            AzureChatOpenAI,
            AzureOpenAIEmbeddings,
            ChatOpenAI,
            OpenAIEmbeddings,
        )
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        if provider == "azure":
            langchain_llm = AzureChatOpenAI(
                azure_deployment=llm_cfg.deployment_name or llm_cfg.model,
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
                temperature=0,
            )
        elif provider == "openai":
            langchain_llm = ChatOpenAI(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
                base_url=getattr(llm_cfg, "base_url", None),
                temperature=0,
                max_retries=2,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai"
            )
        llm = LangchainLLMWrapper(langchain_llm)

        if not needs_embeddings:
            return llm, None

        emb_cfg = self.settings.embedding
        emb_provider = emb_cfg.provider.lower()
        if emb_provider == "azure":
            langchain_embeddings = AzureOpenAIEmbeddings(
                azure_deployment=emb_cfg.deployment_name or emb_cfg.model,
                api_key=emb_cfg.api_key,
                azure_endpoint=emb_cfg.azure_endpoint,
                api_version=getattr(emb_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif emb_provider == "openai":
            langchain_embeddings = OpenAIEmbeddings(
                model=emb_cfg.model,
                api_key=emb_cfg.api_key,
                base_url=getattr(emb_cfg, "base_url", None),
                check_embedding_ctx_length=False,
            )
        else:
            raise ValueError(
                f"Unsupported embedding provider for Ragas: '{emb_provider}'. "
                "Supported: azure, openai"
            )

        embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

        return llm, embeddings

    def _extract_texts(self, chunks: list[Any]) -> list[str]:
        """Extract text strings from various chunk representations.

        Args:
            chunks: List of chunk dicts, strings, or objects with .text.

        Returns:
            List of text strings.
        """
        texts: list[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> list[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
