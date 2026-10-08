#!/usr/bin/env python
"""Generate one answer from the local RAG knowledge base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from scripts.query import build_query_components  # noqa: E402
from src.core.answering import RAGAnswerService  # noqa: E402
from src.core.settings import load_settings  # noqa: E402
from src.core.trace import TraceCollector, TraceContext  # noqa: E402
from src.libs.llm.llm_factory import LLMFactory  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one answer request."""
    parser = argparse.ArgumentParser(
        description="Generate an LLM answer grounded in the local knowledge base."
    )
    parser.add_argument("--query", "-q", required=True, help="Question to answer.")
    parser.add_argument(
        "--collection",
        "-c",
        default="default",
        help="Collection name (default: default).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks supplied to the LLM (default: 5).",
    )
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        help="Refuse before generation when every reranked passage scores below this value (0-1).",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to settings.yaml.",
    )
    return parser.parse_args()


def _print_response(response) -> None:
    """Render the provider-independent response for a terminal user."""
    print("\n" + "=" * 60)
    print(f"状态: {response.status}")
    print("=" * 60)
    print(response.answer)

    score = response.metadata.get("max_rerank_score")
    threshold = response.metadata.get("min_rerank_score")
    if score is not None:
        detail = f"重排最高相关度: {score:.4f}"
        if threshold is not None:
            detail += f"（拒答阈值: {threshold:.4f}）"
        print(detail)
    if response.status == "refused":
        print(f"拒答原因: {response.metadata.get('stop_reason', 'unknown')}")

    if response.citations:
        print("\n参考资料:")
        for citation in response.citations:
            source_name = Path(citation.source).name
            chunk_index = citation.metadata.get("chunk_index", "")
            location = f", Chunk {chunk_index}" if chunk_index != "" else ""
            print(
                f"[{citation.index}] {source_name}{location} "
                f"(chunk_id={citation.chunk_id})"
            )

    model = response.metadata.get("model")
    usage = response.metadata.get("usage")
    if model:
        print(f"\n模型: {model}")
    if usage:
        print(f"Token: {usage}")
    if response.status == "error":
        print(f"错误阶段: {response.metadata.get('error_stage', 'unknown')}")
        print(f"错误原因: {response.metadata.get('error', 'unknown')}")


def main() -> int:
    """Assemble real dependencies and execute one RAG answer request."""
    args = parse_args()

    try:
        settings = load_settings(args.config)
        hybrid_search, reranker = build_query_components(settings, args.collection)
        llm = LLMFactory.create(settings)
    except Exception as exc:
        print(f"[FAIL] 初始化失败: {exc}")
        return 2

    try:
        service = RAGAnswerService(
            hybrid_search=hybrid_search,
            llm=llm,
            reranker=reranker,
            min_rerank_score=args.min_rerank_score,
        )
    except ValueError as exc:
        print(f"[FAIL] 拒答阈值配置错误: {exc}")
        return 2
    trace = TraceContext(trace_type="query")
    trace.metadata.update(
        {
            "source": "answer_cli",
            "query": args.query[:200],
            "collection": args.collection,
            "top_k": args.top_k,
        }
    )

    response = service.answer(
        query=args.query,
        top_k=args.top_k,
        trace=trace,
    )
    trace.metadata["answer_status"] = response.status
    trace.finish()
    TraceCollector().collect(trace)
    _print_response(response)

    return 1 if response.status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
