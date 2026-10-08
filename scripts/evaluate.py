#!/usr/bin/env python
"""Evaluation script for Modular RAG MCP Server.

Runs batch evaluation against a golden test set and outputs a metrics report.

Usage:
    # Run with default settings (custom evaluator)
    python scripts/evaluate.py

    # Specify a custom golden test set
    python scripts/evaluate.py --test-set path/to/golden.json

    # Use a specific collection
    python scripts/evaluate.py --collection technical_docs

    # JSON output
    python scripts/evaluate.py --json

Exit codes:
    0 - Success
    1 - Evaluation failure
    2 - Configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation against a golden test set."
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to golden test set JSON file (default: tests/fixtures/golden_test_set.json)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to search within.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip retrieval (evaluate with mock chunks for testing).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the structured JSON report to this path.",
    )
    parser.add_argument(
        "--answerability",
        nargs="+",
        choices=("answerable", "partial", "unanswerable"),
        default=None,
        help="Evaluate only the selected answerability labels.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Do not create placeholder answers; compute retrieval metrics only.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Evaluate RRF results without the configured reranker.",
    )
    parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help=(
            "Compare RRF top-k with RRF + configured reranker on the same "
            "candidate pool, and print candidate text for golden-set review."
        ),
    )
    parser.add_argument(
        "--review-file",
        default=None,
        help=(
            "Write full rerank comparison candidates to a Markdown review file. "
            "Used with --compare-rerank."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        from src.core.settings import load_settings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        from src.observability.evaluation.eval_runner import EvalRunner

        settings = load_settings()
    except Exception as exc:
        print(f"❌ Configuration error: {exc}", file=sys.stderr)
        return 2

    # Create evaluator from config
    try:
        evaluator = EvaluatorFactory.create(settings)
        evaluator_name = type(evaluator).__name__
    except Exception as exc:
        print(f"❌ Failed to create evaluator: {exc}", file=sys.stderr)
        return 2

    # Create HybridSearch (unless --no-search)
    hybrid_search = None
    if not args.no_search:
        try:
            from src.core.query_engine.dense_retriever import create_dense_retriever
            from src.core.query_engine.hybrid_search import create_hybrid_search
            from src.core.query_engine.query_processor import QueryProcessor
            from src.core.query_engine.sparse_retriever import create_sparse_retriever
            from src.ingestion.storage.bm25_indexer import BM25Indexer
            from src.libs.embedding.embedding_factory import EmbeddingFactory
            from src.libs.vector_store.vector_store_factory import VectorStoreFactory

            collection = args.collection or "default"

            vector_store = VectorStoreFactory.create(
                settings, collection_name=collection,
            )
            embedding_client = EmbeddingFactory.create(settings)
            dense_retriever = create_dense_retriever(
                settings=settings,
                embedding_client=embedding_client,
                vector_store=vector_store,
            )
            bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
            sparse_retriever = create_sparse_retriever(
                settings=settings,
                bm25_indexer=bm25_indexer,
                vector_store=vector_store,
            )
            sparse_retriever.default_collection = collection

            query_processor = QueryProcessor()
            hybrid_search = create_hybrid_search(
                settings=settings,
                query_processor=query_processor,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
            )
            print(f"✅ HybridSearch initialized for collection: {collection}")
        except Exception as exc:
            print(f"❌ Failed to initialize search: {exc}", file=sys.stderr)
            return 2

    # Create and run EvalRunner
    if args.compare_rerank:
        if hybrid_search is None:
            print("❌ --compare-rerank requires an initialized HybridSearch.", file=sys.stderr)
            return 2
        try:
            from src.core.query_engine.reranker import create_core_reranker

            reranker = create_core_reranker(settings)
            if not reranker.is_enabled:
                print(
                    "❌ Configured reranker is not active. Check rerank.enabled, "
                    "provider, model, base_url, and initialization logs.",
                    file=sys.stderr,
                )
                return 2
        except Exception as exc:
            print(f"❌ Failed to create reranker: {exc}", file=sys.stderr)
            return 2

        return _run_rerank_comparison(
            hybrid_search=hybrid_search,
            reranker=reranker,
            evaluator=evaluator,
            test_set_path=args.test_set,
            top_k=args.top_k,
            review_file=args.review_file,
        )

    reranker = None
    if hybrid_search is not None and not args.no_rerank:
        try:
            from src.core.query_engine.reranker import create_core_reranker

            configured_reranker = create_core_reranker(settings=settings)
            if not configured_reranker.is_enabled:
                print(
                    "❌ Configured reranker is not active. Use --no-rerank only "
                    "when intentionally measuring the RRF-only baseline.",
                    file=sys.stderr,
                )
                return 2
            reranker = configured_reranker
            print(
                "✅ Reranker enabled for formal evaluation: "
                f"{settings.rerank.provider}/{settings.rerank.model}"
            )
        except Exception as exc:
            print(f"❌ Failed to initialize reranker: {exc}", file=sys.stderr)
            return 2

    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        reranker=reranker,
        generate_answers=not args.retrieval_only,
        strict_errors=True,
    )

    try:
        print(f"\n🔍 Running evaluation with {evaluator_name}...")
        print(f"📄 Test set: {args.test_set}")
        print(f"🔢 Top-K: {args.top_k}\n")

        report = runner.run(
            test_set_path=args.test_set,
            top_k=args.top_k,
            collection=args.collection,
            include_answerability=(
                set(args.answerability) if args.answerability else None
            ),
        )
    except Exception as exc:
        print(f"❌ Evaluation failed: {exc}", file=sys.stderr)
        return 1

    # Output results
    report_dict = report.to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"💾 JSON report written to: {output_path.resolve()}")

    if args.json:
        print(json.dumps(report_dict, indent=2, ensure_ascii=False))
    else:
        _print_report(report)

    return 0


def _run_rerank_comparison(
    hybrid_search,
    reranker,
    evaluator,
    test_set_path: str,
    top_k: int,
    review_file: str | None = None,
) -> int:
    """Compare RRF and reranked results using one shared candidate pool per query."""
    from src.observability.evaluation.eval_runner import load_test_set

    test_cases = load_test_set(test_set_path)
    candidate_k = top_k * 2
    baseline_metrics = []
    rerank_metrics = []
    retrieval_times = []
    rerank_times = []
    review_lines = [
        "# Reranker Golden Set 人工标注",
        "",
        "判断标准：只阅读该 Chunk，不使用外部知识，是否足以回答问题的核心内容。",
        "请把正确候选前的 `[ ]` 改成 `[x]`。",
        "",
    ]

    print("\n" + "=" * 72)
    print("RERANK A/B COMPARISON")
    print(f"Questions: {len(test_cases)} | Final top_k: {top_k} | Candidates: {candidate_k}")
    print("A = RRF only | B = RRF + configured reranker")
    print("=" * 72)

    for index, test_case in enumerate(test_cases, start=1):
        retrieve_started = time.monotonic()
        search_result = hybrid_search.search(query=test_case.query, top_k=candidate_k)
        candidates = search_result if isinstance(search_result, list) else search_result.results
        retrieval_ms = (time.monotonic() - retrieve_started) * 1000.0
        retrieval_times.append(retrieval_ms)

        baseline = candidates[:top_k]
        rerank_started = time.monotonic()
        rerank_result = reranker.rerank(
            query=test_case.query,
            results=candidates,
            top_k=top_k,
        )
        rerank_ms = (time.monotonic() - rerank_started) * 1000.0
        rerank_times.append(rerank_ms)

        if rerank_result.used_fallback:
            print(
                f"\n❌ [{index}] Reranker fallback: {rerank_result.fallback_reason}",
                file=sys.stderr,
            )
            return 1

        reranked = rerank_result.results
        ground_truth = {"ids": test_case.expected_chunk_ids}
        baseline_score = evaluator.evaluate(
            query=test_case.query,
            retrieved_chunks=baseline,
            ground_truth=ground_truth,
        )
        rerank_score = evaluator.evaluate(
            query=test_case.query,
            retrieved_chunks=reranked,
            ground_truth=ground_truth,
        )
        baseline_metrics.append(baseline_score)
        rerank_metrics.append(rerank_score)

        print(f"\n[{index}] {test_case.query}")
        print(f"    Current expected IDs: {', '.join(test_case.expected_chunk_ids) or '(none)'}")
        print(
            "    A RRF:    "
            f"hit_rate={baseline_score.get('hit_rate', 0.0):.4f}, "
            f"mrr={baseline_score.get('mrr', 0.0):.4f}, "
            f"retrieval={retrieval_ms:.0f} ms"
        )
        print(
            "    B Rerank: "
            f"hit_rate={rerank_score.get('hit_rate', 0.0):.4f}, "
            f"mrr={rerank_score.get('mrr', 0.0):.4f}, "
            f"rerank_extra={rerank_ms:.0f} ms"
        )
        _print_review_candidates(
            baseline=baseline,
            reranked=reranked,
            expected_ids=set(test_case.expected_chunk_ids),
        )
        _append_review_markdown(
            lines=review_lines,
            question_number=index,
            query=test_case.query,
            baseline=baseline,
            reranked=reranked,
            expected_ids=set(test_case.expected_chunk_ids),
        )

    baseline_avg = _average_metrics(baseline_metrics)
    rerank_avg = _average_metrics(rerank_metrics)
    avg_retrieval_ms = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
    avg_rerank_ms = sum(rerank_times) / len(rerank_times) if rerank_times else 0.0

    print("\n" + "=" * 72)
    print("AGGREGATE A/B")
    print(
        "A RRF:    "
        f"hit_rate={baseline_avg.get('hit_rate', 0.0):.4f}, "
        f"mrr={baseline_avg.get('mrr', 0.0):.4f}, "
        f"avg_retrieval={avg_retrieval_ms:.0f} ms"
    )
    print(
        "B Rerank: "
        f"hit_rate={rerank_avg.get('hit_rate', 0.0):.4f}, "
        f"mrr={rerank_avg.get('mrr', 0.0):.4f}, "
        f"avg_rerank_extra={avg_rerank_ms:.0f} ms"
    )
    print(
        "Delta:    "
        f"hit_rate={rerank_avg.get('hit_rate', 0.0) - baseline_avg.get('hit_rate', 0.0):+.4f}, "
        f"mrr={rerank_avg.get('mrr', 0.0) - baseline_avg.get('mrr', 0.0):+.4f}"
    )
    print("=" * 72)
    print("Review task: for each question, list every candidate ID that can support the answer.")

    if review_file:
        review_path = Path(review_file)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")
        print(f"Full candidate review written to: {review_path.resolve()}")
    return 0


def _average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = {key for row in rows for key in row}
    return {
        key: sum(row.get(key, 0.0) for row in rows) / len(rows)
        for key in sorted(keys)
    } if rows else {}


def _print_review_candidates(baseline, reranked, expected_ids: set[str]) -> None:
    """Print unique top-k candidates with both ranks and enough text for human review."""
    baseline_rank = {item.chunk_id: rank for rank, item in enumerate(baseline, start=1)}
    rerank_rank = {item.chunk_id: rank for rank, item in enumerate(reranked, start=1)}
    ordered_ids = list(dict.fromkeys([*rerank_rank.keys(), *baseline_rank.keys()]))
    by_id = {item.chunk_id: item for item in [*baseline, *reranked]}

    print("    Candidate review:")
    for chunk_id in ordered_ids:
        item = by_id[chunk_id]
        marker = "CURRENT_EXPECTED" if chunk_id in expected_ids else "REVIEW"
        rrf_position = baseline_rank.get(chunk_id, "-")
        rerank_position = rerank_rank.get(chunk_id, "-")
        text = " ".join((item.text or "").split())
        print(
            f"      [{marker}] id={chunk_id} "
            f"RRF_rank={rrf_position} Rerank_rank={rerank_position}"
        )
        print(f"        text={text}")


def _append_review_markdown(
    lines: list[str],
    question_number: int,
    query: str,
    baseline,
    reranked,
    expected_ids: set[str],
) -> None:
    """Append full candidate content in a human-friendly labeling format."""
    baseline_rank = {item.chunk_id: rank for rank, item in enumerate(baseline, start=1)}
    rerank_rank = {item.chunk_id: rank for rank, item in enumerate(reranked, start=1)}
    ordered_ids = list(dict.fromkeys([*rerank_rank.keys(), *baseline_rank.keys()]))
    by_id = {item.chunk_id: item for item in [*baseline, *reranked]}

    lines.extend([
        f"## 第 {question_number} 题",
        "",
        f"**问题：** {query}",
        "",
        "当前 expected IDs：",
        "",
        *[f"- `{chunk_id}`" for chunk_id in sorted(expected_ids)],
        "",
    ])

    for chunk_id in ordered_ids:
        item = by_id[chunk_id]
        current = "是" if chunk_id in expected_ids else "否"
        text = (item.text or "").strip()
        lines.extend([
            f"### Candidate `{chunk_id}`",
            "",
            "- [ ] 能够支撑回答",
            f"- RRF 排名：{baseline_rank.get(chunk_id, '-')} ",
            f"- Rerank 排名：{rerank_rank.get(chunk_id, '-')} ",
            f"- 当前是否已标为 expected：{current}",
            "",
            "```text",
            text,
            "```",
            "",
        ])


def _print_report(report) -> None:
    """Print formatted evaluation report."""
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Evaluator: {report.evaluator_name}")
    print(f"  Test Set:  {report.test_set_path}")
    print(f"  Queries:   {len(report.query_results)}")
    print(f"  Time:      {report.total_elapsed_ms:.0f} ms")
    print()

    # Aggregate metrics
    print("─" * 60)
    print("  AGGREGATE METRICS")
    print("─" * 60)
    if report.aggregate_metrics:
        for metric, value in sorted(report.aggregate_metrics.items()):
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {metric:<25s} {bar} {value:.4f}")
    else:
        print("  (no metrics computed)")
    print()

    # Per-query details
    print("─" * 60)
    print("  PER-QUERY RESULTS")
    print("─" * 60)
    for i, qr in enumerate(report.query_results, 1):
        print(f"\n  [{i}] {qr.query}")
        print(f"      Retrieved: {len(qr.retrieved_chunk_ids)} chunks")
        if qr.metrics:
            for metric, value in sorted(qr.metrics.items()):
                print(f"      {metric}: {value:.4f}")
        else:
            print("      (no metrics)")
        print(f"      Time: {qr.elapsed_ms:.0f} ms")

    print()
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
