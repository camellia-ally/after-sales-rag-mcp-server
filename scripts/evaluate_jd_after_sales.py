#!/usr/bin/env python
"""Paired pilot evaluation for the JD after-sales collection.

Source labels in the draft set require human review before publication.
This script does not claim answer correctness or policy validity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SET = ROOT / "tests/fixtures/jd_after_sales_eval_draft.json"
COLLECTION = "jd_after_sales"


def source_name(result) -> str:
    path = result.metadata.get("source_path") or result.metadata.get("source") or ""
    return Path(path.replace("\\", "/")).name


def rank_of(results, sources: set[str]) -> int | None:
    return next((i for i, result in enumerate(results, 1)
                 if source_name(result) in sources), None)


def chunk_rank_of(results, chunk_ids: set[str]) -> int | None:
    return next((i for i, result in enumerate(results, 1)
                 if result.chunk_id in chunk_ids), None)


def summarize(rows: list[dict], arm: str) -> dict:
    ranks = [row[arm]["rank"] for row in rows]
    return {
        "n": len(rows),
        "hit_at_5": round(sum(r is not None and r <= 5 for r in ranks) / len(rows), 4),
        "mrr_at_5": round(sum(1 / r for r in ranks if r is not None and r <= 5) / len(rows), 4),
        "top_1": round(sum(r == 1 for r in ranks) / len(rows), 4),
    }


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["test_cases"]
    if not cases:
        raise ValueError("Empty test set")
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case_id")
    valid_sources = {p.name for p in (ROOT / "data/documents/jd_after_sales").glob("*.md")}
    for case in cases:
        if case["answerability"] not in {"answerable", "partial", "unanswerable"}:
            raise ValueError(f"Bad answerability: {case['case_id']}")
        if case["answerability"] == "answerable" and not case.get("expected_sources"):
            raise ValueError(f"Missing source labels: {case['case_id']}")
        if case["answerability"] == "answerable" and not case.get("expected_chunk_ids"):
            raise ValueError(f"Missing chunk labels: {case['case_id']}")
        unknown = set(case.get("expected_sources", [])) - valid_sources
        if unknown:
            raise ValueError(f"Unknown sources for {case['case_id']}: {sorted(unknown)}")
        if not set(case.get("required_sources", [])) <= set(case.get("expected_sources", [])):
            raise ValueError(f"Required sources are not expected sources: {case['case_id']}")
        if not set(case.get("required_chunk_ids", [])) <= set(case.get("expected_chunk_ids", [])):
            raise ValueError(f"Required chunks are not expected chunks: {case['case_id']}")
        for evidence in case.get("evidence", []):
            if evidence["source"] not in case.get("expected_sources", []):
                raise ValueError(f"Evidence source is not expected: {case['case_id']}")
            source_text = (ROOT / "data/documents/jd_after_sales" / evidence["source"]).read_text(
                encoding="utf-8")
            if evidence["quote"] not in source_text:
                raise ValueError(f"Evidence quote missing from source: {case['case_id']}")
            if not set(evidence["matching_chunk_ids"]) <= set(case.get("expected_chunk_ids", [])):
                raise ValueError(f"Evidence chunk labels disagree: {case['case_id']}")
    return cases


def validate_snapshot(cases: list[dict], collection=None) -> int:
    """Confirm labelled evidence still exists in the local Chroma collection."""
    if collection is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        collection = chromadb.PersistentClient(
            path=str((ROOT / "data/db/chroma").resolve()),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        ).get_collection(COLLECTION)
    expected_ids = {chunk_id for case in cases for chunk_id in case.get("expected_chunk_ids", [])}
    records = collection.get(ids=list(expected_ids), include=["documents", "metadatas"])
    by_id = {chunk_id: (document, metadata) for chunk_id, document, metadata in zip(
        records["ids"], records["documents"], records["metadatas"])}
    missing = expected_ids - set(by_id)
    if missing:
        raise ValueError(f"Labelled chunks missing from collection: {sorted(missing)}")
    for case in cases:
        for evidence in case.get("evidence", []):
            for chunk_id in evidence["matching_chunk_ids"]:
                document, metadata = by_id[chunk_id]
                if evidence["quote"] not in document:
                    raise ValueError(f"Evidence quote missing from indexed chunk: {case['case_id']}")
                indexed_source = Path((metadata.get("source_path") or "").replace(
                    "\\", "/")).name
                if indexed_source != evidence["source"]:
                    raise ValueError(f"Evidence source differs from indexed chunk: {case['case_id']}")
    return collection.count()


def build_components():
    from scripts.query import build_query_components
    from src.core.settings import load_settings

    settings = load_settings()
    hybrid, reranker = build_query_components(settings, COLLECTION)
    if not reranker.is_enabled:
        raise RuntimeError("Configured reranker is disabled")
    return settings, hybrid, reranker


def retrieval(cases: list[dict], hybrid, reranker, rrf_k: int = 60) -> dict:
    from src.core.query_engine.fusion import RRFFusion

    fusion = RRFFusion(k=rrf_k)
    rows = []
    for case in cases:
        if case["answerability"] != "answerable":
            continue
        t0 = time.monotonic()
        detail = hybrid.search(case["query"], top_k=10, return_details=True)
        retrieval_ms = (time.monotonic() - t0) * 1000
        if detail.used_fallback or detail.dense_error or detail.sparse_error:
            raise RuntimeError(f"Retrieval fallback for {case['case_id']}: "
                               f"dense={detail.dense_error}, sparse={detail.sparse_error}")
        dense_candidates = detail.dense_results or []
        sparse_candidates = detail.sparse_results or []
        if not dense_candidates or not sparse_candidates:
            raise RuntimeError(f"Empty retrieval route for {case['case_id']}")
        dense = dense_candidates[:5]
        bm25 = sparse_candidates[:5]
        fused_candidates = fusion.fuse([dense_candidates, sparse_candidates], top_k=10)
        if rrf_k == getattr(getattr(hybrid, "fusion", None), "k", None):
            if [r.chunk_id for r in fused_candidates] != [r.chunk_id for r in detail.results]:
                raise RuntimeError(f"Offline RRF differs from configured search: {case['case_id']}")
        fused = fused_candidates[:5]
        t0 = time.monotonic()
        reranked = reranker.rerank(case["query"], fused_candidates, top_k=5)
        rerank_ms = (time.monotonic() - t0) * 1000
        if reranked.used_fallback:
            raise RuntimeError(f"Reranker fallback for {case['case_id']}: "
                               f"{reranked.fallback_reason}")
        expected = set(case["expected_sources"])
        expected_chunks = set(case["expected_chunk_ids"])
        row = {"case_id": case["case_id"], "query": case["query"],
               "category": case.get("category", "unclassified"),
               "expected_sources": sorted(expected),
               "expected_chunk_ids": sorted(expected_chunks),
               "required_chunk_ids": case.get("required_chunk_ids", []),
               "dense_candidates": [{"chunk_id": r.chunk_id, "source": source_name(r)}
                                    for r in dense_candidates],
               "bm25_candidates": [{"chunk_id": r.chunk_id, "source": source_name(r)}
                                   for r in sparse_candidates],
               "suspect_image_chunk_ids": [r.chunk_id for r in fused_candidates
                                           if "[IMAGE:" in (r.text or "")],
               "retrieval_ms": round(retrieval_ms, 1), "rerank_ms": round(rerank_ms, 1)}
        for name, results in (("dense", dense), ("bm25", bm25), ("rrf", fused),
                              ("rrf_rerank", reranked.results)):
            row[name] = {"rank": rank_of(results, expected),
                         "chunk_rank": chunk_rank_of(results, expected_chunks),
                         "top_5": [{"chunk_id": r.chunk_id,
                                    "source": source_name(r)} for r in results[:5]]}
            required = set(case.get("required_sources", []))
            if required:
                row[name]["all_required_at_5"] = required <= {
                    source_name(r) for r in results[:5]}
            required_chunks = set(case.get("required_chunk_ids", []))
            if required_chunks:
                row[name]["all_required_chunks_at_5"] = required_chunks <= {
                    r.chunk_id for r in results[:5]}
        rows.append(row)
        print(f"{case['case_id']}: chunk rank Dense={row['dense']['chunk_rank']} "
              f"BM25={row['bm25']['chunk_rank']} "
              f"RRF={row['rrf']['chunk_rank']} "
              f"RRF+rerank={row['rrf_rerank']['chunk_rank']}")
    if not rows:
        raise ValueError("No answerable cases")
    cross_document = [r for r in rows if "all_required_at_5" in r["dense"]]
    arms = ("dense", "bm25", "rrf", "rrf_rerank")
    summary = {arm: summarize(rows, arm) for arm in arms}
    for arm in summary:
        chunk_ranks = [r[arm]["chunk_rank"] for r in rows]
        summary[arm]["chunk_hit_at_5"] = round(
            sum(rank is not None for rank in chunk_ranks) / len(rows), 4)
        summary[arm]["chunk_mrr_at_5"] = round(
            sum(1 / rank for rank in chunk_ranks if rank is not None) / len(rows), 4)
        summary[arm]["chunk_top_1"] = round(
            sum(rank == 1 for rank in chunk_ranks) / len(rows), 4)
        summary[arm]["all_required_at_5"] = (
            round(sum(r[arm]["all_required_at_5"] for r in cross_document)
                  / len(cross_document), 4) if cross_document else None)
        summary[arm]["all_required_chunks_at_5"] = (
            round(sum(r[arm]["all_required_chunks_at_5"] for r in cross_document)
                  / len(cross_document), 4) if cross_document else None)
    categories = {}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        categories[category] = {
            arm: {"n": len(subset),
                  "chunk_hit_at_5": round(sum(r[arm]["chunk_rank"] is not None
                                               for r in subset) / len(subset), 4),
                  "chunk_top_1": round(sum(r[arm]["chunk_rank"] == 1
                                            for r in subset) / len(subset), 4),
                  "chunk_mrr_at_5": round(sum(
                      1 / r[arm]["chunk_rank"] for r in subset
                      if r[arm]["chunk_rank"] is not None) / len(subset), 4),
                  "all_required_chunks_at_5": (
                      round(sum(r[arm]["all_required_chunks_at_5"] for r in subset)
                            / len(subset), 4)
                      if all("all_required_chunks_at_5" in r[arm] for r in subset)
                      else None)}
            for arm in arms
        }
    return {"label_status": "agent_draft; human review required",
            "metric_unit": "source document and evidence chunk (any expected item)",
            "rrf_k": rrf_k,
            "n_cross_document": len(cross_document),
            "cases_with_suspect_image_chunks": sum(
                bool(r["suspect_image_chunk_ids"]) for r in rows),
            "candidate_pool": "same hybrid search, top 10 for RRF and reranker",
            "summary": summary,
            "by_category": categories,
            "mean_retrieval_ms": round(sum(r["retrieval_ms"] for r in rows) / len(rows), 1),
            "mean_extra_rerank_ms": round(sum(r["rerank_ms"] for r in rows) / len(rows), 1),
            "cases": rows}


def tune_rrf(report_path: Path) -> dict:
    """Score predeclared RRF k values from saved development rankings, without APIs."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "retrieval" or report.get("split") != "dev":
        raise ValueError("--input must be a development retrieval report")
    rows = report["cases"]
    scores = {}
    for k in (5, 20, 60):
        case_scores = []
        for row in rows:
            ranks = {}
            for arm in ("dense_candidates", "bm25_candidates"):
                for rank, item in enumerate(row[arm], 1):
                    ranks[item["chunk_id"]] = ranks.get(item["chunk_id"], 0) + 1 / (k + rank)
            top_5 = sorted(ranks, key=lambda chunk_id: (-ranks[chunk_id], chunk_id))[:5]
            rank = next((i for i, chunk_id in enumerate(top_5, 1)
                         if chunk_id in row["expected_chunk_ids"]), None)
            required = row.get("required_chunk_ids", [])
            case_scores.append({"case_id": row["case_id"], "category": row["category"],
                                "chunk_rank": rank,
                                "all_required_chunks_at_5": (
                                    set(required) <= set(top_5) if required else None)})
        cross = [r for r in case_scores if r["all_required_chunks_at_5"] is not None]
        scores[str(k)] = {
            "chunk_hit_at_5": round(sum(r["chunk_rank"] is not None for r in case_scores)
                                    / len(case_scores), 4),
            "chunk_mrr_at_5": round(sum(1 / r["chunk_rank"] for r in case_scores
                                      if r["chunk_rank"] is not None) / len(case_scores), 4),
            "all_required_chunks_at_5": (
                round(sum(r["all_required_chunks_at_5"] for r in cross) / len(cross), 4)
                if cross else None),
            "cases": case_scores,
        }
    selected = max((5, 20, 60), key=lambda k: (
        scores[str(k)]["chunk_hit_at_5"],
        scores[str(k)]["all_required_chunks_at_5"] or 0,
        scores[str(k)]["chunk_mrr_at_5"],
        k == 60))
    return {"source_report": str(report_path.resolve()),
            "selection_rule": "highest chunk Hit@5, then all-required@5, then MRR@5; tie prefers k=60",
            "scores": scores, "selected_rrf_k": selected}


class StrictHybridForRefusal:
    """Reject silent retrieval fallback during answer status evaluation."""

    def __init__(self, hybrid):
        self.hybrid = hybrid

    def search(self, *args, **kwargs):
        detail = self.hybrid.search(*args, **{**kwargs, "return_details": True})
        if detail.used_fallback or detail.dense_error or detail.sparse_error:
            raise RuntimeError("Retrieval fallback during refusal evaluation: "
                               f"dense={detail.dense_error}, sparse={detail.sparse_error}")
        return detail.results


def refusal_scores(cases: list[dict], hybrid, reranker) -> dict:
    """Collect relevance scores without answer-generation API calls."""
    if not reranker.is_enabled:
        raise RuntimeError("An enabled reranker is required for score calibration")
    strict_hybrid = StrictHybridForRefusal(hybrid)
    rows = []
    for case in cases:
        if case["answerability"] == "partial":
            continue
        candidates = strict_hybrid.search(case["query"], top_k=10)
        reranked = reranker.rerank(case["query"], candidates, top_k=5)
        if reranked.used_fallback:
            raise RuntimeError(f"Reranker fallback for {case['case_id']}: "
                               f"{reranked.fallback_reason}")
        scores = [result.metadata.get("rerank_score") for result in reranked.results]
        if not scores or any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            for score in scores
        ):
            raise RuntimeError(f"Missing or invalid rerank scores for {case['case_id']}")
        row = {"case_id": case["case_id"], "query": case["query"],
               "expected": case["answerability"],
               "max_rerank_score": max(scores)}
        rows.append(row)
        print(f"{row['case_id']}: {row['expected']} max_rerank_score="
              f"{row['max_rerank_score']:.4f}")
    return {"scope": "reranker score calibration; no answer generation",
            "cases": rows}


def refusal(cases: list[dict], settings, hybrid, reranker,
            min_rerank_score: float | None = None) -> dict:
    from src.core.answering import RAGAnswerService
    from src.libs.llm.llm_factory import LLMFactory

    service = RAGAnswerService(hybrid_search=StrictHybridForRefusal(hybrid),
                               llm=LLMFactory.create(settings), reranker=reranker,
                               min_rerank_score=min_rerank_score)
    rows = []
    for case in cases:
        if case["answerability"] == "partial":
            continue  # Partial answers need separate manual content judgement.
        t0 = time.monotonic()
        response = service.answer(case["query"], top_k=5)
        if response.status == "error":
            raise RuntimeError(f"Answer error for {case['case_id']}: {response.metadata}")
        if response.status not in {"answered", "refused"}:
            raise RuntimeError(f"Unknown answer status for {case['case_id']}: {response.status}")
        row = {"case_id": case["case_id"], "query": case["query"],
               "expected": case["answerability"], "status": response.status,
               "answer": response.answer,
               "max_rerank_score": response.metadata.get("max_rerank_score"),
               "stop_reason": response.metadata.get("stop_reason"),
               "elapsed_ms": round((time.monotonic() - t0) * 1000, 1)}
        rows.append(row)
        print(f"{case['case_id']}: expected={case['answerability']} got={response.status}")
    negative = [r for r in rows if r["expected"] == "unanswerable"]
    positive = [r for r in rows if r["expected"] == "answerable"]
    if not negative or not positive:
        raise ValueError("Need both answerable and unanswerable cases")
    return {"label_status": "agent_draft; human review required",
            "scope": "refusal decision only; answer factuality needs manual review",
            "min_rerank_score": min_rerank_score,
            "unanswerable_refusal_rate": round(sum(r["status"] == "refused" for r in negative) / len(negative), 4),
            "answerable_false_refusal_rate": round(sum(r["status"] == "refused" for r in positive) / len(positive), 4),
            "n_unanswerable": len(negative), "n_answerable": len(positive), "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "retrieval", "refusal", "scores", "tune"), required=True)
    parser.add_argument("--test-set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF smoothing parameter for retrieval mode (default: 60).")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-rerank-score", type=float,
                        help="Optional reranker relevance cutoff for refusal mode (0-1).")
    parser.add_argument("--input", type=Path, help="Development retrieval report for --mode tune.")
    args = parser.parse_args()
    try:
        if args.mode == "tune":
            if not args.input:
                raise ValueError("--mode tune requires --input")
            report = tune_rrf(args.input)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
            print(json.dumps({k: v for k, v in report.items() if k != "scores"},
                             ensure_ascii=False, indent=2))
            for k, metrics in report["scores"].items():
                print(f"k={k}: " + json.dumps({key: value for key, value in metrics.items()
                                             if key != "cases"}, ensure_ascii=False))
            return 0
        cases = load_cases(args.test_set)
        if args.rrf_k <= 0:
            raise ValueError("--rrf-k must be positive")
        print(f"Loaded {len(cases)} draft cases: "
              f"{sum(c['answerability'] == 'answerable' for c in cases)} answerable, "
              f"{sum(c['answerability'] == 'partial' for c in cases)} partial, "
              f"{sum(c['answerability'] == 'unanswerable' for c in cases)} unanswerable")
        if args.mode == "check":
            count = validate_snapshot(cases)
            print(f"Validated labelled chunks against {count} indexed chunks")
            return 0
        settings, hybrid, reranker = build_components()
        if args.mode == "retrieval":
            validate_snapshot(cases, hybrid.dense_retriever.vector_store.collection)
        if args.mode in {"refusal", "scores"} and args.rrf_k != hybrid.fusion.k:
            from src.core.query_engine.fusion import RRFFusion

            hybrid.fusion = RRFFusion(k=args.rrf_k)
        if args.mode == "retrieval":
            report = retrieval(cases, hybrid, reranker, args.rrf_k)
        elif args.mode == "scores":
            report = refusal_scores(cases, hybrid, reranker)
        else:
            report = refusal(cases, settings, hybrid, reranker,
                             min_rerank_score=args.min_rerank_score)
        report.update({"mode": args.mode, "collection": COLLECTION,
                       "rrf_k": args.rrf_k,
                       "split": json.loads(args.test_set.read_text(encoding="utf-8")).get("split"),
                       "test_set": str(args.test_set.resolve()),
                       "reranker": f"{settings.rerank.provider}/{settings.rerank.model}"})
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            print(f"Report: {args.output.resolve()}")
        print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "test_set"}},
                         ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
