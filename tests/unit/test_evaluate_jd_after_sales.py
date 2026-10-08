"""Checks for the paired JD pilot evaluation's scoring and failure protection."""

import json
from types import SimpleNamespace

import pytest

from scripts.evaluate_jd_after_sales import (
    StrictHybridForRefusal, chunk_rank_of, rank_of, refusal_scores, retrieval, source_name,
    summarize, tune_rrf,
)


def result(source: str, chunk_id: str):
    return SimpleNamespace(metadata={"source_path": f"data/documents/jd_after_sales/{source}"},
                           chunk_id=chunk_id, score=1.0, text="evidence")


def test_source_metrics_use_rank_and_fixed_denominator():
    rows = [
        {"dense": {"rank": 1}},
        {"dense": {"rank": 4}},
        {"dense": {"rank": None}},
    ]
    assert summarize(rows, "dense") == {
        "n": 3, "hit_at_5": 0.6667, "mrr_at_5": 0.4167, "top_1": 0.3333,
    }
    assert source_name(result("policy.md", "a")) == "policy.md"
    assert rank_of([result("other.md", "a"), result("policy.md", "b")],
                   {"policy.md"}) == 2
    assert chunk_rank_of([result("policy.md", "a")], {"b"}) is None


def test_hybrid_fallback_fails_instead_of_counting_as_rrf():
    class Hybrid:
        def search(self, *_args, **_kwargs):
            return SimpleNamespace(used_fallback=True, dense_error=None,
                                   sparse_error="BM25 unavailable")

    cases = [{"case_id": "case_1", "query": "question",
              "answerability": "answerable", "expected_sources": ["policy.md"],
              "expected_chunk_ids": ["a"]}]
    with pytest.raises(RuntimeError, match="Retrieval fallback"):
        retrieval(cases, Hybrid(), None)


def test_reranker_fallback_fails_instead_of_counting_as_improvement():
    one = result("policy.md", "a")

    class Hybrid:
        def search(self, *_args, **_kwargs):
            return SimpleNamespace(used_fallback=False, dense_error=None,
                                   sparse_error=None, dense_results=[one],
                                   sparse_results=[one], results=[one])

    class Reranker:
        def rerank(self, *_args, **_kwargs):
            return SimpleNamespace(used_fallback=True,
                                   fallback_reason="provider unavailable")

    cases = [{"case_id": "case_1", "query": "question",
              "answerability": "answerable", "expected_sources": ["policy.md"],
              "expected_chunk_ids": ["a"]}]
    with pytest.raises(RuntimeError, match="Reranker fallback"):
        retrieval(cases, Hybrid(), Reranker())


def test_tuning_uses_only_saved_development_rankings(tmp_path):
    report = {
        "mode": "retrieval", "split": "dev",
        "cases": [{
            "case_id": "dev_1", "category": "lexical",
            "expected_chunk_ids": ["target"], "required_chunk_ids": [],
            "dense_candidates": [{"chunk_id": "target"}, {"chunk_id": "noise"}],
            "bm25_candidates": [{"chunk_id": "target"}, {"chunk_id": "noise"}],
        }],
    }
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = tune_rrf(path)
    assert result["selected_rrf_k"] == 60  # Predeclared tie rule.
    assert result["scores"]["5"]["chunk_hit_at_5"] == 1.0
    report["split"] = "holdout"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="development"):
        tune_rrf(path)


def test_retrieval_reports_bm25_and_category_from_shared_routes():
    target = result("policy.md", "target")
    noise = result("other.md", "noise")

    class Hybrid:
        fusion = None

        def search(self, *_args, **_kwargs):
            return SimpleNamespace(used_fallback=False, dense_error=None,
                                   sparse_error=None, dense_results=[noise, target],
                                   sparse_results=[target, noise], results=[])

    class Reranker:
        def rerank(self, _query, candidates, **_kwargs):
            return SimpleNamespace(used_fallback=False, results=candidates[:5])

    cases = [{"case_id": "lexical_1", "query": "SN code", "category": "lexical",
              "answerability": "answerable", "expected_sources": ["policy.md"],
              "expected_chunk_ids": ["target"]}]
    report = retrieval(cases, Hybrid(), Reranker())
    assert report["cases"][0]["dense"]["chunk_rank"] == 2
    assert report["cases"][0]["bm25"]["chunk_rank"] == 1
    assert report["by_category"]["lexical"]["bm25"]["chunk_hit_at_5"] == 1.0


def test_refusal_retrieval_fallback_is_not_silently_scored():
    class Hybrid:
        def search(self, *_args, **kwargs):
            assert kwargs["return_details"] is True
            return SimpleNamespace(used_fallback=True, dense_error="offline",
                                   sparse_error=None, results=[])

    with pytest.raises(RuntimeError, match="Retrieval fallback"):
        StrictHybridForRefusal(Hybrid()).search("question", return_details=False)


def test_refusal_scores_uses_reranker_without_answer_generation():
    candidate = result("policy.md", "evidence")

    class Hybrid:
        def search(self, *_args, **kwargs):
            assert kwargs["return_details"] is True
            return SimpleNamespace(used_fallback=False, dense_error=None,
                                   sparse_error=None, results=[candidate])

    class Reranker:
        is_enabled = True

        def rerank(self, _query, _candidates, **_kwargs):
            ranked = result("policy.md", "evidence")
            ranked.metadata["rerank_score"] = 0.82
            return SimpleNamespace(used_fallback=False, results=[ranked])

    cases = [{"case_id": "case_1", "query": "question", "answerability": "answerable"}]
    report = refusal_scores(cases, Hybrid(), Reranker())

    assert report["cases"][0]["max_rerank_score"] == 0.82
