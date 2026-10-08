#!/usr/bin/env python
"""Run the formal generation-quality evaluation on the portfolio golden set."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from src.libs.llm.llm_factory import LLMFactory  # noqa: E402
from src.libs.vector_store.vector_store_factory import VectorStoreFactory  # noqa: E402
from src.observability.evaluation.eval_runner import load_test_set  # noqa: E402
from src.observability.evaluation.ragas_evaluator import RagasEvaluator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate real RAG answers with Ragas Faithfulness and rule checks."
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set_portfolio.json",
    )
    parser.add_argument("--collection", default="portfolio")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default="reports/portfolio_generation_baseline.json",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
    )
    parser.add_argument(
        "--resume-judges",
        action="store_true",
        help="Reuse answers in --output and retry only missing Ragas scores.",
    )
    return parser.parse_args()


def _expected_status(answerability: str) -> str:
    return "refused" if answerability == "unanswerable" else "answered"


def _citation_dict(citation: Any) -> dict[str, Any]:
    if hasattr(citation, "to_dict"):
        return citation.to_dict()
    return {
        "index": citation.index,
        "chunk_id": citation.chunk_id,
        "source": citation.source,
    }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    answered = [row for row in rows if row["status"] == "answered"]
    faithfulness = [
        row["faithfulness"]
        for row in rows
        if isinstance(row.get("faithfulness"), (int, float))
    ]
    citation_rows = [row for row in answered if row.get("citation_valid") is not None]

    return {
        "evaluated_count": total,
        "answered_count": len(answered),
        "refused_count": sum(row["status"] == "refused" for row in rows),
        "error_count": sum(row["status"] == "error" for row in rows),
        "judge_error_count": sum(bool(row.get("judge_error")) for row in rows),
        "status_accuracy": (
            sum(bool(row["status_correct"]) for row in rows) / total if total else 0.0
        ),
        "citation_coverage": (
            sum(bool(row["citations"]) for row in answered) / len(answered)
            if answered
            else 0.0
        ),
        "citation_validity": (
            sum(bool(row["citation_valid"]) for row in citation_rows)
            / len(citation_rows)
            if citation_rows
            else 0.0
        ),
        "faithfulness_count": len(faithfulness),
        "faithfulness_mean": (
            sum(faithfulness) / len(faithfulness) if faithfulness else None
        ),
    }


def _write_report(
    output_path: Path,
    *,
    complete: bool,
    args: argparse.Namespace,
    ragas_version: str,
    total_cases: int,
    rows: list[dict[str, Any]],
) -> None:
    report = {
        "report_type": "portfolio_generation_baseline",
        "complete": complete,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "test_set": args.test_set,
        "collection": args.collection,
        "top_k": args.top_k,
        "ragas_version": ragas_version,
        "metrics": _summarise(rows),
        "expected_case_count": total_cases,
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)

    try:
        settings = load_settings(args.config)
        cases = load_test_set(args.test_set)
        vector_store = VectorStoreFactory.create(
            settings,
            collection_name=args.collection,
        )
        faithfulness_evaluator = RagasEvaluator(
            settings=settings,
            metrics=["faithfulness"],
        )
        import ragas

        ragas_version = ragas.__version__
    except Exception as exc:
        print(f"[FAIL] 初始化生成评测失败: {exc}", file=sys.stderr)
        return 2

    if args.resume_judges:
        if not output_path.exists():
            print(f"[FAIL] 找不到待恢复报告: {output_path}", file=sys.stderr)
            return 2
        try:
            existing_report = json.loads(output_path.read_text(encoding="utf-8"))
            rows = existing_report["results"]
        except (KeyError, json.JSONDecodeError, OSError) as exc:
            print(f"[FAIL] 无法读取待恢复报告: {exc}", file=sys.stderr)
            return 2

        if len(rows) != len(cases):
            print(
                f"[FAIL] 报告题数为{len(rows)}，Golden Set题数为{len(cases)}。",
                file=sys.stderr,
            )
            return 2

        _write_report(
            output_path,
            complete=False,
            args=args,
            ragas_version=ragas_version,
            total_cases=len(cases),
            rows=rows,
        )
        for index, row in enumerate(rows, start=1):
            if row["status"] != "answered" or row.get("faithfulness") is not None:
                continue
            print(f"[{index}/{len(rows)}] 补算Faithfulness: {row['case_id']}")
            try:
                retrieved_ids = row["retrieved_chunk_ids"]
                context_records = vector_store.get_by_ids(retrieved_ids)
                if len(context_records) != len(retrieved_ids):
                    raise RuntimeError(
                        "Could not reload every retrieved Chunk for Ragas evaluation."
                    )
                row["faithfulness"] = faithfulness_evaluator.evaluate(
                    query=row["query"],
                    retrieved_chunks=context_records,
                    generated_answer=row["answer"],
                )["faithfulness"]
                row["judge_error"] = None
                print(f"    faithfulness={row['faithfulness']:.4f}")
            except Exception as exc:
                row["judge_error"] = str(exc)
                print(f"    [WARN] Ragas失败: {exc}")

            _write_report(
                output_path,
                complete=False,
                args=args,
                ragas_version=ragas_version,
                total_cases=len(cases),
                rows=rows,
            )

        _write_report(
            output_path,
            complete=True,
            args=args,
            ragas_version=ragas_version,
            total_cases=len(cases),
            rows=rows,
        )
        metrics = _summarise(rows)
        print("\n" + "=" * 72)
        print("GENERATION EVALUATION")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"报告: {output_path.resolve()}")
        return 1 if metrics["error_count"] or metrics["judge_error_count"] else 0

    try:
        hybrid_search, reranker = build_query_components(settings, args.collection)
        if not reranker.is_enabled:
            raise RuntimeError("Configured reranker is not active.")
        answer_llm = LLMFactory.create(settings)
        answer_service = RAGAnswerService(
            hybrid_search=hybrid_search,
            llm=answer_llm,
            reranker=reranker,
        )
    except Exception as exc:
        print(f"[FAIL] 初始化答案链路失败: {exc}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    _write_report(
        output_path,
        complete=False,
        args=args,
        ragas_version=ragas_version,
        total_cases=len(cases),
        rows=rows,
    )

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}: {case.query}")
        response = answer_service.answer(case.query, top_k=args.top_k)
        retrieved_ids = list(response.metadata.get("retrieved_chunk_ids", []))
        citations = [_citation_dict(item) for item in response.citations]
        cited_ids = [item["chunk_id"] for item in citations]
        citation_valid = (
            all(chunk_id in retrieved_ids for chunk_id in cited_ids)
            if response.status == "answered"
            else None
        )

        faithfulness: float | None = None
        judge_error: str | None = None
        if response.status == "answered":
            try:
                context_records = vector_store.get_by_ids(retrieved_ids)
                if len(context_records) != len(retrieved_ids):
                    raise RuntimeError(
                        "Could not reload every retrieved Chunk for Ragas evaluation."
                    )
                faithfulness = faithfulness_evaluator.evaluate(
                    query=case.query,
                    retrieved_chunks=context_records,
                    generated_answer=response.answer,
                )["faithfulness"]
            except Exception as exc:
                judge_error = str(exc)

        row = {
            "case_id": case.case_id,
            "query": case.query,
            "answerability": case.answerability,
            "expected_status": _expected_status(case.answerability),
            "status": response.status,
            "status_correct": response.status == _expected_status(case.answerability),
            "answer": response.answer,
            "reference_answer": case.reference_answer,
            "retrieved_chunk_ids": retrieved_ids,
            "citations": citations,
            "citation_valid": citation_valid,
            "faithfulness": faithfulness,
            "judge_error": judge_error,
            "model": response.metadata.get("model"),
            "usage": response.metadata.get("usage"),
            "stop_reason": response.metadata.get("stop_reason"),
            "system_error": response.metadata.get("error"),
        }
        rows.append(row)
        _write_report(
            output_path,
            complete=False,
            args=args,
            ragas_version=ragas_version,
            total_cases=len(cases),
            rows=rows,
        )

        score_text = "-" if faithfulness is None else f"{faithfulness:.4f}"
        print(
            f"    status={response.status}, status_correct={row['status_correct']}, "
            f"citations={len(citations)}, faithfulness={score_text}"
        )
        if judge_error:
            print(f"    [WARN] Ragas失败: {judge_error}")

    _write_report(
        output_path,
        complete=True,
        args=args,
        ragas_version=ragas_version,
        total_cases=len(cases),
        rows=rows,
    )
    metrics = _summarise(rows)
    print("\n" + "=" * 72)
    print("GENERATION EVALUATION")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"报告: {output_path.resolve()}")

    return 1 if metrics["error_count"] or metrics["judge_error_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
