#!/usr/bin/env python
"""Run paired retrieval, reranking, and refusal experiments in one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts/evaluate_jd_after_sales.py"
RETRIEVAL_SET = "tests/fixtures/jd_after_sales_retrieval_50_v3.json"
REFUSAL_SET = "tests/fixtures/jd_after_sales_refusal_10_v3.json"
RRF_K = "60"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_stage(name: str, arguments: list[str], dry_run: bool = False) -> None:
    command = [sys.executable, str(EVALUATOR), *arguments]
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/jd_after_sales_full_pilot.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stage order without calling retrieval or model APIs.")
    args = parser.parse_args()
    output = args.output.resolve()
    stage_paths = {name: output.with_name(f"{output.stem}_{name}.json")
                   for name in ("retrieval", "refusal")}
    try:
        for name, test_set in (("retrieval", RETRIEVAL_SET),
                               ("refusal", REFUSAL_SET)):
            run_stage(f"check_{name}", ["--mode", "check", "--test-set", test_set],
                      dry_run=args.dry_run)

        run_stage("retrieval_and_rerank", [
            "--mode", "retrieval", "--test-set", RETRIEVAL_SET,
            "--rrf-k", RRF_K, "--output", str(stage_paths["retrieval"]),
        ], dry_run=args.dry_run)
        run_stage("refusal", [
            "--mode", "refusal", "--test-set", REFUSAL_SET,
            "--rrf-k", RRF_K,
            "--output", str(stage_paths["refusal"]),
        ], dry_run=args.dry_run)

        if args.dry_run:
            print("\nDry run complete. No API calls or reports were written.")
            return 0

        retrieval = read_report(stage_paths["retrieval"])
        refusal = read_report(stage_paths["refusal"])
        if retrieval.get("mode") != "retrieval" or len(retrieval.get("cases", [])) != 50:
            raise RuntimeError("Incomplete retrieval stage report")
        if (refusal.get("mode") != "refusal" or
                refusal.get("n_unanswerable") != 10 or
                refusal.get("n_answerable") != 6):
            raise RuntimeError("Incomplete refusal stage report")
        summary = {
            "status": "complete_pilot",
            "label_status": "agent_draft; human review required",
            "collection": "jd_after_sales",
            "rrf_k": int(RRF_K),
            "retrieval": {"n": len(retrieval["cases"]),
                          "summary": retrieval["summary"],
                          "by_category": retrieval["by_category"],
                          "mean_extra_rerank_ms": retrieval["mean_extra_rerank_ms"]},
            "refusal": {key: refusal[key] for key in (
                "n_answerable", "n_unanswerable", "unanswerable_refusal_rate",
                "answerable_false_refusal_rate")},
            "stage_reports": {key: str(path) for key, path in stage_paths.items()},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print("\nFULL PILOT SUMMARY")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Combined report: {output}")
        return 0
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Completed stage reports, if any, remain in the reports directory.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
