"""The one-command pilot must keep dry runs offline and stop on stage errors."""

import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate_jd_full


def test_dry_run_prints_all_stages_without_running_subprocesses(
    monkeypatch, tmp_path, capsys,
):
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("Dry run called a subprocess")

    monkeypatch.setattr(evaluate_jd_full.subprocess, "run", unexpected_run)
    output = tmp_path / "full.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_jd_full.py", "--dry-run",
                                       "--output", str(output)])
    assert evaluate_jd_full.main() == 0
    printed = capsys.readouterr().out
    assert printed.index("check_retrieval") < printed.index("check_refusal")
    assert printed.index("check_refusal") < printed.index("retrieval_and_rerank")
    assert printed.index("retrieval_and_rerank") < printed.index("[refusal]")
    assert "offline_tuning" not in printed
    assert "jd_after_sales_retrieval_50_v3.json" in printed
    assert "jd_after_sales_refusal_10_v3.json" in printed
    assert not output.exists()


def test_failed_stage_stops_pipeline(monkeypatch):
    monkeypatch.setattr(evaluate_jd_full.subprocess, "run", lambda *_a, **_k:
                        SimpleNamespace(returncode=2))
    with pytest.raises(RuntimeError, match="check_retrieval failed"):
        evaluate_jd_full.run_stage("check_retrieval", ["--mode", "check"])
