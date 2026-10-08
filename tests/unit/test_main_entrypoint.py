"""The installed command must launch the real server without stdout noise."""

import main as entrypoint
from src.mcp_server import server


def test_main_delegates_to_server_without_stdout(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(server, "main", lambda: calls.append("started") or 7)

    assert entrypoint.main() == 7
    assert calls == ["started"]
    assert capsys.readouterr().out == ""
