"""Tests for command-line entry points."""

from __future__ import annotations

import sys

from customer_support_agent import cli


def test_use_mcp_flag_validates_server_and_invokes_agent(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(sys, "argv", ["main.py", "--once", "How many refunds?", "--use-mcp"])
    monkeypatch.setattr(cli, "verify_mcp_server", lambda url: captured.setdefault("verified_url", url))
    def fake_invoke_agent(*args, **kwargs):
        captured["invoke_kwargs"] = kwargs
        return []

    monkeypatch.setattr(cli, "invoke_agent", fake_invoke_agent)
    monkeypatch.setattr(cli, "_print_reasoning", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "final_answer", lambda _messages: "done")

    cli.main()

    assert captured["verified_url"] == "http://127.0.0.1:8000/mcp"
    invoke_kwargs = captured["invoke_kwargs"]
    assert invoke_kwargs["use_mcp"] is True
    assert invoke_kwargs["mcp_server_url"] == "http://127.0.0.1:8000/mcp"

    output = capsys.readouterr().out
    assert "Assistant: done" in output
