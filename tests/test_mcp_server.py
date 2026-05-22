"""Tests for the FastMCP server."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd

from customer_support_agent import dataset
from mcp_server import _is_loopback_host, _schedule_graceful_shutdown, mcp, shutdown_server


def test_mcp_server_exposes_dataset_tools(monkeypatch, capsys) -> None:
    frame = pd.DataFrame(
        [
            {
                "flags": "",
                "instruction": "Where is my refund?",
                "category": "REFUND",
                "intent": "get_refund",
                "response": "We can help with your refund.",
            },
            {
                "flags": "",
                "instruction": "Track my package",
                "category": "SHIPPING",
                "intent": "track_order",
                "response": "Here is how to track your shipment.",
            },
        ]
    )
    monkeypatch.setattr(dataset, "load_customer_support_data", lambda: frame)

    async def run() -> None:
        from fastmcp import Client

        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools}
            assert {"list_categories", "count_rows", "show_examples"} <= tool_names

            categories = await client.call_tool("list_categories")
            assert categories.data["categories"] == ["REFUND", "SHIPPING"]

            refund_count = await client.call_tool("count_rows", {"category": "REFUND"})
            assert refund_count.data["row_count"] == 1

    asyncio.run(run())

    captured = capsys.readouterr()
    assert "[mcp tool call] list_categories({})" in captured.out
    assert "[mcp observation] list_categories:" in captured.out
    assert '[mcp tool call] count_rows({"category": "REFUND", "intent": null, "search_text": null})' in captured.out


def test_shutdown_route_rejects_non_local_client() -> None:
    request = SimpleNamespace(client=SimpleNamespace(host="192.168.1.20"))

    response = asyncio.run(shutdown_server(request))

    assert response.status_code == 403


def test_schedule_graceful_shutdown_uses_sigterm(monkeypatch) -> None:
    calls: list[tuple[float, object, object, object]] = []

    class FakeLoop:
        def call_later(self, delay, callback, pid, sig):
            calls.append((delay, callback, pid, sig))

    monkeypatch.setattr("mcp_server.asyncio.get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr("mcp_server.os.getpid", lambda: 4321)

    _schedule_graceful_shutdown()

    assert calls == [(0.2, __import__("os").kill, 4321, __import__("signal").SIGTERM)]


def test_loopback_host_helper() -> None:
    assert _is_loopback_host("127.0.0.1") is True
    assert _is_loopback_host("::1") is True
    assert _is_loopback_host("localhost") is True
    assert _is_loopback_host("192.168.1.10") is False
