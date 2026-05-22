"""Tests for MCP-backed client tool wrappers."""

from __future__ import annotations

from customer_support_agent.mcp_tools import get_mcp_dataset_tools, verify_mcp_server


class FakeTool:
    def __init__(self, name: str):
        self.name = name


class FakeCallResult:
    def __init__(self, data):
        self.data = data


class FakeClient:
    def __init__(self, _url: str):
        self.url = _url

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self):
        return [
            FakeTool("list_categories"),
            FakeTool("list_intents"),
            FakeTool("count_rows"),
            FakeTool("show_examples"),
            FakeTool("distribution"),
            FakeTool("sample_responses_for_summary"),
        ]

    async def call_tool(self, name: str, arguments: dict):
        return FakeCallResult({"tool_name": name, "arguments": arguments})


def test_verify_mcp_server_accepts_required_toolset(monkeypatch) -> None:
    monkeypatch.setattr("customer_support_agent.mcp_tools.Client", FakeClient)

    verify_mcp_server("http://127.0.0.1:8000/mcp")


def test_mcp_dataset_tool_proxies_call_remote_client(monkeypatch) -> None:
    monkeypatch.setattr("customer_support_agent.mcp_tools.Client", FakeClient)

    tools = {tool.name: tool for tool in get_mcp_dataset_tools("http://127.0.0.1:8000/mcp")}
    result = tools["count_rows"].invoke({"category": "REFUND"})

    assert result["tool_name"] == "count_rows"
    assert result["arguments"] == {
        "category": "REFUND",
        "intent": None,
        "search_text": None,
    }
