"""FastMCP server exposing customer-support dataset tools."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import signal
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from customer_support_agent.config import (
    get_mcp_server_host,
    get_mcp_server_path,
    get_mcp_server_port,
)
from customer_support_agent.tools import (
    count_rows_impl,
    distribution_impl,
    list_categories_impl,
    list_intents_impl,
    sample_responses_for_summary_impl,
    show_examples_impl,
)


mcp = FastMCP(
    name="CustomerSupportDatasetMCP",
    instructions=(
        "This MCP server exposes analysis tools for the Bitext customer support dataset. "
        "Use these tools to inspect categories, intents, counts, examples, and grouped distributions."
    ),
)


def _log_mcp_tool_call(name: str, **kwargs: Any) -> None:
    print(f"[mcp tool call] {name}({json.dumps(kwargs, ensure_ascii=True, sort_keys=True)})")


def _log_mcp_tool_result(name: str, result: dict[str, Any]) -> None:
    preview = json.dumps(result, ensure_ascii=True)[:1200]
    print(f"[mcp observation] {name}: {preview}")


def _run_logged_tool(name: str, fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    _log_mcp_tool_call(name, **kwargs)
    result = fn(**kwargs)
    _log_mcp_tool_result(name, result)
    return result


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _schedule_graceful_shutdown(delay_seconds: float = 0.2) -> None:
    loop = asyncio.get_running_loop()
    loop.call_later(delay_seconds, os.kill, os.getpid(), signal.SIGTERM)


@mcp.custom_route("/shutdown", methods=["POST"], include_in_schema=False)
async def shutdown_server(request: Request) -> JSONResponse:
    """Gracefully stop the local MCP server."""

    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        return JSONResponse(
            {"ok": False, "error": "shutdown is allowed only from localhost"},
            status_code=403,
        )

    _schedule_graceful_shutdown()
    return JSONResponse({"ok": True, "message": "MCP server shutdown scheduled."})


@mcp.tool
def list_categories() -> dict[str, list[str]]:
    """List every category available in the Bitext customer support dataset."""

    return _run_logged_tool("list_categories", list_categories_impl)


@mcp.tool
def list_intents(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
) -> dict:
    """List available exact intent labels, optionally narrowed by category or text search."""

    return _run_logged_tool(
        "list_intents",
        list_intents_impl,
        category=category,
        intent=intent,
        search_text=search_text,
    )


@mcp.tool
def count_rows(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
) -> dict:
    """Count dataset rows after optional category, intent, and text filters."""

    return _run_logged_tool(
        "count_rows",
        count_rows_impl,
        category=category,
        intent=intent,
        search_text=search_text,
    )


@mcp.tool
def show_examples(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
    limit: int = 3,
    offset: int = 0,
) -> dict:
    """Return matching examples including each customer instruction and support response."""

    return _run_logged_tool(
        "show_examples",
        show_examples_impl,
        category=category,
        intent=intent,
        search_text=search_text,
        limit=limit,
        offset=offset,
    )


@mcp.tool
def distribution(
    group_by: str,
    category: str | None = None,
    intent: str | None = None,
    top_n: int = 30,
) -> dict:
    """Calculate a count distribution for category, intent, or flags in the dataset."""

    return _run_logged_tool(
        "distribution",
        distribution_impl,
        group_by=group_by,
        category=category,
        intent=intent,
        top_n=top_n,
    )


@mcp.tool
def sample_responses_for_summary(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """Collect representative response samples for qualitative summaries."""

    return _run_logged_tool(
        "sample_responses_for_summary",
        sample_responses_for_summary_impl,
        category=category,
        intent=intent,
        search_text=search_text,
        limit=limit,
        offset=offset,
    )


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=get_mcp_server_host(),
        port=get_mcp_server_port(),
        path=get_mcp_server_path(),
    )
