"""LangChain tool wrappers that proxy dataset analysis through the MCP server."""

from __future__ import annotations

import asyncio
import concurrent.futures
from functools import lru_cache
from typing import Annotated, Any

from fastmcp import Client
from langchain_core.tools import tool

from customer_support_agent.config import get_mcp_server_url
from customer_support_agent.tools import (
    DistributionInput,
    EmptyInput,
    ExamplesInput,
    FilterInput,
)


REQUIRED_MCP_TOOLS = {
    "list_categories",
    "list_intents",
    "count_rows",
    "show_examples",
    "distribution",
    "sample_responses_for_summary",
}


def _normalize_mcp_server_url(mcp_server_url: str | None) -> str:
    return (mcp_server_url or get_mcp_server_url()).strip()


async def _call_mcp_tool_async(
    mcp_server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    async with Client(mcp_server_url) as client:
        result = await client.call_tool(tool_name, arguments)
        return result.data


def _call_mcp_tool(
    mcp_server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call an MCP tool synchronously, safe to use from both sync and async contexts.

    ``asyncio.run()`` raises RuntimeError when there is already a running event
    loop (e.g. inside Streamlit or a Jupyter notebook).  In that case we spin up
    a fresh loop in a dedicated background thread so the caller's loop is not
    blocked or corrupted.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop is not None and loop.is_running():
            # Already inside an event loop — run the coroutine in a separate thread
            # that has its own event loop.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    _call_mcp_tool_async(mcp_server_url, tool_name, arguments),
                )
                return future.result()
        else:
            return asyncio.run(_call_mcp_tool_async(mcp_server_url, tool_name, arguments))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to call MCP tool '{tool_name}' at {mcp_server_url}: {exc}"
        ) from exc


def verify_mcp_server(mcp_server_url: str | None = None) -> None:
    """Fail fast if the standalone MCP server is unavailable or incomplete."""

    url = _normalize_mcp_server_url(mcp_server_url)

    async def _verify() -> None:
        async with Client(url) as client:
            tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}
        missing = sorted(REQUIRED_MCP_TOOLS.difference(tool_names))
        if missing:
            raise RuntimeError(f"MCP server is missing required tools: {missing}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, _verify()).result()
        else:
            asyncio.run(_verify())
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to the MCP server at {url}. "
            "Start `python mcp_server.py` first, then rerun the client with `--use-mcp`."
        ) from exc


@lru_cache(maxsize=None)
def get_mcp_dataset_tools(mcp_server_url: str | None = None) -> list:
    """Return LangChain tools that proxy dataset analysis through the MCP server."""

    url = _normalize_mcp_server_url(mcp_server_url)

    @tool("list_categories", args_schema=EmptyInput)
    def list_categories() -> dict[str, list[str]]:
        """List every category available in the Bitext customer support dataset."""

        return _call_mcp_tool(url, "list_categories", {})

    @tool("list_intents", args_schema=FilterInput)
    def list_intents(
        category: str | None = None,
        intent: str | None = None,
        search_text: str | None = None,
    ) -> dict:
        """List available exact intent labels, optionally narrowed by category or text search."""

        return _call_mcp_tool(
            url,
            "list_intents",
            {"category": category, "intent": intent, "search_text": search_text},
        )

    @tool("count_rows", args_schema=FilterInput)
    def count_rows(
        category: str | None = None,
        intent: str | None = None,
        search_text: str | None = None,
    ) -> dict:
        """Count dataset rows after optional category, intent, and text filters."""

        return _call_mcp_tool(
            url,
            "count_rows",
            {"category": category, "intent": intent, "search_text": search_text},
        )

    @tool("show_examples", args_schema=ExamplesInput)
    def show_examples(
        category: str | None = None,
        intent: str | None = None,
        search_text: str | None = None,
        limit: int = 3,
        offset: int = 0,
    ) -> dict:
        """Return matching examples for listing, including each customer query and support response."""

        return _call_mcp_tool(
            url,
            "show_examples",
            {
                "category": category,
                "intent": intent,
                "search_text": search_text,
                "limit": limit,
                "offset": offset,
            },
        )

    @tool("distribution", args_schema=DistributionInput)
    def distribution(
        group_by: Annotated[str, "category, intent, or flags"],
        category: str | None = None,
        intent: str | None = None,
        top_n: int = 30,
    ) -> dict:
        """Calculate a count distribution for category, intent, or flags in the dataset."""

        return _call_mcp_tool(
            url,
            "distribution",
            {
                "group_by": group_by,
                "category": category,
                "intent": intent,
                "top_n": top_n,
            },
        )

    @tool("sample_responses_for_summary", args_schema=ExamplesInput)
    def sample_responses_for_summary(
        category: str | None = None,
        intent: str | None = None,
        search_text: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Collect response samples for open-ended summaries about how agents reply."""

        return _call_mcp_tool(
            url,
            "sample_responses_for_summary",
            {
                "category": category,
                "intent": intent,
                "search_text": search_text,
                "limit": limit,
                "offset": offset,
            },
        )

    return [
        list_categories,
        list_intents,
        count_rows,
        show_examples,
        distribution,
        sample_responses_for_summary,
    ]
