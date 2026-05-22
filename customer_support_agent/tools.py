"""LangChain tools for analyzing the customer support dataset."""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from customer_support_agent import dataset


class EmptyInput(BaseModel):
    """Input schema for tools that do not need parameters."""


class FilterInput(BaseModel):
    """Optional filters shared by several data tools."""

    category: str | None = Field(
        default=None,
        description="Dataset category such as ACCOUNT, ORDER, REFUND, SHIPPING, or FEEDBACK.",
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Exact dataset intent label such as get_refund, cancel_order, or track_order. "
            "Use this only when you already know the exact label."
        ),
    )
    search_text: str | None = Field(
        default=None,
        description=(
            "Plain text to search in customer instructions by default. "
            "Prefer this for natural-language requests like login issues, delayed delivery, "
            "billing, damaged items, or wanting money back."
        ),
    )


class ExamplesInput(FilterInput):
    """Input schema for the example retrieval tool."""

    limit: int = Field(default=3, ge=1, le=20, description="Maximum number of examples to return.")
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of matching examples to skip, useful for follow-up requests like 'show more'.",
    )


class DistributionInput(BaseModel):
    """Input schema for distribution calculations."""

    group_by: str = Field(
        description="Column to group by. Must be one of: category, intent, flags.",
        pattern="^(category|intent|flags)$",
    )
    category: str | None = Field(default=None, description="Optional category filter.")
    intent: str | None = Field(default=None, description="Optional intent filter.")
    top_n: int = Field(default=30, ge=1, le=100, description="Maximum number of groups to return.")


def list_categories_impl() -> dict[str, list[str]]:
    """List every category available in the Bitext customer support dataset."""

    frame = dataset.load_customer_support_data()
    return {"categories": sorted(frame["category"].dropna().unique().tolist())}


def list_intents_impl(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
) -> dict:
    """List available exact intent labels, optionally narrowed by category or text search."""

    frame = dataset.filter_rows(
        category=category,
        intent=intent,
        search_text=search_text,
        search_columns=("instruction",),
    )
    return {
        "count": int(frame["intent"].nunique()),
        "intents": sorted(frame["intent"].dropna().unique().tolist()),
    }


def count_rows_impl(category: str | None = None, intent: str | None = None, search_text: str | None = None) -> dict:
    """Count dataset rows after optional category, intent, and text filters."""

    frame = dataset.filter_rows(
        category=category,
        intent=intent,
        search_text=search_text,
        search_columns=("instruction",),
    )
    return {
        "row_count": int(len(frame)),
        "filters": {"category": category, "intent": intent, "search_text": search_text},
    }


def show_examples_impl(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
    limit: int = 3,
    offset: int = 0,
) -> dict:
    """Return matching examples for listing, including each customer query and support response."""

    frame = dataset.filter_rows(
        category=category,
        intent=intent,
        search_text=search_text,
        search_columns=("instruction",),
    )
    examples = [
        dataset.row_to_public_dict(row)
        for _, row in frame.iloc[offset : offset + limit].iterrows()
    ]
    return {
        "total_matches": int(len(frame)),
        "offset": offset,
        "returned": len(examples),
        "examples": examples,
    }


def distribution_impl(
    group_by: Annotated[str, "category, intent, or flags"],
    category: str | None = None,
    intent: str | None = None,
    top_n: int = 30,
) -> dict:
    """Calculate a count distribution for category, intent, or flags in the dataset."""

    frame = dataset.filter_rows(category=category, intent=intent)
    counts = frame[group_by].value_counts().head(top_n)
    return {
        "group_by": group_by,
        "total_rows": int(len(frame)),
        "distribution": {str(key): int(value) for key, value in counts.items()},
    }


def sample_responses_for_summary_impl(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """Collect response samples for open-ended summaries about how agents reply."""

    frame = dataset.filter_rows(
        category=category,
        intent=intent,
        search_text=search_text,
        search_columns=("instruction", "response"),
    )
    examples = [
        dataset.row_to_public_dict(row)
        for _, row in frame.iloc[offset : offset + limit].iterrows()
    ]
    return {
        "total_matches": int(len(frame)),
        "sample_size": len(examples),
        "examples": examples,
    }


@tool("list_categories", args_schema=EmptyInput)
def list_categories() -> dict[str, list[str]]:
    """List every category available in the Bitext customer support dataset."""

    return list_categories_impl()


@tool("list_intents", args_schema=FilterInput)
def list_intents(category: str | None = None, intent: str | None = None, search_text: str | None = None) -> dict:
    """List available exact intent labels, optionally narrowed by category or text search."""

    return list_intents_impl(category=category, intent=intent, search_text=search_text)


@tool("count_rows", args_schema=FilterInput)
def count_rows(category: str | None = None, intent: str | None = None, search_text: str | None = None) -> dict:
    """Count dataset rows after optional category, intent, and text filters."""

    return count_rows_impl(category=category, intent=intent, search_text=search_text)


@tool("show_examples", args_schema=ExamplesInput)
def show_examples(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
    limit: int = 3,
    offset: int = 0,
) -> dict:
    """Return matching examples for listing, including each customer query and support response.

    Prefer `search_text` for natural-language concepts. Only use `intent` when you know the
    exact dataset intent label already exists. Text matching is applied to customer
    instructions so generic support-script wording does not create false positives.
    """

    return show_examples_impl(
        category=category,
        intent=intent,
        search_text=search_text,
        limit=limit,
        offset=offset,
    )


@tool("distribution", args_schema=DistributionInput)
def distribution(
    group_by: Annotated[str, "category, intent, or flags"],
    category: str | None = None,
    intent: str | None = None,
    top_n: int = 30,
) -> dict:
    """Calculate a count distribution for category, intent, or flags in the dataset."""

    return distribution_impl(group_by=group_by, category=category, intent=intent, top_n=top_n)


@tool("sample_responses_for_summary", args_schema=ExamplesInput)
def sample_responses_for_summary(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """Collect response samples for open-ended summaries about how agents reply."""

    return sample_responses_for_summary_impl(
        category=category,
        intent=intent,
        search_text=search_text,
        limit=limit,
        offset=offset,
    )


DATASET_TOOLS = [
    list_categories,
    list_intents,
    count_rows,
    show_examples,
    distribution,
    sample_responses_for_summary,
]


def get_dataset_tools(use_mcp: bool = False, mcp_server_url: str | None = None) -> list:
    """Return either local tools or MCP-backed tool proxies."""

    if not use_mcp:
        return DATASET_TOOLS

    from customer_support_agent.mcp_tools import get_mcp_dataset_tools

    return get_mcp_dataset_tools(mcp_server_url)
