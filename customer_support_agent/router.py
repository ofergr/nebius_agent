"""Query router for dataset-scoped requests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


Route = Literal["structured", "unstructured", "out_of_scope"]


class RouteDecision(BaseModel):
    """Router output stored in graph state."""

    route: Route
    reason: str


DATASET_TERMS = {
    "account",
    "accounts",
    "agent",
    "agents",
    "bitext",
    "billing",
    "broken",
    "cancellation",
    "cancel",
    "category",
    "categories",
    "complaint",
    "complaints",
    "compensation",
    "customer",
    "customers",
    "damaged",
    "dataset",
    "delay",
    "delays",
    "delayed",
    "delivery",
    "fee",
    "examples",
    "feedback",
    "help",
    "invoice",
    "intent",
    "intents",
    "issue",
    "late",
    "login",
    "lost package",
    "missing item",
    "order",
    "orders",
    "package",
    "packages",
    "payment",
    "policy",
    "problem",
    "refund policy",
    "reimbursement",
    "return",
    "returns",
    "rows",
    "refund",
    "refunds",
    "response",
    "responses",
    "shipping",
    "shipment",
    "sign in",
    "support",
    "support request",
    "track",
    "tracking",
    "wrong item",
}

STRUCTURED_TERMS = {
    "common",
    "compare",
    "count",
    "counts",
    "difference",
    "distribution",
    "exist",
    "examples",
    "filter",
    "how many",
    "least common",
    "list",
    "mention",
    "mentions",
    "most common",
    "show",
    "top",
    "total",
    "what categories",
    "what intents",
}

UNSTRUCTURED_TERMS = {
    "how do",
    "summarize",
    "summary",
    "typically",
    "usual",
    "usually",
    "pattern",
    "describe",
}

OUT_OF_SCOPE_HINTS = {
    "best crm",
    "champions league",
    "president of",
    "poem",
    "weather",
    "stock price",
}


def route_query(query: str) -> RouteDecision:
    """Classify a user query before tool selection."""

    lowered = query.casefold()
    has_dataset_term = any(term in lowered for term in DATASET_TERMS)
    has_structured_term = any(term in lowered for term in STRUCTURED_TERMS)
    has_unstructured_term = any(term in lowered for term in UNSTRUCTURED_TERMS)

    if any(hint in lowered for hint in OUT_OF_SCOPE_HINTS):
        return RouteDecision(route="out_of_scope", reason="The question asks for non-dataset knowledge.")

    if has_unstructured_term and has_dataset_term:
        return RouteDecision(route="unstructured", reason="The question asks for a qualitative dataset summary.")

    if has_structured_term and has_dataset_term:
        return RouteDecision(route="structured", reason="The question asks for concrete dataset facts.")

    if has_structured_term:
        return RouteDecision(
            route="structured",
            reason="The question uses a dataset-analysis pattern even if it does not name a category directly.",
        )

    if has_dataset_term:
        return RouteDecision(route="structured", reason="The question appears to be about the dataset.")

    return RouteDecision(route="out_of_scope", reason="The question does not appear related to the dataset.")
