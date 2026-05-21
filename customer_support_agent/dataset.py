"""Dataset loading and analysis helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from datasets import load_dataset

from customer_support_agent.config import get_settings


TEXT_COLUMNS = ("instruction", "response")

SEARCH_TERM_EXPANSIONS = {
    "billing": ("billing", "payment", "charged", "invoice", "fee"),
    "login": (
        "login",
        "log in",
        "sign in",
        "password",
        "recover password",
        "registration",
        "account locked",
        "access account",
    ),
    "money back": ("money back", "refund", "reimbursement", "compensation", "return"),
}


@lru_cache(maxsize=1)
def load_customer_support_data() -> pd.DataFrame:
    """Load the Bitext customer support dataset as a normalized DataFrame."""

    settings = get_settings()
    dataset = load_dataset(settings.dataset_name, split=settings.dataset_split)
    frame = dataset.to_pandas()
    expected_columns = {"flags", "instruction", "category", "intent", "response"}
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    frame = frame.copy()
    frame["category"] = frame["category"].astype(str).str.upper()
    frame["intent"] = frame["intent"].astype(str)
    frame["instruction"] = frame["instruction"].astype(str)
    frame["response"] = frame["response"].astype(str)
    return frame


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.casefold() in {"none", "null", "nil", "n/a"}:
        return None
    return stripped or None


def _expand_search_terms(search_text: str | None) -> tuple[str, ...]:
    if not search_text:
        return ()

    lowered = search_text.casefold()
    expanded_terms: list[str] = [search_text]
    for trigger, terms in SEARCH_TERM_EXPANSIONS.items():
        if trigger in lowered:
            expanded_terms.extend(terms)

    # Preserve order while dropping duplicates.
    return tuple(dict.fromkeys(term for term in expanded_terms if term))


def filter_rows(
    category: str | None = None,
    intent: str | None = None,
    search_text: str | None = None,
    search_columns: tuple[str, ...] = ("instruction",),
) -> pd.DataFrame:
    """Return rows matching optional category, intent, and text search filters."""

    frame = load_customer_support_data()
    category = _normalize(category)
    intent = _normalize(intent)
    search_text = _normalize(search_text)

    if category:
        frame = frame[frame["category"].str.casefold() == category.casefold()]
    if intent:
        frame = frame[frame["intent"].str.casefold() == intent.casefold()]
    if search_text:
        needles = tuple(term.casefold() for term in _expand_search_terms(search_text))
        mask = False
        for column in search_columns:
            column_text = frame[column].str.casefold()
            column_mask = False
            for needle in needles:
                column_mask = column_mask | column_text.str.contains(needle, regex=False)
            mask = mask | column_mask
        frame = frame[mask]
    return frame


def row_to_public_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a dataset row to fields useful for command-line display."""

    return {
        "instruction": row["instruction"],
        "category": row["category"],
        "intent": row["intent"],
        "response": row["response"],
    }
