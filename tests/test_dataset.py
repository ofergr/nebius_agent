"""Tests for dataset filtering helpers."""

from __future__ import annotations

import pandas as pd

from customer_support_agent import dataset


def test_filter_rows_ignores_stringified_null_filters(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "flags": "",
                "instruction": "Where is my package?",
                "category": "SHIPPING",
                "intent": "track_order",
                "response": "Here is how to track your shipment.",
            }
        ]
    )
    monkeypatch.setattr(dataset, "load_customer_support_data", lambda: frame)

    result = dataset.filter_rows(category="SHIPPING", intent="null", search_text="null")

    assert len(result) == 1


def test_filter_rows_defaults_to_instruction_search(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "flags": "",
                "instruction": "I need to cancel my order",
                "category": "ORDER",
                "intent": "cancel_order",
                "response": "Please login to your account to continue.",
            }
        ]
    )
    monkeypatch.setattr(dataset, "load_customer_support_data", lambda: frame)

    result = dataset.filter_rows(search_text="login")

    assert len(result) == 0


def test_filter_rows_can_search_both_instruction_and_response(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "flags": "",
                "instruction": "I need to cancel my order",
                "category": "ORDER",
                "intent": "cancel_order",
                "response": "Please login to your account to continue.",
            }
        ]
    )
    monkeypatch.setattr(dataset, "load_customer_support_data", lambda: frame)

    result = dataset.filter_rows(search_text="login", search_columns=("instruction", "response"))

    assert len(result) == 1


def test_filter_rows_expands_login_related_search_terms(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "flags": "",
                "instruction": "I forgot my password and cannot access my account",
                "category": "ACCOUNT",
                "intent": "recover_password",
                "response": "Use the password recovery flow.",
            }
        ]
    )
    monkeypatch.setattr(dataset, "load_customer_support_data", lambda: frame)

    result = dataset.filter_rows(search_text="login-related issues")

    assert len(result) == 1


def test_filter_rows_does_not_expand_missing_item_to_track_order(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "flags": "",
                "instruction": "track order {{Order Number}}",
                "category": "ORDER",
                "intent": "track_order",
                "response": "Here is how to track your order.",
            }
        ]
    )
    monkeypatch.setattr(dataset, "load_customer_support_data", lambda: frame)

    result = dataset.filter_rows(search_text="missing item")

    assert len(result) == 0
