"""Tests for agent loop behavior."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from customer_support_agent.agent import (
    _empty_observation_response,
    _should_stop_after_repeated_empty_tool_calls,
)


def test_stops_after_repeated_identical_empty_tool_calls() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sample_responses_for_summary",
                    "args": {
                        "category": "SHIPPING",
                        "intent": "track_package",
                        "search_text": "lost package",
                        "limit": 10,
                        "offset": 0,
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"total_matches": 0, "sample_size": 0, "examples": []}',
            tool_call_id="call-1",
            name="sample_responses_for_summary",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sample_responses_for_summary",
                    "args": {
                        "category": "SHIPPING",
                        "intent": "track_package",
                        "search_text": "lost package",
                        "limit": 10,
                        "offset": 0,
                    },
                    "id": "call-2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"total_matches": 0, "sample_size": 0, "examples": []}',
            tool_call_id="call-2",
            name="sample_responses_for_summary",
        ),
    ]

    assert _should_stop_after_repeated_empty_tool_calls(messages) is True


def test_does_not_stop_when_second_tool_call_differs() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sample_responses_for_summary",
                    "args": {"search_text": "lost package", "limit": 10, "offset": 0},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"total_matches": 0, "sample_size": 0, "examples": []}',
            tool_call_id="call-1",
            name="sample_responses_for_summary",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sample_responses_for_summary",
                    "args": {"search_text": "track order", "limit": 10, "offset": 0},
                    "id": "call-2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"total_matches": 0, "sample_size": 0, "examples": []}',
            tool_call_id="call-2",
            name="sample_responses_for_summary",
        ),
    ]

    assert _should_stop_after_repeated_empty_tool_calls(messages) is False


def test_empty_count_rows_observation_gets_direct_response() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "count_rows",
                    "args": {"search_text": "return", "category": None, "intent": None},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"row_count": 0, "filters": {"category": null, "intent": null, "search_text": "return"}}',
            tool_call_id="call-1",
            name="count_rows",
        ),
    ]

    assert _empty_observation_response(messages) == (
        "I couldn't find any matching rows in the dataset for that request."
    )


def test_empty_summary_observation_gets_direct_response() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sample_responses_for_summary",
                    "args": {"search_text": "lost package", "limit": 10, "offset": 0},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"total_matches": 0, "sample_size": 0, "examples": []}',
            tool_call_id="call-1",
            name="sample_responses_for_summary",
        ),
    ]

    assert _empty_observation_response(messages) == (
        "I couldn't find matching rows in the dataset for that request, so I don't have examples to summarize."
    )
