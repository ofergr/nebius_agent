"""Tests for shared chat presentation helpers."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from customer_support_agent.chat_view import final_answer, new_messages_for_turn, reasoning_steps


def test_new_messages_for_turn_trims_prior_history() -> None:
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="first answer"),
        HumanMessage(content="second"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "count_rows",
                    "args": {"category": "REFUND"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content='{"row_count": 3}', tool_call_id="call-1", name="count_rows"),
        AIMessage(content="second answer"),
    ]

    trimmed = new_messages_for_turn(messages)

    assert len(trimmed) == 3
    assert isinstance(trimmed[0], AIMessage)
    assert isinstance(trimmed[1], ToolMessage)
    assert str(trimmed[2].content) == "second answer"


def test_reasoning_steps_labels_local_and_mcp() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "show_examples",
                    "args": {"category": "CONTACT", "limit": 1},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content='{"returned": 1}', tool_call_id="call-1", name="show_examples"),
    ]

    local_steps = reasoning_steps(messages, tool_mode="local")
    mcp_steps = reasoning_steps(messages, tool_mode="mcp")

    assert local_steps[0]["label"] == "tool call"
    assert local_steps[1]["label"] == "observation"
    assert mcp_steps[0]["label"] == "mcp tool call"
    assert mcp_steps[1]["label"] == "mcp observation"


def test_final_answer_returns_last_non_tool_ai_message() -> None:
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[{"name": "count_rows", "args": {}, "id": "call-1", "type": "tool_call"}]),
        ToolMessage(content='{"row_count": 2}', tool_call_id="call-1", name="count_rows"),
        AIMessage(content="There are 2 rows."),
    ]

    assert final_answer(messages) == "There are 2 rows."
