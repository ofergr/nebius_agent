"""Shared formatting helpers for CLI and Streamlit chat views."""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


class ReasoningStep(TypedDict):
    """A single visible reasoning step for the UI."""

    kind: Literal["tool_call", "observation"]
    label: str
    content: str


def final_answer(messages: list[BaseMessage]) -> str:
    """Return the last non-tool assistant answer from a turn."""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    return "I could not produce a final answer."


def new_messages_for_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return only the messages produced in the current user turn."""

    last_human_index = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            last_human_index = index
            break

    if last_human_index is None:
        return messages
    return messages[last_human_index + 1 :]


def reasoning_steps(
    messages: list[BaseMessage],
    tool_mode: Literal["local", "mcp"] = "local",
) -> list[ReasoningStep]:
    """Convert messages into display-friendly tool-call and observation steps."""

    tool_call_label = "mcp tool call" if tool_mode == "mcp" else "tool call"
    observation_label = "mcp observation" if tool_mode == "mcp" else "observation"
    steps: list[ReasoningStep] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                steps.append(
                    {
                        "kind": "tool_call",
                        "label": tool_call_label,
                        "content": f"{call['name']}({call['args']})",
                    }
                )
        elif isinstance(message, ToolMessage):
            steps.append(
                {
                    "kind": "observation",
                    "label": observation_label,
                    "content": f"{message.name}: {str(message.content)[:1200]}",
                }
            )
    return steps
