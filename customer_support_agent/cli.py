"""Interactive command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from customer_support_agent.agent import invoke_agent


def _print_reasoning(messages: Iterable[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"\n[tool call] {call['name']}({call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"[observation] {message.name}: {message.content[:1200]}")


def _final_answer(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    return "I could not produce a final answer."


def main() -> None:
    """Run an interactive agent session."""

    parser = argparse.ArgumentParser(description="Customer support dataset analyst agent")
    parser.add_argument("--once", help="Ask one question and exit.")
    args = parser.parse_args()

    if args.once:
        messages = invoke_agent(args.once)
        _print_reasoning(messages)
        print(f"\nAssistant: {_final_answer(messages)}")
        return

    print("Customer Support Data Analyst Agent")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.casefold() in {"exit", "quit"}:
            break
        if not question:
            continue
        messages = invoke_agent(question)
        _print_reasoning(messages)
        print(f"\nAssistant: {_final_answer(messages)}")
