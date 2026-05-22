"""Interactive command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from customer_support_agent.agent import invoke_agent, reset_checkpoint_db


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
    parser.add_argument(
        "--session",
        default="default",
        help="Persistent session ID for restoring the same conversation across turns and restarts.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete the persistent LangGraph checkpoint database and exit.",
    )
    args = parser.parse_args()

    if args.reset_db:
        path = reset_checkpoint_db()
        print(f"Reset checkpoint database: {path}")
        return

    if args.once:
        messages = invoke_agent(args.once, session_id=args.session)
        _print_reasoning(messages)
        print(f"\nAssistant: {_final_answer(messages)}")
        return

    print("Customer Support Data Analyst Agent")
    print(f"Session: {args.session}")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.casefold() in {"exit", "quit"}:
            break
        if not question:
            continue
        messages = invoke_agent(question, session_id=args.session)
        _print_reasoning(messages)
        print(f"\nAssistant: {_final_answer(messages)}")
