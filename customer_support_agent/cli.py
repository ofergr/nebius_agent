"""Interactive command-line interface."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from customer_support_agent.agent import invoke_agent, reset_checkpoint_db
from customer_support_agent.profile import normalize_user_id


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


def _message_signature(message: BaseMessage) -> str:
    if isinstance(message, ToolMessage):
        return f"tool|{message.name}|{message.tool_call_id}|{message.content}"
    if isinstance(message, AIMessage):
        return f"ai|{message.content}|{message.tool_calls}"
    return f"{type(message).__name__}|{message.content}"


def _new_messages_for_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return only the messages produced in the current user turn.

    The checkpointed graph returns the full session history, so the CLI trims the
    output to the last human turn and everything after it.
    """

    last_human_index = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type == "human":
            last_human_index = index
            break

    if last_human_index is None:
        return messages
    return messages[last_human_index + 1 :]


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
        "--user",
        default=None,
        help="Persistent user ID for storing a separate long-term profile across sessions.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete the persistent LangGraph checkpoint database and exit.",
    )
    args = parser.parse_args()

    if args.reset_db:
        checkpoint_path, profile_dir = reset_checkpoint_db()
        print(f"Reset checkpoint database: {checkpoint_path}")
        print(f"Reset user profiles: {profile_dir}")
        return

    if args.once:
        messages = invoke_agent(args.once, session_id=args.session, user_id=args.user)
        _print_reasoning(_new_messages_for_turn(messages))
        print(f"\nAssistant: {_final_answer(messages)}")
        return

    print("Customer Support Data Analyst Agent")
    print(f"Session: {args.session}")
    print(f"User: {normalize_user_id(args.user or args.session)}")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.casefold() in {"exit", "quit"}:
            break
        if not question:
            continue
        messages = invoke_agent(question, session_id=args.session, user_id=args.user)
        _print_reasoning(_new_messages_for_turn(messages))
        print(f"\nAssistant: {_final_answer(messages)}")
