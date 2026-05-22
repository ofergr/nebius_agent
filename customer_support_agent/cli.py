"""Interactive command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from customer_support_agent.agent import invoke_agent, reset_checkpoint_db
from customer_support_agent.config import get_mcp_server_url
from customer_support_agent.mcp_tools import verify_mcp_server
from customer_support_agent.profile import normalize_user_id


def _print_reasoning(
    messages: Iterable[BaseMessage],
    tool_mode: Literal["local", "mcp"] = "local",
) -> None:
    tool_call_label = "mcp tool call" if tool_mode == "mcp" else "tool call"
    observation_label = "mcp observation" if tool_mode == "mcp" else "observation"
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"\n[{tool_call_label}] {call['name']}({call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"[{observation_label}] {message.name}: {message.content[:1200]}")


def _final_answer(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    return "I could not produce a final answer."


def _new_messages_for_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return only the messages produced in the current user turn.

    The checkpointed graph returns the full session history, so the CLI trims the
    output to the last human turn and everything after it.
    """

    last_human_index = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
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
    parser.add_argument(
        "--use-mcp",
        action="store_true",
        help="Connect the agent to the standalone MCP server instead of using local tools.",
    )
    args = parser.parse_args()

    if args.reset_db:
        checkpoint_path, profile_dir = reset_checkpoint_db()
        print(f"Reset checkpoint database: {checkpoint_path}")
        print(f"Reset user profiles: {profile_dir}")
        return

    tool_mode: Literal["local", "mcp"] = "mcp" if args.use_mcp else "local"
    mcp_server_url = get_mcp_server_url() if args.use_mcp else None
    if args.use_mcp:
        verify_mcp_server(mcp_server_url)

    if args.once:
        messages = invoke_agent(
            args.once,
            session_id=args.session,
            user_id=args.user,
            use_mcp=args.use_mcp,
            mcp_server_url=mcp_server_url,
        )
        _print_reasoning(_new_messages_for_turn(messages), tool_mode=tool_mode)
        print(f"\nAssistant: {_final_answer(messages)}")
        return

    print("Customer Support Data Analyst Agent")
    print(f"Session: {args.session}")
    print(f"User: {normalize_user_id(args.user or args.session)}")
    if args.use_mcp:
        print(f"Tool Mode: MCP ({mcp_server_url})")
    else:
        print("Tool Mode: Local")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.casefold() in {"exit", "quit"}:
            break
        if not question:
            continue
        messages = invoke_agent(
            question,
            session_id=args.session,
            user_id=args.user,
            use_mcp=args.use_mcp,
            mcp_server_url=mcp_server_url,
        )
        _print_reasoning(_new_messages_for_turn(messages), tool_mode=tool_mode)
        print(f"\nAssistant: {_final_answer(messages)}")
