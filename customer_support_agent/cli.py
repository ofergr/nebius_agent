"""Interactive command-line interface."""

from __future__ import annotations

import argparse
from typing import Literal

from customer_support_agent.agent import invoke_agent, reset_checkpoint_db
from customer_support_agent.chat_view import final_answer, new_messages_for_turn, reasoning_steps
from customer_support_agent.config import get_mcp_server_url
from customer_support_agent.mcp_tools import verify_mcp_server
from customer_support_agent.profile import normalize_user_id


def _print_reasoning(messages, tool_mode: Literal["local", "mcp"] = "local") -> None:
    for step in reasoning_steps(messages, tool_mode=tool_mode):
        prefix = "\n" if step["kind"] == "tool_call" else ""
        print(f"{prefix}[{step['label']}] {step['content']}")


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
        _print_reasoning(new_messages_for_turn(messages), tool_mode=tool_mode)
        print(f"\nAssistant: {final_answer(messages)}")
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
        _print_reasoning(new_messages_for_turn(messages), tool_mode=tool_mode)
        print(f"\nAssistant: {final_answer(messages)}")
