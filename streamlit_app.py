"""Streamlit chat UI for the customer support dataset agent."""

from __future__ import annotations

from html import escape
from typing import Literal

import streamlit as st

from customer_support_agent.agent import invoke_agent
from customer_support_agent.chat_view import final_answer, new_messages_for_turn, reasoning_steps
from customer_support_agent.config import get_mcp_server_url
from customer_support_agent.mcp_tools import verify_mcp_server
from customer_support_agent.profile import normalize_user_id


REASONING_BLOCK_CSS = """
<style>
.reasoning-block {
    background: rgba(250, 250, 250, 0.06);
    border-radius: 0.75rem;
    padding: 1rem;
    margin: 0.5rem 0 1rem 0;
    overflow-x: auto;
}
.reasoning-block pre {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
    margin: 0;
    font-family: "SFMono-Regular", SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, monospace;
    font-size: 0.95rem;
    line-height: 1.5;
}
</style>
"""


def _conversation_key(
    session_id: str,
    user_id: str | None,
    tool_mode: Literal["local", "mcp"],
) -> str:
    normalized_user = normalize_user_id(user_id or session_id)
    return f"{tool_mode}:{session_id}:{normalized_user}"


def _render_reasoning_step(step: dict[str, str]) -> None:
    st.markdown(f"**[{step['label']}]**")
    st.markdown(
        f"<div class='reasoning-block'><pre>{escape(step['content'])}</pre></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the Streamlit chat UI."""

    st.set_page_config(page_title="Customer Support Dataset Agent", page_icon="💬", layout="wide")
    st.markdown(REASONING_BLOCK_CSS, unsafe_allow_html=True)
    st.title("Customer Support Dataset Agent")
    st.caption("Streamlit chat UI with visible tool reasoning and persistent session IDs.")

    with st.sidebar:
        st.header("Session")
        session_id = st.text_input(
            "Session ID",
            value=st.session_state.get("streamlit_session_id", "default"),
            help="Reuse the same session ID to continue the same LangGraph conversation across turns and restarts.",
        ).strip() or "default"
        st.session_state["streamlit_session_id"] = session_id

        user_id = st.text_input(
            "User ID",
            value=st.session_state.get("streamlit_user_id", ""),
            help="Optional long-term semantic profile key. Leave blank to default to the session ID.",
        ).strip() or None
        st.session_state["streamlit_user_id"] = user_id or ""

        use_mcp = st.checkbox(
            "Use MCP server",
            value=st.session_state.get("streamlit_use_mcp", False),
            help="When enabled, the agent calls tools through the standalone MCP server instead of local in-process tools.",
        )
        st.session_state["streamlit_use_mcp"] = use_mcp

        tool_mode: Literal["local", "mcp"] = "mcp" if use_mcp else "local"
        mcp_server_url = get_mcp_server_url() if use_mcp else None
        if use_mcp:
            st.caption(f"MCP endpoint: `{mcp_server_url}`")
        else:
            st.caption("Tool mode: local")

    transcripts = st.session_state.setdefault("chat_transcripts", {})
    conversation_key = _conversation_key(session_id, user_id, tool_mode)
    turns = transcripts.setdefault(conversation_key, [])

    st.info(
        f"Session: `{session_id}` | User: `{normalize_user_id(user_id or session_id)}` | "
        f"Tool mode: `{tool_mode}`"
    )

    for index, turn in enumerate(turns, start=1):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            with st.expander(f"Reasoning steps for turn {index}", expanded=False):
                for step in turn["steps"]:
                    _render_reasoning_step(step)

    question = st.chat_input("Ask about the Bitext customer support dataset...")
    if not question:
        return

    if use_mcp:
        try:
            verify_mcp_server(mcp_server_url)
        except RuntimeError as exc:
            with st.chat_message("assistant"):
                st.error(str(exc))
            return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                messages = invoke_agent(
                    question,
                    session_id=session_id,
                    user_id=user_id,
                    use_mcp=use_mcp,
                    mcp_server_url=mcp_server_url,
                )
            except Exception as exc:  # pragma: no cover - UI safety path
                st.error(str(exc))
                return

        turn_messages = new_messages_for_turn(messages)
        steps = reasoning_steps(turn_messages, tool_mode=tool_mode)
        answer = final_answer(messages)
        st.markdown(answer)
        with st.expander("Reasoning steps", expanded=True):
            if not steps:
                st.caption("No visible tool steps for this turn.")
            for step in steps:
                _render_reasoning_step(step)

    turns.append({"question": question, "answer": answer, "steps": steps})


if __name__ == "__main__":
    main()
