"""Tests for conversation-history answers."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from customer_support_agent.agent import AgentState, _next_after_router, build_graph, reset_runtime_caches
from customer_support_agent.config import Settings


def _settings(tmp_path) -> Settings:
    return Settings(
        nebius_api_key="test-key",
        nebius_base_url="https://api.tokenfactory.nebius.com/v1",
        nebius_model="meta-llama/Llama-3.3-70B-Instruct",
        dataset_name="unused",
        dataset_split="train",
        max_iterations=12,
        checkpoint_db_path=str(tmp_path / "memory.sqlite"),
        user_profile_dir=str(tmp_path / "profiles"),
    )


def test_next_after_router_uses_session_memory_route() -> None:
    state: AgentState = {
        "messages": [],
        "route": "session_memory",
        "route_reason": "history",
        "profile_context": "",
    }

    assert _next_after_router(state) == "session_memory"


def test_session_history_question_lists_previous_questions(monkeypatch, tmp_path) -> None:
    from customer_support_agent import agent

    class StubLLM:
        def bind_tools(self, _tools):
            return self

        def invoke(self, messages):
            last_human = next(
                (message.content for message in reversed(messages) if isinstance(message, HumanMessage)),
                "",
            )
            if last_human == "What questions did I ask so far?":
                raise AssertionError("LLM should not be called for session history questions.")
            return AIMessage(content=f"Handled: {last_human}")

    settings = _settings(tmp_path)
    monkeypatch.setattr(agent, "build_llm", lambda _settings=None: StubLLM())
    reset_runtime_caches()

    first = agent.invoke_agent("How many refunds did we get?", settings=settings, session_id="demo")
    assert isinstance(first[-1], AIMessage)

    second = agent.invoke_agent("What about shipping?", settings=settings, session_id="demo")
    assert isinstance(second[-1], AIMessage)

    third = agent.invoke_agent("What questions did I ask so far?", settings=settings, session_id="demo")

    final_text = str(third[-1].content)
    assert "1. How many refunds did we get?" in final_text
    assert "2. What about shipping?" in final_text
    assert "What questions did I ask so far?" not in final_text
