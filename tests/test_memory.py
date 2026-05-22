"""Tests for persistent conversation memory."""

from __future__ import annotations

import json

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from customer_support_agent import agent, dataset
from customer_support_agent.config import Settings


class FakeBoundLLM:
    """Deterministic stand-in for tool-calling conversation tests."""

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        last_human = next(
            (message.content for message in reversed(messages) if isinstance(message, HumanMessage)),
            "",
        )
        latest_tool = next(
            (message for message in reversed(messages) if isinstance(message, ToolMessage)),
            None,
        )

        if latest_tool is not None and messages[-1] is latest_tool:
            payload = json.loads(str(latest_tool.content))
            instructions = [example["instruction"] for example in payload["examples"]]
            return AIMessage(content=" | ".join(instructions))

        if last_human == "Show me 3 examples from the REFUND category":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "show_examples",
                        "args": {"category": "REFUND", "limit": 3, "offset": 0},
                        "id": "call-first",
                        "type": "tool_call",
                    }
                ],
            )

        if last_human == "Show me 3 more":
            previous_questions = [
                str(message.content)
                for message in messages
                if isinstance(message, HumanMessage) and message.content != last_human
            ]
            assert "Show me 3 examples from the REFUND category" in previous_questions
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "show_examples",
                        "args": {"category": "REFUND", "limit": 3, "offset": 3},
                        "id": "call-second",
                        "type": "tool_call",
                    }
                ],
            )
        raise AssertionError(f"Unexpected message flow for prompt: {last_human}")


def _settings(tmp_path) -> Settings:
    return Settings(
        nebius_api_key="test-key",
        nebius_base_url="https://api.tokenfactory.nebius.com/v1",
        nebius_model="meta-llama/Llama-3.3-70B-Instruct",
        dataset_name="unused",
        dataset_split="train",
        max_iterations=12,
        checkpoint_db_path=str(tmp_path / "memory.sqlite"),
    )


def test_session_history_persists_across_graph_rebuilds(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "flags": "",
                "instruction": "Refund order A",
                "category": "REFUND",
                "intent": "get_refund",
                "response": "Refund response A",
            },
            {
                "flags": "",
                "instruction": "Refund order B",
                "category": "REFUND",
                "intent": "get_refund",
                "response": "Refund response B",
            },
            {
                "flags": "",
                "instruction": "Refund order C",
                "category": "REFUND",
                "intent": "get_refund",
                "response": "Refund response C",
            },
            {
                "flags": "",
                "instruction": "Refund order D",
                "category": "REFUND",
                "intent": "get_refund",
                "response": "Refund response D",
            },
            {
                "flags": "",
                "instruction": "Refund order E",
                "category": "REFUND",
                "intent": "get_refund",
                "response": "Refund response E",
            },
            {
                "flags": "",
                "instruction": "Refund order F",
                "category": "REFUND",
                "intent": "get_refund",
                "response": "Refund response F",
            },
        ]
    )
    settings = _settings(tmp_path)

    monkeypatch.setattr(dataset, "load_customer_support_data", lambda: frame)
    monkeypatch.setattr(agent, "build_llm", lambda _settings=None: FakeBoundLLM())
    agent.reset_runtime_caches()

    first_messages = agent.invoke_agent(
        "Show me 3 examples from the REFUND category",
        settings=settings,
        session_id="memory-demo",
    )
    assert str(first_messages[-1].content) == "Refund order A | Refund order B | Refund order C"

    agent.reset_runtime_caches()
    monkeypatch.setattr(agent, "build_llm", lambda _settings=None: FakeBoundLLM())

    second_messages = agent.invoke_agent(
        "Show me 3 more",
        settings=settings,
        session_id="memory-demo",
    )

    assert str(second_messages[-1].content) == "Refund order D | Refund order E | Refund order F"
    assert any(
        isinstance(message, HumanMessage) and message.content == "Show me 3 examples from the REFUND category"
        for message in second_messages
    )
