"""LangGraph ReAct agent construction."""

from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from customer_support_agent.config import Settings, get_settings
from customer_support_agent.router import RouteDecision, route_query
from customer_support_agent.tools import DATASET_TOOLS


class AgentState(TypedDict):
    """Graph state for the Task 1 agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    route_reason: str


SYSTEM_PROMPT = """You are a data analyst agent for the Bitext customer support dataset.

You must answer only questions about this dataset. Use tools for data facts, counts,
examples, distributions, and evidence gathering. Do not invent dataset statistics.

For structured questions, call the most specific analysis tool or chain tools when needed.
For open-ended questions, gather representative rows first, then summarize patterns from
the gathered evidence. Keep answers concise and mention relevant filters/counts when known.

When the user asks to show, list, or give examples, the final answer must list the
returned examples individually. Include each example's customer instruction and support
response. Do not replace example listings with a high-level summary.

When the user asks to compare categories, intents, or groups, the final answer must
explicitly describe each side of the comparison and then state the overlap or key
difference. Prefer concrete counts and names over vague wording.

After a tool returns a valid observation, produce the final answer from that observation
unless another different tool call is clearly required. Do not repeat the same tool call
with the same arguments after it has returned data.
"""


def _tool_call_signature(message: BaseMessage | None) -> tuple[str, str] | None:
    if not isinstance(message, AIMessage) or not message.tool_calls or len(message.tool_calls) != 1:
        return None

    call = message.tool_calls[0]
    return (str(call["name"]), json.dumps(call["args"], sort_keys=True))


def _has_empty_observation(message: BaseMessage | None) -> bool:
    if not isinstance(message, ToolMessage):
        return False

    try:
        payload = json.loads(str(message.content))
    except json.JSONDecodeError:
        return False

    if payload.get("total_matches") == 0:
        return True
    if payload.get("sample_size") == 0:
        return True
    if payload.get("row_count") == 0:
        return True
    if payload.get("returned") == 0 and payload.get("examples") == []:
        return True
    return False


def _empty_observation_response(messages: list[BaseMessage]) -> str | None:
    if len(messages) < 2:
        return None

    latest_tool = messages[-1]
    latest_call = messages[-2]
    if not _has_empty_observation(latest_tool):
        return None

    signature = _tool_call_signature(latest_call)
    if signature is None:
        return None

    tool_name, _ = signature
    if tool_name == "count_rows":
        return "I couldn't find any matching rows in the dataset for that request."
    if tool_name == "show_examples":
        return "I couldn't find matching examples in the dataset for that request."
    if tool_name == "sample_responses_for_summary":
        return "I couldn't find matching rows in the dataset for that request, so I don't have examples to summarize."
    return None


def _should_stop_after_repeated_empty_tool_calls(messages: list[BaseMessage]) -> bool:
    if len(messages) < 4:
        return False

    latest_tool = messages[-1]
    latest_call = messages[-2]
    previous_tool = messages[-3]
    previous_call = messages[-4]

    if not _has_empty_observation(latest_tool) or not _has_empty_observation(previous_tool):
        return False

    latest_signature = _tool_call_signature(latest_call)
    previous_signature = _tool_call_signature(previous_call)
    return latest_signature is not None and latest_signature == previous_signature


def build_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Build a Nebius Token Factory chat model using an OpenAI-compatible endpoint."""

    settings = settings or get_settings()
    if not settings.nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Copy .env.example to .env and add your Nebius Token Factory key."
        )
    return ChatOpenAI(
        model=settings.nebius_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
    )


def _router_node(state: AgentState) -> dict[str, str | list[AIMessage]]:
    last_user_message = next(
        (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
        None,
    )
    query = last_user_message.content if last_user_message else ""
    decision = route_query(str(query))
    if decision.route == "out_of_scope":
        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "messages": [
                AIMessage(
                    content=(
                        "I can only answer questions about the Bitext customer support dataset "
                        "for this assignment, so I can't help with that request."
                    )
                )
            ],
        }
    return {"route": decision.route, "route_reason": decision.reason}


def _next_after_router(state: AgentState) -> str:
    return END if state["route"] == "out_of_scope" else "agent"


def build_graph(settings: Settings | None = None):
    """Compile the LangGraph ReAct graph used by the CLI."""

    llm = build_llm(settings).bind_tools(DATASET_TOOLS)
    tool_node = ToolNode(DATASET_TOOLS)

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        empty_response = _empty_observation_response(state["messages"])
        if empty_response is not None:
            return {"messages": [AIMessage(content=empty_response)]}

        if _should_stop_after_repeated_empty_tool_calls(state["messages"]):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I couldn't find matching rows in the dataset for that request, "
                            "so I don't have examples to summarize."
                        )
                    )
                ]
            }

        route_context = (
            f"Router classified this request as {state['route']}. "
            f"Router reason: {state['route_reason']}."
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=route_context)]
        messages.extend(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("router", _router_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", _next_after_router, {"agent": "agent", END: END})
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def invoke_agent(question: str, settings: Settings | None = None) -> list[BaseMessage]:
    """Run the graph once and return all emitted messages."""

    settings = settings or get_settings()
    graph = build_graph(settings)
    # LangGraph's recursion_limit counts graph node executions, not full ReAct
    # iterations. Each tool cycle usually costs agent -> tools -> agent steps.
    graph_step_limit = settings.max_iterations * 2 + 4
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)], "route": "", "route_reason": ""},
            config={"recursion_limit": graph_step_limit},
        )
    except GraphRecursionError:
        return [
            AIMessage(
                content=(
                    "I reached the maximum reasoning iteration limit before producing a final answer. "
                    "Please narrow the question or ask for a specific count, distribution, or example set."
                )
            )
        ]
    return result["messages"]
