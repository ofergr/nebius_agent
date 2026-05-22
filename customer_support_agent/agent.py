"""LangGraph ReAct agent construction."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from customer_support_agent.config import Settings, get_settings
from customer_support_agent.recommender import build_suggestion, extract_category_hint, refine_suggestion
from customer_support_agent.profile import (
    _extract_topics,
    load_user_profile,
    message_has_profile_update,
    normalize_user_id,
    render_profile_answer,
    render_profile_update_acknowledgement,
    render_profile_summary,
    reset_user_profiles,
    update_user_profile,
)
from customer_support_agent.router import route_query
from customer_support_agent.tools import get_dataset_tools


class AgentState(TypedDict):
    """Graph state for the Task 1 agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    route_reason: str
    profile_context: str
    # Bonus B: holds a proposed tool call while waiting for user confirmation.
    # None means no suggestion is pending.  Keys: tool, args, description.
    pending_suggestion: dict | None


DEFAULT_SESSION_ID = "default"


SYSTEM_PROMPT = """You are a data analyst agent for the Bitext customer support dataset.

You must answer only questions about this dataset. Use tools for data facts, counts,
examples, distributions, and evidence gathering. Do not invent dataset statistics.

For structured questions, call the most specific analysis tool or chain tools when needed.
For open-ended questions, gather representative rows first, then summarize patterns from
the gathered evidence. Keep answers concise and mention relevant filters/counts when known.

When the user asks to summarize, describe patterns, or explain how the dataset typically
looks for a category or intent, you must use `sample_responses_for_summary` first and
then write a synthesized summary. Do not answer summary requests by only listing raw
examples unless the user explicitly asked for examples.

When the user asks to show, list, or give examples, the final answer must list the
returned examples individually. Include each example's customer instruction and support
response. Do not replace example listings with a high-level summary. When referring to
dataset categories in the answer, prefer the exact category label from the data, such as
CONTACT, REFUND, or SHIPPING, instead of generic paraphrases like "issue" unless the user
explicitly used that wording.

When the user asks to compare categories, intents, or groups, the final answer must
explicitly describe each side of the comparison and then state the overlap or key
difference. Prefer concrete counts and names over vague wording.

Use prior turns from the same session to resolve follow-up requests. For requests like
"show 3 more", continue the same example set with the correct next offset. For requests
like "what about refunds?" or "what is the total count of the last two?", reuse the
relevant earlier counts, filters, and results from the conversation history.

If a user profile is provided, use it only as distilled long-term context such as the
user's name, preferences, or frequent interests. Do not treat it as a replay of the
whole conversation.

After a tool returns a valid observation, produce the final answer from that observation
unless another different tool call is clearly required. Do not repeat the same tool call
with the same arguments after it has returned data.

When `show_examples` returns fewer examples than requested and `truncated` is true,
mention in your answer that only the available examples are shown and that the dataset
has fewer remaining matches than requested.
"""


_CHECKPOINTERS: dict[str, SqliteSaver] = {}
_CHECKPOINTER_CONNECTIONS: dict[str, sqlite3.Connection] = {}
# Cache key: (nebius_model, use_mcp, mcp_server_url) — only the fields that
# change the graph structure.  Using the full Settings object would work today
# because it is a frozen dataclass, but an explicit tuple is safer and clearer.
_GRAPH_CACHE: dict[tuple[str, bool, str | None], object] = {}


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


def _normalize_session_id(session_id: str | None) -> str:
    if session_id is None:
        return DEFAULT_SESSION_ID
    stripped = session_id.strip().casefold()
    return stripped or DEFAULT_SESSION_ID


def _checkpoint_path(settings: Settings) -> str:
    return os.path.abspath(settings.checkpoint_db_path)


def get_checkpointer(settings: Settings | None = None) -> SqliteSaver:
    """Return a reusable SQLite-backed LangGraph checkpointer."""

    settings = settings or get_settings()
    path = _checkpoint_path(settings)
    if path not in _CHECKPOINTERS:
        connection = sqlite3.connect(path, check_same_thread=False)
        _CHECKPOINTER_CONNECTIONS[path] = connection
        _CHECKPOINTERS[path] = SqliteSaver(connection)
    return _CHECKPOINTERS[path]


def reset_runtime_caches() -> None:
    """Clear cached graphs and close checkpoint connections.

    This is mainly useful in tests to simulate an application restart.
    """

    _GRAPH_CACHE.clear()
    _CHECKPOINTERS.clear()
    for connection in _CHECKPOINTER_CONNECTIONS.values():
        connection.close()
    _CHECKPOINTER_CONNECTIONS.clear()


def reset_checkpoint_db(settings: Settings | None = None) -> tuple[str, str]:
    """Delete both persistent memory stores and clear cached connections."""

    settings = settings or get_settings()
    path = _checkpoint_path(settings)
    reset_runtime_caches()
    if os.path.exists(path):
        os.remove(path)
    profile_dir = reset_user_profiles(settings)
    return path, profile_dir


def _router_node(state: AgentState) -> dict[str, str | list[AIMessage]]:
    last_user_message = next(
        (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
        None,
    )
    query = last_user_message.content if last_user_message else ""
    has_pending = bool(state.get("pending_suggestion"))
    decision = route_query(str(query), has_pending_suggestion=has_pending)
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
    if state["route"] == "out_of_scope":
        return END
    if state["route"] == "profile":
        return "profile"
    if state["route"] == "profile_update":
        return "profile_update"
    if state["route"] == "session_memory":
        return "session_memory"
    if state["route"] == "recommend":
        return "recommend"
    if state["route"] == "confirm":
        return "confirm"
    if state["route"] == "refine":
        return "refine"
    return "agent"


def build_graph(
    settings: Settings | None = None,
    use_mcp: bool = False,
    mcp_server_url: str | None = None,
):
    """Compile the LangGraph ReAct graph used by the CLI."""

    settings = settings or get_settings()
    cache_key = (settings.nebius_model, use_mcp, mcp_server_url)
    cached_graph = _GRAPH_CACHE.get(cache_key)
    if cached_graph is not None:
        return cached_graph

    dataset_tools = get_dataset_tools(use_mcp=use_mcp, mcp_server_url=mcp_server_url)
    base_llm = build_llm(settings)          # unbound — for confirm/refine formatting
    llm = base_llm.bind_tools(dataset_tools) # tool-bound — for the ReAct agent node
    tool_node = ToolNode(dataset_tools)

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
        if state.get("profile_context"):
            messages.append(SystemMessage(content=f"Known user profile:\n{state['profile_context']}"))
        messages.extend(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    def profile_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [AIMessage(content=state["profile_context"])]}

    def profile_update_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [AIMessage(content=state["profile_context"])]}

    def session_memory_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        from collections import Counter

        human_messages = [
            str(message.content)
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ]
        # Exclude the current "what did I ask?" question itself.
        if human_messages:
            human_messages = human_messages[:-1]

        if not human_messages:
            answer = "This is the first question in the current session, so there is no earlier conversation to summarize yet."
        else:
            answer_lines = ["So far in this session, you asked:"]
            for index, question in enumerate(human_messages, start=1):
                answer_lines.append(f"{index}. {question}")

            # Summarise which dataset topics came up most in this session.
            session_topics: Counter[str] = Counter()
            for question in human_messages:
                session_topics.update(_extract_topics(question))
            if session_topics:
                top = session_topics.most_common(3)
                topic_str = ", ".join(f"{t} ({c})" for t, c in top)
                answer_lines.append(f"\nThe most-visited topics in this session were: {topic_str}.")

            answer = "\n".join(answer_lines)

        return {"messages": [AIMessage(content=answer)]}

    def recommend_node(state: AgentState) -> dict:
        """Propose a follow-up query without executing it."""
        from customer_support_agent.profile import UserProfile, _extract_topics
        from collections import Counter as _Counter

        # Check whether the user named a category explicitly in their request
        # (e.g. "suggest a question about REFUND").
        last_human = next(
            (str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        category_hint = extract_category_hint(last_human)

        # Build a lightweight profile stub from whatever topic context we have.
        stub_profile = UserProfile(user_id="__stub__")
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                stub_profile.topic_counts = dict(
                    _Counter(stub_profile.topic_counts) + _extract_topics(str(msg.content))
                )

        suggestion = build_suggestion(stub_profile, state["messages"], category_hint=category_hint)
        if suggestion is None:
            return {
                "pending_suggestion": None,
                "messages": [
                    AIMessage(
                        content=(
                            "I don't have enough context yet to suggest a useful next query. "
                            "Try asking about a specific category like REFUND or SHIPPING first."
                        )
                    )
                ],
            }

        reply = (
            f"{suggestion['description']}\n\n"
            f"Should I go ahead, or would you like to refine the suggestion?"
        )
        return {
            "pending_suggestion": suggestion,
            "messages": [AIMessage(content=reply)],
        }

    def refine_node(state: AgentState) -> dict:
        """Use the LLM to interpret free-text refinement and update the suggestion."""
        current = state.get("pending_suggestion") or {}
        last_human = next(
            (str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )

        # First try the local heuristic — fast and free.
        updated = refine_suggestion(current, last_human)

        # If the local rewrite changed something, use it directly.
        if updated != current:
            reply = f"{updated['description']}\n\nShall I go ahead?"
            return {"pending_suggestion": updated, "messages": [AIMessage(content=reply)]}

        # Otherwise fall back to the LLM for genuinely open-ended refinement.
        refine_prompt = (
            f"The user was shown this query suggestion:\n\"{current.get('description', '')}\"\n\n"
            f"They replied: \"{last_human}\"\n\n"
            f"Produce a revised one-sentence suggestion that incorporates their feedback. "
            f"The suggestion must describe a query against the Bitext customer support dataset "
            f"and end with 'Shall I go ahead?'. Do not execute any tools."
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=refine_prompt)]
        response = base_llm.invoke(messages)
        refined_description = str(response.content).strip()

        # Carry forward the same tool/args, just update the description.
        updated = {**current, "description": refined_description.replace("Shall I go ahead?", "").strip()}
        reply = f"{updated['description']}\n\nShall I go ahead?"
        return {"pending_suggestion": updated, "messages": [AIMessage(content=reply)]}

    def confirm_node(state: AgentState) -> dict:
        """Execute the pending suggestion and clear it."""
        suggestion = state.get("pending_suggestion")
        if not suggestion:
            return {
                "pending_suggestion": None,
                "messages": [AIMessage(content="There is no pending suggestion to execute.")],
            }

        tool_name = suggestion["tool"]
        args = suggestion["args"]

        # Find and call the matching tool directly.
        matching = next((t for t in dataset_tools if t.name == tool_name), None)
        if matching is None:
            return {
                "pending_suggestion": None,
                "messages": [
                    AIMessage(content=f"I couldn't find the tool '{tool_name}' to execute.")
                ],
            }

        try:
            result = matching.invoke(args)
        except Exception as exc:
            return {
                "pending_suggestion": None,
                "messages": [AIMessage(content=f"The tool call failed: {exc}")],
            }

        # Hand the result to the LLM to produce a readable answer.
        format_prompt = (
            f"Here is the result of {tool_name}({args}):\n{result}\n\n"
            f"Write a concise, readable answer for the user based on this data."
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=format_prompt)]
        response = base_llm.invoke(messages)
        return {
            "pending_suggestion": None,
            "messages": [AIMessage(content=str(response.content))],
        }

    graph = StateGraph(AgentState)
    graph.add_node("router", _router_node)
    graph.add_node("agent", agent_node)
    graph.add_node("profile", profile_node)
    graph.add_node("profile_update", profile_update_node)
    graph.add_node("session_memory", session_memory_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("refine", refine_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _next_after_router,
        {
            "agent": "agent",
            "profile": "profile",
            "profile_update": "profile_update",
            "session_memory": "session_memory",
            "recommend": "recommend",
            "refine": "refine",
            "confirm": "confirm",
            END: END,
        },
    )
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    graph.add_edge("profile", END)
    graph.add_edge("profile_update", END)
    graph.add_edge("session_memory", END)
    graph.add_edge("recommend", END)
    graph.add_edge("refine", END)
    graph.add_edge("confirm", END)
    compiled = graph.compile(checkpointer=get_checkpointer(settings))
    _GRAPH_CACHE[cache_key] = compiled
    return compiled


def invoke_agent(
    question: str,
    settings: Settings | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    use_mcp: bool = False,
    mcp_server_url: str | None = None,
) -> list[BaseMessage]:
    """Run the graph once and return all emitted messages."""

    settings = settings or get_settings()
    graph = build_graph(settings, use_mcp=use_mcp, mcp_server_url=mcp_server_url)
    session_id = _normalize_session_id(session_id)
    user_id = normalize_user_id(user_id or session_id)

    # Peek at the checkpoint to detect whether a suggestion is pending,
    # so the router receives the correct has_pending_suggestion flag.
    checkpoint = graph.get_state(config={"configurable": {"thread_id": session_id}})
    has_pending = bool(
        checkpoint and checkpoint.values and checkpoint.values.get("pending_suggestion")
    )
    route_decision = route_query(question, has_pending_suggestion=has_pending)

    def latest_topic_context(messages: list[BaseMessage]) -> str | None:
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            content = str(message.content)
            if content == question:
                continue
            if route_query(content).route in {"structured", "unstructured"}:
                return content
        return None

    if route_decision.route == "profile_update":
        profile = update_user_profile(user_id, question, settings)
        profile_context = render_profile_update_acknowledgement(profile)
    else:
        profile = load_user_profile(user_id, settings)
        profile_context = render_profile_answer(profile)
        if profile.facts or profile.name or profile.preferences or profile.topic_counts:
            profile_context = render_profile_summary(profile)

    # LangGraph's recursion_limit counts graph node executions, not full ReAct
    # iterations. Each tool cycle usually costs agent -> tools -> agent steps.
    graph_step_limit = settings.max_iterations * 2 + 4
    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "profile_context": profile_context,
            },
            config={
                "recursion_limit": graph_step_limit,
                "configurable": {"thread_id": session_id},
            },
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
    topic_context = latest_topic_context(result["messages"])
    if route_decision.route not in {"profile", "profile_update", "recommend", "confirm", "refine"}:
        update_user_profile(user_id, question, settings, topic_context=topic_context)
    return result["messages"]
