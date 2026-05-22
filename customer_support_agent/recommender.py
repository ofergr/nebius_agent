"""Local heuristic for Bonus B query recommendations.

No LLM is involved here.  The suggester inspects the user's profile topic
counts and the current session's HumanMessage history to propose the most
useful next query, then returns a structured suggestion dict that the graph
stores in ``AgentState.pending_suggestion``.

Suggestion dict schema
----------------------
{
    "tool":        str,           # tool name to call on confirm
    "args":        dict,          # kwargs for that tool
    "description": str,           # natural-language proposal shown to the user
}
"""

from __future__ import annotations

import re
from collections import Counter

from langchain_core.messages import BaseMessage, HumanMessage

from customer_support_agent.profile import UserProfile


# ---------------------------------------------------------------------------
# What the user might want to do with a topic, in priority order.
# Each entry is (action_label, tool, args_template).
# We skip an action if the session history shows they already did it.
# ---------------------------------------------------------------------------
_TOPIC_ACTIONS = [
    ("distribution of intents", "distribution", {"group_by": "intent"}),
    ("show examples",           "show_examples", {"limit": 5}),
    ("count rows",              "count_rows",    {}),
    ("summary",                 "sample_responses_for_summary", {"limit": 10}),
]

# Maps profile topic keys → canonical dataset category labels.
_TOPIC_TO_CATEGORY: dict[str, str] = {
    "refund":    "REFUND",
    "delivery":  "DELIVERY",
    "shipping":  "SHIPPING",
    "account":   "ACCOUNT",
    "order":     "ORDER",
    "billing":   "BILLING",
    "feedback":  "FEEDBACK",
}


def _session_tool_calls(messages: list[BaseMessage]) -> list[tuple[str, dict]]:
    """Return (tool_name, args) pairs for every tool call in the session so far."""
    from langchain_core.messages import AIMessage

    calls = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                calls.append((call["name"], call.get("args", {})))
    return calls


def _already_did(tool: str, category: str, session_calls: list[tuple[str, dict]]) -> bool:
    """Return True if the session already called *tool* for *category*."""
    for name, args in session_calls:
        if name == tool and (args.get("category") or "").upper() == category.upper():
            return True
    return False


def _best_topic(profile: UserProfile, session_messages: list[BaseMessage]) -> str | None:
    """Pick the most relevant topic: prefer the most-asked profile topic that
    has a known category mapping, falling back to the most-recent session topic."""

    # 1. Profile-based: most frequent topic with a known category.
    if profile.topic_counts:
        for topic, _ in Counter(profile.topic_counts).most_common():
            if topic in _TOPIC_TO_CATEGORY:
                return topic

    # 2. Session-based: look at recent HumanMessages for dataset keywords.
    from customer_support_agent.profile import _extract_topics

    session_topics: Counter[str] = Counter()
    for message in reversed(session_messages):
        if isinstance(message, HumanMessage):
            session_topics.update(_extract_topics(str(message.content)))
        if len(session_topics) > 0:
            break  # use the most recent message that had a topic

    for topic, _ in session_topics.most_common():
        if topic in _TOPIC_TO_CATEGORY:
            return topic

    return None


def extract_category_hint(text: str) -> str | None:
    """Return a canonical category name if the text explicitly names one."""
    upper = text.upper()
    for category in _TOPIC_TO_CATEGORY.values():
        if category in upper:
            return category
    return None


def build_suggestion(
    profile: UserProfile,
    session_messages: list[BaseMessage],
    category_hint: str | None = None,
) -> dict | None:
    """Return the best suggestion dict, or None if nothing useful can be proposed.

    If ``category_hint`` is provided it is used directly, bypassing the
    profile/session heuristic that normally picks the topic.
    """

    if category_hint is not None:
        category = category_hint.upper()
    else:
        topic = _best_topic(profile, session_messages)
        if topic is None:
            return None
        category = _TOPIC_TO_CATEGORY[topic]
    session_calls = _session_tool_calls(session_messages)

    for action_label, tool, extra_args in _TOPIC_ACTIONS:
        if not _already_did(tool, category, session_calls):
            args = {**extra_args, "category": category}
            description = _build_description(action_label, tool, category)
            return {
                "tool": tool,
                "args": args,
                "description": description,
            }

    # They've done everything for this topic — suggest a different category.
    other_categories = [c for t, c in _TOPIC_TO_CATEGORY.items() if t != topic]
    if other_categories:
        category = other_categories[0]
        action_label, tool, extra_args = _TOPIC_ACTIONS[0]
        args = {**extra_args, "category": category}
        return {
            "tool": tool,
            "args": args,
            "description": _build_description(action_label, tool, category),
        }

    return None


def _build_description(action_label: str, tool: str, category: str) -> str:
    if tool == "distribution":
        return (
            f"Based on your interests, you might want to see the distribution of intents "
            f"in the {category} category."
        )
    if tool == "show_examples":
        return (
            f"You could look at 5 examples from the {category} category to see "
            f"what those interactions look like."
        )
    if tool == "count_rows":
        return (
            f"You haven't counted the {category} rows yet — "
            f"that would tell you how large that category is."
        )
    if tool == "sample_responses_for_summary":
        return (
            f"You might want a qualitative summary of how agents respond "
            f"to {category} requests."
        )
    return f"You might want to explore the {category} category with {tool}."


def refine_suggestion(current: dict, user_feedback: str) -> dict:
    """Adjust a pending suggestion based on free-text user feedback.

    This is a lightweight local rewrite — the LLM handles the real open-ended
    refinement in ``refine_node``, but we expose this for testing the heuristic
    path directly.
    """
    lowered = user_feedback.casefold()
    updated = dict(current)

    # Tool override keywords.
    if re.search(r"\bexample(s)?\b", lowered):
        updated["tool"] = "show_examples"
        updated["args"] = {**current["args"], "limit": 5}
        updated["args"].pop("group_by", None)
        updated["args"].pop("top_n", None)
        category = current["args"].get("category", "")
        updated["description"] = (
            f"How about showing 5 examples from the {category} category instead?"
        )
    elif re.search(r"\bdistribution\b|\bintent(s)?\b", lowered):
        updated["tool"] = "distribution"
        updated["args"] = {"group_by": "intent", "category": current["args"].get("category")}
        category = current["args"].get("category", "")
        updated["description"] = (
            f"How about showing the intent distribution for the {category} category?"
        )
    elif re.search(r"\bcount\b|\bhow many\b", lowered):
        updated["tool"] = "count_rows"
        updated["args"] = {"category": current["args"].get("category")}
        category = current["args"].get("category", "")
        updated["description"] = (
            f"How about counting the rows in the {category} category?"
        )
    elif re.search(r"\bsummar(y|ize)\b", lowered):
        updated["tool"] = "sample_responses_for_summary"
        updated["args"] = {"category": current["args"].get("category"), "limit": 10}
        updated["args"].pop("group_by", None)
        category = current["args"].get("category", "")
        updated["description"] = (
            f"How about a qualitative summary of {category} responses?"
        )

    # Category override keywords.
    for topic, category in _TOPIC_TO_CATEGORY.items():
        if topic in lowered:
            updated["args"] = {**updated["args"], "category": category}
            # Rebuild description with the new category.
            tool = updated["tool"]
            action_label = next(
                (a for a, t, _ in _TOPIC_ACTIONS if t == tool), "explore"
            )
            updated["description"] = _build_description(action_label, tool, category)
            break

    return updated
