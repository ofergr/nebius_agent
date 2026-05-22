"""Tests for Bonus B: query recommender."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from customer_support_agent.profile import UserProfile
from customer_support_agent.recommender import build_suggestion, refine_suggestion
from customer_support_agent.router import route_query


# ---------------------------------------------------------------------------
# Router — new routes
# ---------------------------------------------------------------------------


def test_router_detects_recommend_phrase():
    decision = route_query("What should I query next?")
    assert decision.route == "recommend"


def test_router_detects_recommend_suggest_variant():
    decision = route_query("What do you suggest?")
    assert decision.route == "recommend"


def test_router_detects_confirm_when_pending():
    decision = route_query("yes", has_pending_suggestion=True)
    assert decision.route == "confirm"


def test_router_detects_do_it_when_pending():
    decision = route_query("do it", has_pending_suggestion=True)
    assert decision.route == "confirm"


def test_router_does_not_confirm_without_pending():
    # "yes" with no pending suggestion should NOT route to confirm.
    decision = route_query("yes", has_pending_suggestion=False)
    assert decision.route != "confirm"


def test_router_detects_refine_when_pending():
    # A non-confirm reply with no dataset term while a suggestion is pending → refine.
    decision = route_query("make it shorter instead", has_pending_suggestion=True)
    assert decision.route == "refine"


def test_router_detects_refine_for_preference_phrase():
    # "I'd rather see..." has no dataset terms → refine.
    decision = route_query("I'd rather try something different", has_pending_suggestion=True)
    assert decision.route == "refine"


def test_router_dataset_question_overrides_refine():
    # A real dataset question while pending should still be routed normally.
    decision = route_query("How many refund requests did we get?", has_pending_suggestion=True)
    assert decision.route in {"structured", "unstructured"}


# ---------------------------------------------------------------------------
# build_suggestion — local heuristic
# ---------------------------------------------------------------------------


def _make_profile(topic_counts: dict) -> UserProfile:
    return UserProfile(user_id="test", topic_counts=topic_counts)


def _tool_call_message(tool_name: str, category: str) -> AIMessage:
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": tool_name, "args": {"category": category}, "id": "x"}]
    return msg


def test_build_suggestion_returns_dict_for_known_topic():
    profile = _make_profile({"refund": 3})
    suggestion = build_suggestion(profile, [])
    assert suggestion is not None
    assert "tool" in suggestion
    assert "args" in suggestion
    assert "description" in suggestion


def test_build_suggestion_targets_most_frequent_topic():
    profile = _make_profile({"refund": 5, "shipping": 1})
    suggestion = build_suggestion(profile, [])
    assert suggestion["args"].get("category") == "REFUND"


def test_build_suggestion_skips_already_done_action():
    profile = _make_profile({"refund": 3})
    # Simulate that distribution was already called for REFUND.
    prior_call = _tool_call_message("distribution", "REFUND")
    suggestion = build_suggestion(profile, [prior_call])
    assert suggestion is not None
    # Should suggest something other than distribution.
    assert suggestion["tool"] != "distribution"


def test_build_suggestion_returns_none_without_context():
    profile = _make_profile({})
    suggestion = build_suggestion(profile, [])
    assert suggestion is None


def test_build_suggestion_falls_back_to_session_history():
    profile = _make_profile({})
    messages = [HumanMessage(content="How many refund requests did we get?")]
    suggestion = build_suggestion(profile, messages)
    assert suggestion is not None
    assert suggestion["args"].get("category") == "REFUND"


# ---------------------------------------------------------------------------
# refine_suggestion — local heuristic rewrite
# ---------------------------------------------------------------------------


def _base_suggestion(tool="distribution", category="REFUND") -> dict:
    return {
        "tool": tool,
        "args": {"group_by": "intent", "category": category},
        "description": f"See the intent distribution for {category}.",
    }


def test_refine_to_examples():
    updated = refine_suggestion(_base_suggestion(), "I'd rather see examples instead")
    assert updated["tool"] == "show_examples"
    assert "REFUND" in updated["description"]


def test_refine_to_count():
    updated = refine_suggestion(_base_suggestion(), "just give me the count")
    assert updated["tool"] == "count_rows"


def test_refine_changes_category():
    updated = refine_suggestion(_base_suggestion(category="REFUND"), "show me shipping instead")
    assert updated["args"]["category"] == "SHIPPING"


def test_refine_no_change_leaves_suggestion_identical():
    base = _base_suggestion()
    updated = refine_suggestion(base, "hmm")
    # No recognised keyword → returns same tool/args.
    assert updated["tool"] == base["tool"]
    assert updated["args"].get("category") == base["args"].get("category")
