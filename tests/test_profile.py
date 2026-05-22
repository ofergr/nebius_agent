"""Tests for persistent user profile memory."""

from __future__ import annotations

from customer_support_agent.config import Settings
from customer_support_agent.profile import (
    load_user_profile,
    message_has_profile_update,
    render_profile_answer,
    render_profile_update_acknowledgement,
    reset_user_profiles,
    update_user_profile,
)


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


def test_profile_persists_name_preferences_and_topics(tmp_path) -> None:
    settings = _settings(tmp_path)

    update_user_profile(
        "Alice",
        "My name is Alice. I prefer concise answers and I often ask about refunds.",
        settings,
    )
    update_user_profile("alice", "Please use metric units. Shipping questions matter to me too.", settings)

    profile = load_user_profile("ALICE", settings)

    assert profile.name == "Alice"
    assert "concise answers and I often ask about refunds" in profile.preferences
    assert "metric units" in " ".join(profile.preferences)
    assert profile.topic_counts["refund"] >= 1
    assert profile.topic_counts["shipping"] >= 1


def test_render_profile_answer_for_empty_profile(tmp_path) -> None:
    settings = _settings(tmp_path)
    profile = load_user_profile("new-user", settings)

    answer = render_profile_answer(profile)

    assert "don't have much stored about you yet" in answer


def test_reset_user_profiles_clears_saved_profiles(tmp_path) -> None:
    settings = _settings(tmp_path)
    update_user_profile("alice", "My name is Alice.", settings)

    reset_user_profiles(settings)

    profile = load_user_profile("alice", settings)
    assert profile.name is None


def test_profile_extracts_simple_name_and_age(tmp_path) -> None:
    settings = _settings(tmp_path)

    profile = update_user_profile("ofer", "my name is ofer and i am 55 years old", settings)

    assert profile.name == "ofer"
    assert "Age: 55" in profile.facts


def test_profile_update_acknowledgement_mentions_name_and_age(tmp_path) -> None:
    settings = _settings(tmp_path)
    profile = update_user_profile("ofer", "my name is ofer and i am 55 years old", settings)

    answer = render_profile_update_acknowledgement(profile)

    assert "your name is ofer" in answer.casefold()
    assert "you are 55 years old" in answer.casefold()


def test_message_has_profile_update_detects_self_disclosure() -> None:
    assert message_has_profile_update("My name is ofer and I am 55 years old.") is True


def test_topic_counts_inherit_context_for_generic_followups(tmp_path) -> None:
    settings = _settings(tmp_path)

    update_user_profile("ofer", "How many refund requests did we get?", settings)
    update_user_profile(
        "ofer",
        "can you give me an example?",
        settings,
        topic_context="How many refund requests did we get?",
    )
    update_user_profile(
        "ofer",
        "give me one more example",
        settings,
        topic_context="How many refund requests did we get?",
    )

    profile = load_user_profile("ofer", settings)

    assert profile.topic_counts["refund"] == 3
