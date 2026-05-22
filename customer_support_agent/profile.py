"""Persistent user profile helpers for semantic memory."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from customer_support_agent.config import Settings, get_settings


@dataclass
class UserProfile:
    """Distilled facts about a user, stored separately from conversation history."""

    user_id: str
    name: str | None = None
    preferences: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    topic_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str | None = None


TOPIC_KEYWORDS = {
    "refund": ("refund", "refunds", "money back", "reimbursement", "compensation", "return"),
    "shipping": ("shipping", "shipment", "delivery", "package", "track"),
    "account": ("account", "login", "log in", "sign in", "password"),
    "order": ("order", "purchase", "cancel order"),
    "billing": ("billing", "invoice", "payment", "charged", "fee"),
    "feedback": ("feedback", "complaint", "review"),
}

PROFILE_UPDATE_PATTERNS = (
    r"\bmy name is\b",
    r"\bcall me\b",
    r"\bi am \d{1,3} years old\b",
    r"\bi'm \d{1,3} years old\b",
    r"\bi prefer\b",
    r"\bplease use\b",
    r"\bi work as\b",
    r"\bi work in\b",
    r"\bi live in\b",
)


def normalize_user_id(user_id: str | None) -> str:
    """Normalize user IDs for profile storage."""

    if user_id is None:
        return "default"
    normalized = user_id.strip().casefold()
    return normalized or "default"


def _profile_dir(settings: Settings) -> str:
    return os.path.abspath(settings.user_profile_dir)


def _profile_path(user_id: str, settings: Settings) -> str:
    safe_user_id = normalize_user_id(user_id).replace("/", "_")
    return os.path.join(_profile_dir(settings), f"{safe_user_id}.json")


def load_user_profile(user_id: str | None, settings: Settings | None = None) -> UserProfile:
    """Load a user's persistent profile from disk."""

    settings = settings or get_settings()
    normalized_user_id = normalize_user_id(user_id)
    path = _profile_path(normalized_user_id, settings)
    if not os.path.exists(path):
        return UserProfile(user_id=normalized_user_id)

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return UserProfile(
        user_id=normalized_user_id,
        name=payload.get("name"),
        preferences=list(payload.get("preferences", [])),
        facts=list(payload.get("facts", [])),
        topic_counts=dict(payload.get("topic_counts", {})),
        updated_at=payload.get("updated_at"),
    )


def save_user_profile(profile: UserProfile, settings: Settings | None = None) -> str:
    """Persist a user's distilled profile to disk."""

    settings = settings or get_settings()
    directory = _profile_dir(settings)
    os.makedirs(directory, exist_ok=True)
    path = _profile_path(profile.user_id, settings)
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "user_id": profile.user_id,
                "name": profile.name,
                "preferences": profile.preferences,
                "facts": profile.facts,
                "topic_counts": profile.topic_counts,
                "updated_at": profile.updated_at,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return path


def reset_user_profiles(settings: Settings | None = None) -> str:
    """Delete all persisted user profiles."""

    settings = settings or get_settings()
    directory = _profile_dir(settings)
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    return directory


def _append_unique(items: list[str], value: str) -> None:
    normalized_existing = {item.casefold() for item in items}
    cleaned = value.strip().rstrip(".")
    if cleaned and cleaned.casefold() not in normalized_existing:
        items.append(cleaned)


def _extract_name(text: str) -> str | None:
    patterns = (
        r"\bmy name is ([A-Za-z]+(?: [A-Za-z]+){0,2})(?=\s+(?:and|but)\b|[,.?!]|$)",
        r"\bcall me ([A-Za-z]+(?: [A-Za-z]+){0,2})(?=\s+(?:and|but)\b|[,.?!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_preferences(text: str) -> list[str]:
    patterns = (
        r"\bi prefer ([^.?!]+)",
        r"\bplease use ([^.?!]+)",
        r"\bplease call me ([^.?!]+)",
        r"\bi want ([^.?!]+)",
    )
    results: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            preference = match.group(1).strip()
            if len(preference) <= 120:
                results.append(preference)
    return results


def _extract_facts(text: str) -> list[str]:
    patterns = (
        r"\bi am (\d{1,3}) years old\b",
        r"\bi'm (\d{1,3}) years old\b",
        r"\bi work as ([^.?!]+)",
        r"\bi work in ([^.?!]+)",
        r"\bi live in ([^.?!]+)",
        r"\bi am a[n]? ([^.?!]+)",
    )
    results: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if "years old" in match.group(0).casefold():
                fact = f"Age: {match.group(1)}"
            else:
                fact = match.group(0).strip()
            if len(fact) <= 120:
                results.append(fact)
    return results


def _extract_topics(text: str) -> Counter[str]:
    lowered = text.casefold()
    counts: Counter[str] = Counter()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(keyword.casefold())}\b", lowered) for keyword in keywords):
            counts[topic] += 1
    return counts


def update_user_profile(
    user_id: str | None,
    message: str,
    settings: Settings | None = None,
    topic_context: str | None = None,
) -> UserProfile:
    """Update a user's persistent semantic profile from a new message."""

    settings = settings or get_settings()
    profile = load_user_profile(user_id, settings)

    if name := _extract_name(message):
        profile.name = name

    for preference in _extract_preferences(message):
        _append_unique(profile.preferences, preference)

    for fact in _extract_facts(message):
        _append_unique(profile.facts, fact)

    message_topics = _extract_topics(message)
    if not message_topics and topic_context:
        message_topics = _extract_topics(topic_context)

    topic_counts = Counter(profile.topic_counts)
    topic_counts.update(message_topics)
    profile.topic_counts = dict(topic_counts)

    save_user_profile(profile, settings)
    return profile


def message_has_profile_update(message: str) -> bool:
    """Return whether a message contains self-disclosed profile information."""

    lowered = message.casefold()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in PROFILE_UPDATE_PATTERNS)


def render_profile_summary(profile: UserProfile) -> str:
    """Render the user profile as a concise assistant-facing summary."""

    parts: list[str] = []
    if profile.name:
        parts.append(f"Name: {profile.name}")
    if profile.preferences:
        parts.append("Preferences: " + "; ".join(profile.preferences[:5]))
    if profile.facts:
        parts.append("Facts: " + "; ".join(profile.facts[:5]))
    if profile.topic_counts:
        top_topics = Counter(profile.topic_counts).most_common(3)
        parts.append(
            "Frequent topics: " + ", ".join(f"{topic} ({count})" for topic, count in top_topics)
        )
    return "\n".join(parts)


def render_profile_answer(profile: UserProfile) -> str:
    """Render the profile as a direct answer to 'what do you remember about me?'."""

    lines: list[str] = []
    if profile.name:
        lines.append(f"I remember your name as {profile.name}.")
    if profile.preferences:
        lines.append("Your preferences I have stored: " + "; ".join(profile.preferences[:5]) + ".")
    if profile.facts:
        lines.append("Other facts I have stored: " + "; ".join(profile.facts[:5]) + ".")
    if profile.topic_counts:
        top_topics = Counter(profile.topic_counts).most_common(3)
        lines.append(
            "You often ask about: " + ", ".join(f"{topic} ({count})" for topic, count in top_topics) + "."
        )
    if not lines:
        return (
            "I don't have much stored about you yet. If you tell me your name, preferences, "
            "or what topics you care about, I'll keep that in your profile."
        )
    return "\n".join(lines)


def render_profile_update_acknowledgement(profile: UserProfile) -> str:
    """Render a short acknowledgement after learning new user details."""

    parts: list[str] = ["Thanks, I'll remember that."]
    remembered: list[str] = []
    if profile.name:
        remembered.append(f"your name is {profile.name}")
    age_fact = next((fact for fact in profile.facts if fact.startswith("Age: ")), None)
    if age_fact:
        remembered.append(age_fact.casefold().replace("age: ", "you are ") + " years old")
    if remembered:
        parts.append("I now know that " + " and ".join(remembered) + ".")
    return " ".join(parts)
