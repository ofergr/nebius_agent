"""Tests for the query router."""

from customer_support_agent.router import route_query


def test_routes_structured_dataset_question() -> None:
    decision = route_query("What is the distribution of intents in the ACCOUNT category?")

    assert decision.route == "structured"


def test_routes_unstructured_dataset_question() -> None:
    decision = route_query("Summarize how agents respond to complaint intents.")

    assert decision.route == "unstructured"


def test_routes_keyword_search_dataset_question() -> None:
    decision = route_query("How many rows mention a delayed delivery")

    assert decision.route == "structured"


def test_routes_compare_question_without_explicit_category_keyword() -> None:
    decision = route_query("Compare refund and shipping in terms of common intents")

    assert decision.route == "structured"


def test_routes_structured_analysis_pattern_without_dataset_term() -> None:
    decision = route_query("What are the most common intents?")

    assert decision.route == "structured"


def test_routes_out_of_scope_question() -> None:
    decision = route_query("Who is the president of France?")

    assert decision.route == "out_of_scope"


def test_routes_profile_question() -> None:
    decision = route_query("What do you remember about me?")

    assert decision.route == "profile"


def test_routes_session_memory_question() -> None:
    decision = route_query("What questions did I ask so far?")

    assert decision.route == "session_memory"


def test_routes_profile_question_with_variant_wording() -> None:
    decision = route_query("What do you know about be?")

    assert decision.route == "profile"


def test_routes_follow_up_example_question() -> None:
    decision = route_query("Can you give me an example?")

    assert decision.route == "structured"


def test_routes_profile_update_statement() -> None:
    decision = route_query("my name is ofer and i am 55 years old")

    assert decision.route == "profile_update"


def test_routes_give_me_more_question() -> None:
    decision = route_query("give me 2 more")

    assert decision.route == "structured"
