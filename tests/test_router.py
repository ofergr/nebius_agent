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
