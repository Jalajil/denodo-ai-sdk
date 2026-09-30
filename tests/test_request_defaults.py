"""Public request defaults and explicit caller overrides."""

from importlib import import_module

import pytest


QUESTION_ENDPOINTS = [
    "answerQuestion",
    "answerDataQuestion",
    "answerQuestionUsingViews",
    "streamAnswerQuestion",
    "streamAnswerQuestionUsingViews",
]


def request_model(endpoint):
    module = import_module(f"api.endpoints.{endpoint}")
    return getattr(module, f"{endpoint}Request")


def minimal_request(endpoint, **overrides):
    values = {"question": "Find the orders", **overrides}
    if "UsingViews" in endpoint:
        values["vector_search_tables"] = []
    return request_model(endpoint)(**values)


@pytest.mark.parametrize("endpoint", QUESTION_ENDPOINTS)
def test_minimal_question_requests_enable_review_and_return_data(endpoint):
    request = minimal_request(endpoint)
    assert request.enable_query_reviewer is True
    assert request.verbose is False
    assert request.vector_search_sample_data_k == 8
    if hasattr(request, "mode"):
        assert request.mode == "data"


@pytest.mark.parametrize("endpoint", QUESTION_ENDPOINTS)
def test_callers_can_override_question_defaults(endpoint):
    request = minimal_request(
        endpoint,
        enable_query_reviewer=False,
        verbose=True,
        vector_search_sample_data_k=2,
        mode="metadata",
    )
    assert request.enable_query_reviewer is False
    assert request.verbose is True
    assert request.vector_search_sample_data_k == 2
    if hasattr(request, "mode"):
        assert request.mode == "metadata"


@pytest.mark.parametrize("endpoint", ["answerMetadataQuestion", "deepQuery"])
def test_other_sample_retrieval_endpoints_request_eight_rows(endpoint):
    request = minimal_request(endpoint)
    assert request.vector_search_sample_data_k == 8
    if hasattr(request, "verbose"):
        assert request.verbose is False
