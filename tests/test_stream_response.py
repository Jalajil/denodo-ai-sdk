"""Non-verbose streams retain the SQL and data returned by the pipeline."""

import asyncio
import json
from importlib import import_module

import pytest


@pytest.mark.parametrize("endpoint", ["streamAnswerQuestion", "streamAnswerQuestionUsingViews"])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_stream_returns_json_or_text_for_results_and_clarifications(monkeypatch, endpoint, verbose, ambiguous):
    module = import_module(f"api.endpoints.{endpoint}")
    request_type = getattr(module, f"{endpoint}Request")
    request_values = {"question": "Find the customer", "verbose": verbose}
    if "UsingViews" in endpoint:
        request_values["vector_search_tables"] = []
    request = request_type(**request_values)
    payload = {
        "answer": "The customer is Smith, Jane.",
        "sql_query": "SELECT name, city FROM customers",
        "execution_result": {"rows": [["Smith, Jane", None]]},
    }

    async def retrieve(**kwargs):
        tables = [{"view_name": "customers", "view_json": {"id": "1", "tableName": "customers", "schema": []}}]
        return tables, [], {}, "", {}

    async def categorize(**kwargs):
        category_response = (
            "<ambiguous_input><type>TEMP</type><cl_question>Which year?</cl_question></ambiguous_input>"
            if ambiguous else ""
        )
        return "SQL", category_response, [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def process_sql_category(**kwargs):
        assert not ambiguous, "Clarification must not execute a query"
        return payload

    monkeypatch.setattr(module.state_manager, "get_llm", lambda **kwargs: object())
    monkeypatch.setattr(module.state_manager, "get_vector_store", lambda **kwargs: object())
    monkeypatch.setattr(module.ai_tools, "get_relevant_tables", retrieve)
    monkeypatch.setattr(module.ai_tools, "sql_category", categorize)
    monkeypatch.setattr(module, "process_sql_category", process_sql_category)

    async def call_endpoint():
        if "UsingViews" in endpoint:
            response = await module.streamAnswerQuestionUsingViews(request, "test-auth", {})
        else:
            response = await module.process_stream_question(request, "test-auth", {})
        chunks = [chunk async for chunk in response.body_iterator]
        body = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in chunks)
        return response, body

    response, body = asyncio.run(call_endpoint())
    if verbose:
        assert response.media_type == "text/plain"
        if ambiguous:
            assert "Which year?" in body
        else:
            assert body == payload["answer"]
    else:
        assert response.media_type == "application/json"
        data = json.loads(body)
        if ambiguous:
            assert "Which year?" in data["answer"]
            assert data["execution_result"] == {}
            assert data["sql_query"] == ""
        else:
            assert data == payload
