"""Query resolution retries against controlled Data Marketplace and LLM outcomes."""

import asyncio
from types import SimpleNamespace

import pytest

from api.utils.ai_tools.types import empty_tokens
from api.utils.data_category import resolve
from api.utils.data_category.fixing import FixOutcome, ReviewOutcome
from utils.data_marketplace.vql_execution_outcomes import ExecutionOutcome, ExecutionStatus


def run_resolution(monkeypatch, execution_statuses, reviews=(), *, reviewer_enabled=True, fixes=()):
    executions = iter(execution_statuses)
    reviews = iter(reviews)
    fixes = iter(fixes)
    executed_queries = []

    async def execute_vql(vql, **kwargs):
        executed_queries.append(vql)
        status = next(executions)
        return ExecutionOutcome(
            status=status,
            data={"rows": [{"name": "found"}]} if status is ExecutionStatus.SUCCESS else {},
            error="Unknown column" if status is ExecutionStatus.EXECUTION_ERROR else "",
        )

    async def reviewer_step(conversation, llm, query, note, session_id):
        verdict, vql = next(reviews)
        return ReviewOutcome(verdict=verdict, vql=vql, thoughts="Review result")

    async def fixer_step(conversation, llm, query, error, session_id):
        return FixOutcome(vql=next(fixes), thoughts="Fix result")

    monkeypatch.setattr(resolve, "execute_vql", execute_vql)
    monkeypatch.setattr(resolve, "reviewer_step", reviewer_step)
    monkeypatch.setattr(resolve, "fixer_step", fixer_step)
    request = SimpleNamespace(
        question="Find the name",
        vql_execute_rows_limit=100,
        vector_search_sample_data_k=8,
        enable_query_reviewer=reviewer_enabled,
        enable_query_fixer=True,
    )
    generated = SimpleNamespace(vql="SELECT name FROM orders", explanation="Find names", tokens=empty_tokens())
    result = asyncio.run(resolve.resolve_query(request, generated, "test-auth", object(), {}, []))
    return result, executed_queries


def test_third_review_can_recover_an_empty_initial_query(monkeypatch):
    result, executed = run_resolution(
        monkeypatch,
        [ExecutionStatus.EMPTY] * 3 + [ExecutionStatus.SUCCESS],
        [("rewrite", f"SELECT name FROM orders WHERE id = {i}") for i in range(1, 4)],
    )
    assert result.resolution is resolve.Resolution.SUCCESS
    assert result.attempts == 3
    assert result.vql == "SELECT name FROM orders WHERE id = 3"
    assert len(executed) == 4


def test_review_stops_after_three_retries_when_all_results_are_empty(monkeypatch):
    result, executed = run_resolution(
        monkeypatch,
        [ExecutionStatus.EMPTY] * 4,
        [("rewrite", f"SELECT name FROM orders WHERE id = {i}") for i in range(1, 4)],
    )
    assert result.resolution is resolve.Resolution.EMPTY
    assert result.attempts == 3
    assert len(executed) == 4


@pytest.mark.parametrize("status", [ExecutionStatus.EMPTY, ExecutionStatus.EXECUTION_ERROR])
def test_review_can_recover_on_third_attempt_after_a_failed_rewrite(monkeypatch, status):
    result, executed = run_resolution(
        monkeypatch,
        [ExecutionStatus.EMPTY, status, status, ExecutionStatus.SUCCESS],
        [("rewrite", f"SELECT name FROM orders WHERE id = {i}") for i in range(1, 4)],
    )
    assert result.resolution is resolve.Resolution.SUCCESS
    assert result.attempts == 3
    assert len(executed) == 4


def test_review_stops_as_soon_as_a_rewrite_succeeds(monkeypatch):
    result, executed = run_resolution(
        monkeypatch,
        [ExecutionStatus.EMPTY, ExecutionStatus.SUCCESS],
        [("rewrite", "SELECT name FROM orders WHERE id = 1")],
    )
    assert result.resolution is resolve.Resolution.SUCCESS
    assert result.attempts == 1
    assert len(executed) == 2


def test_review_keep_verdict_does_not_execute_the_query_again(monkeypatch):
    result, executed = run_resolution(
        monkeypatch,
        [ExecutionStatus.EMPTY],
        [("keep", "SELECT name FROM orders")],
    )
    assert result.resolution is resolve.Resolution.EMPTY
    assert result.attempts == 1
    assert executed == ["SELECT name FROM orders"]


def test_request_can_disable_review(monkeypatch):
    result, executed = run_resolution(monkeypatch, [ExecutionStatus.EMPTY], reviewer_enabled=False)
    assert result.resolution is resolve.Resolution.EMPTY
    assert result.attempts == 0
    assert executed == ["SELECT name FROM orders"]


def test_initial_query_error_still_has_two_fixer_attempts(monkeypatch):
    result, executed = run_resolution(
        monkeypatch,
        [ExecutionStatus.EXECUTION_ERROR] * 3,
        fixes=["SELECT name FROM orders WHERE id = 1", "SELECT name FROM orders WHERE id = 2"],
    )
    assert result.resolution is resolve.Resolution.EXHAUSTED
    assert result.attempts == 2
    assert len(executed) == 3
