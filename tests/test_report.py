import pytest

from llm_cost.pricing import default_pricing, parse_pricing
from llm_cost.report import build_report, compare_models
from llm_cost.usage import load_usage


def test_build_report_sorts_groups_by_key():
    records, problems = load_usage(
        "\n".join(
            [
                '{"model": "gpt-4o-mini", "usage": {"prompt_tokens": 100, "completion_tokens": 10}}',
                '{"model": "claude-opus-5", "usage": {"input_tokens": 100, "output_tokens": 10}}',
            ]
        )
    )
    report = build_report(records, default_pricing(), group_by="model", problems=problems)
    assert [group.key for group in report.groups] == ["claude-opus-5", "gpt-4o-mini"]
    assert report.total_calls == 2
    assert report.total_cost == pytest.approx(
        sum(group.total_cost for group in report.groups)
    )


def test_build_report_carries_problems_and_unknown_models():
    records, problems = load_usage(
        "\n".join(
            [
                "not json",
                '{"model": "internal-router-v3", "usage": {"input_tokens": 1, "output_tokens": 1}}',
            ]
        )
    )
    report = build_report(records, default_pricing(), problems=problems)
    assert report.unknown_models == {"internal-router-v3"}
    assert len(report.problems) == 1
    assert report.groups == []


def test_report_to_dict_shape():
    records, problems = load_usage(
        '{"model": "gpt-4o-mini", "usage": {"prompt_tokens": 100, "completion_tokens": 10}}'
    )
    report = build_report(records, default_pricing(), problems=problems)
    as_dict = report.to_dict()
    assert as_dict["group_by"] == "model"
    assert as_dict["total_calls"] == 1
    assert as_dict["unknown_models"] == []
    assert as_dict["problems"] == []
    assert as_dict["groups"][0]["group"] == "gpt-4o-mini"


def test_compare_models_ranks_cheapest_first():
    table = default_pricing()
    ranked = compare_models(table, input_tokens=10000, output_tokens=1000, provider="anthropic")
    costs = [cost.total_cost for _, cost, _ in ranked]
    assert costs == sorted(costs)
    assert ranked[0][2] == pytest.approx(1.0)
    assert all(price.provider == "anthropic" for price, _, _ in ranked)


def test_compare_models_ties_break_by_name():
    table = parse_pricing(
        {
            "models": {
                "b-model": {"input": 1.0, "output": 1.0},
                "a-model": {"input": 1.0, "output": 1.0},
            },
            "replace": True,
        }
    )
    ranked = compare_models(table, input_tokens=1000, output_tokens=1000)
    assert [price.model for price, _, _ in ranked] == ["a-model", "b-model"]


def test_compare_models_respects_shortlist():
    table = default_pricing()
    ranked = compare_models(
        table,
        input_tokens=1000,
        output_tokens=1000,
        models=["claude-opus-5", "claude-haiku-4-5"],
    )
    assert {price.model for price, _, _ in ranked} == {"claude-opus-5", "claude-haiku-4-5"}


def test_compare_models_no_match_returns_empty():
    table = default_pricing()
    ranked = compare_models(table, input_tokens=1000, output_tokens=100, provider="no-such-provider")
    assert ranked == []
