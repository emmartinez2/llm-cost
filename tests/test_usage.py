import pytest

from llm_cost.pricing import default_pricing
from llm_cost.usage import aggregate, load_usage, parse_record


def test_parse_record_anthropic_shape():
    record = parse_record(
        {
            "model": "claude-opus-5",
            "usage": {
                "input_tokens": 18400,
                "output_tokens": 2100,
                "cache_read_input_tokens": 52000,
                "cache_creation_input_tokens": 9000,
            },
        }
    )
    assert record.model == "claude-opus-5"
    assert record.input_tokens == 18400
    assert record.output_tokens == 2100
    assert record.cached_input_tokens == 52000
    assert record.cache_write_tokens == 9000


def test_parse_record_openai_shape_subtracts_cached_from_prompt():
    record = parse_record(
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 31000,
                "completion_tokens": 420,
                "prompt_tokens_details": {"cached_tokens": 24000},
            },
        }
    )
    assert record.input_tokens == 7000
    assert record.cached_input_tokens == 24000
    assert record.cache_write_tokens == 0


def test_parse_record_openai_shape_without_cache_details():
    record = parse_record(
        {"model": "gpt-4o", "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    )
    assert record.input_tokens == 100
    assert record.cached_input_tokens == 0


def test_parse_record_keeps_original_fields_for_grouping():
    record = parse_record(
        {
            "model": "gpt-4o",
            "team": "search",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }
    )
    assert record.group_key("team") == "search"
    assert record.group_key("model") == "gpt-4o"
    assert record.group_key("missing") == "(none)"


def test_parse_record_groups_dates_by_calendar_day():
    record = parse_record(
        {
            "model": "gpt-4o",
            "date": "2026-06-01T09:12:00Z",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    assert record.group_key("date") == "2026-06-01"


def test_parse_record_requires_model():
    with pytest.raises(ValueError):
        parse_record({"usage": {"input_tokens": 1}})


def test_parse_record_requires_usage_object():
    with pytest.raises(ValueError):
        parse_record({"model": "m"})


def test_parse_record_rejects_unrecognised_usage_shape():
    with pytest.raises(ValueError):
        parse_record({"model": "m", "usage": {"tokens": 1}})


def test_parse_record_rejects_cached_exceeding_prompt():
    with pytest.raises(ValueError):
        parse_record(
            {
                "model": "m",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "prompt_tokens_details": {"cached_tokens": 20},
                },
            }
        )


def test_parse_record_rejects_non_object():
    with pytest.raises(ValueError):
        parse_record(["not", "a", "dict"])


def test_load_usage_skips_blank_lines():
    text = '\n{"model": "gpt-4o", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}\n\n'
    records, problems = load_usage(text)
    assert len(records) == 1
    assert problems == []


def test_load_usage_collects_malformed_lines_by_default():
    text = "not json\n" + '{"model": "gpt-4o", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}'
    records, problems = load_usage(text)
    assert len(records) == 1
    assert len(problems) == 1
    assert problems[0].startswith("line 1:")


def test_load_usage_strict_raises_on_first_bad_line():
    text = "not json\nalso not json"
    with pytest.raises(ValueError) as excinfo:
        load_usage(text, strict=True)
    assert "line 1" in str(excinfo.value)


def test_aggregate_groups_by_model_and_skips_unknown():
    records, _ = load_usage(
        "\n".join(
            [
                '{"model": "claude-opus-5", "usage": {"input_tokens": 1000, "output_tokens": 100}}',
                '{"model": "claude-opus-5", "usage": {"input_tokens": 2000, "output_tokens": 200}}',
                '{"model": "internal-router-v3", "usage": {"input_tokens": 500, "output_tokens": 50}}',
            ]
        )
    )
    groups, unknown = aggregate(records, default_pricing(), group_by="model")
    assert unknown == {"internal-router-v3"}
    assert set(groups) == {"claude-opus-5"}
    combined = groups["claude-opus-5"]
    assert combined.calls == 2
    assert combined.input_tokens == 3000
    assert combined.output_tokens == 300


def test_aggregate_by_team_combines_different_models():
    records, _ = load_usage(
        "\n".join(
            [
                '{"model": "claude-opus-5", "team": "agents", "usage": {"input_tokens": 1000, "output_tokens": 100}}',
                '{"model": "gpt-4o-mini", "team": "agents", "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}',
            ]
        )
    )
    groups, unknown = aggregate(records, default_pricing(), group_by="team")
    assert unknown == set()
    assert groups["agents"].calls == 2
    assert groups["agents"].model == "(mixed)"
