import pytest

from llm_cost.pricing import (
    ModelPrice,
    PricingTable,
    UnknownModelError,
    default_pricing,
    parse_pricing,
)


def test_default_pricing_has_builtin_models():
    table = default_pricing()
    assert table.source == "built-in"
    assert "claude-opus-5" in table
    assert len(table) == len(table.models())


def test_resolve_is_case_insensitive():
    table = default_pricing()
    exact = table.resolve("claude-opus-5")
    assert table.resolve("Claude-Opus-5") is exact


def test_resolve_strips_provider_prefix():
    table = default_pricing()
    assert table.resolve("anthropic.claude-opus-5") is table.resolve("claude-opus-5")
    assert table.resolve("anthropic/claude-opus-5") is table.resolve("claude-opus-5")


def test_resolve_tolerates_date_suffix():
    table = default_pricing()
    # A dated snapshot name should fall back to the longest known prefix.
    assert table.resolve("gpt-4o-2026-01-31") is table.resolve("gpt-4o")


def test_resolve_prefers_longer_prefix_match():
    table = default_pricing()
    resolved = table.resolve("claude-opus-4-8-20260101")
    assert resolved.model == "claude-opus-4-8"


def test_resolve_unknown_model_raises_with_readable_message():
    table = default_pricing()
    with pytest.raises(UnknownModelError) as excinfo:
        table.resolve("internal-router-v3")
    assert "internal-router-v3" in str(excinfo.value)
    # UnknownModelError overrides __str__ so no stray repr quoting leaks in.
    assert not str(excinfo.value).startswith("'")


def test_resolve_empty_name_raises():
    table = default_pricing()
    with pytest.raises(UnknownModelError):
        table.resolve("")


def test_model_price_defaults_cache_prices_to_input():
    price = ModelPrice("m", input=2.0, output=4.0)
    assert price.cached_input == 2.0
    assert price.cache_write == 2.0


def test_model_price_rejects_negative_prices():
    with pytest.raises(ValueError):
        ModelPrice("m", input=-1.0, output=4.0)


def test_parse_pricing_merges_onto_builtin_by_default():
    table = parse_pricing({"models": {"my-model": {"input": 1.0, "output": 2.0}}})
    assert "my-model" in table
    assert "claude-opus-5" in table


def test_parse_pricing_replace_drops_builtin():
    table = parse_pricing(
        {"models": {"my-model": {"input": 1.0, "output": 2.0}}, "replace": True}
    )
    assert "my-model" in table
    assert "claude-opus-5" not in table


def test_parse_pricing_flat_shape():
    table = parse_pricing({"my-model": {"input": 1.0, "output": 2.0}})
    assert table.as_of == "unspecified"
    assert "my-model" in table


def test_parse_pricing_missing_required_field_raises():
    with pytest.raises(ValueError):
        parse_pricing({"models": {"my-model": {"input": 1.0}}})


def test_parse_pricing_non_numeric_field_raises():
    with pytest.raises(ValueError):
        parse_pricing({"models": {"my-model": {"input": 1.0, "output": "cheap"}}})


def test_parse_pricing_not_an_object_raises():
    with pytest.raises(ValueError):
        parse_pricing(["not", "a", "dict"])


def test_parse_pricing_empty_models_raises():
    with pytest.raises(ValueError):
        parse_pricing({"models": {}})


def test_load_pricing_missing_file_raises(tmp_path):
    from llm_cost.pricing import load_pricing

    with pytest.raises(ValueError):
        load_pricing(str(tmp_path / "missing.json"))


def test_load_pricing_reads_file(tmp_path):
    from llm_cost.pricing import load_pricing

    path = tmp_path / "prices.json"
    path.write_text('{"as_of": "2026-07-01", "models": {"m": {"input": 1, "output": 2}}}')
    table = load_pricing(str(path))
    assert table.as_of == "2026-07-01"
    assert table.source == str(path)
    assert table.resolve("m").output == 2.0


def test_load_pricing_invalid_json_raises(tmp_path):
    from llm_cost.pricing import load_pricing

    path = tmp_path / "prices.json"
    path.write_text("{not json")
    with pytest.raises(ValueError):
        load_pricing(str(path))


def test_pricing_table_models_sorted():
    table = PricingTable({"b": ModelPrice("b", 1, 1), "a": ModelPrice("a", 1, 1)})
    assert table.models() == ["a", "b"]
