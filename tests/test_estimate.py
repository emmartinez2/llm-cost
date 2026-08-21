import pytest

from llm_cost.estimate import estimate_cost
from llm_cost.pricing import ModelPrice


PRICE = ModelPrice(
    "test-model", input=5.0, output=25.0, cached_input=0.5, cache_write=6.25
)


def test_estimate_cost_basic_breakdown():
    cost = estimate_cost(PRICE, input_tokens=12000, output_tokens=800)
    assert cost.input_cost == pytest.approx(0.06)
    assert cost.output_cost == pytest.approx(0.02)
    assert cost.cached_input_cost == 0.0
    assert cost.cache_write_cost == 0.0
    assert cost.total_cost == pytest.approx(0.08)
    assert cost.cost_per_call == pytest.approx(0.08)
    assert cost.total_tokens == 12800


def test_estimate_cost_prices_all_four_token_classes():
    cost = estimate_cost(
        PRICE,
        input_tokens=1000000,
        output_tokens=1000000,
        cached_input_tokens=1000000,
        cache_write_tokens=1000000,
    )
    assert cost.input_cost == pytest.approx(5.0)
    assert cost.output_cost == pytest.approx(25.0)
    assert cost.cached_input_cost == pytest.approx(0.5)
    assert cost.cache_write_cost == pytest.approx(6.25)
    assert cost.total_cost == pytest.approx(36.75)


def test_estimate_cost_scales_with_calls():
    one = estimate_cost(PRICE, input_tokens=1000, output_tokens=100, calls=1)
    fifty = estimate_cost(PRICE, input_tokens=1000, output_tokens=100, calls=50)
    assert fifty.total_cost == pytest.approx(one.total_cost * 50)
    assert fifty.input_tokens == one.input_tokens * 50
    assert fifty.cost_per_call == pytest.approx(one.total_cost)


def test_estimate_cost_zero_calls_gives_zero_cost_per_call():
    cost = estimate_cost(PRICE, input_tokens=1000, output_tokens=100, calls=0)
    assert cost.total_cost == 0.0
    assert cost.cost_per_call == 0.0


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_cost(PRICE, input_tokens=-1)


def test_estimate_cost_rejects_non_integer_tokens():
    with pytest.raises(ValueError):
        estimate_cost(PRICE, input_tokens=1.5)


def test_estimate_cost_defaults_are_zero():
    cost = estimate_cost(PRICE)
    assert cost.total_cost == 0.0
    assert cost.total_tokens == 0


def test_cost_breakdown_to_dict_rounds_to_six_places():
    cost = estimate_cost(PRICE, input_tokens=1, output_tokens=1)
    as_dict = cost.to_dict()
    assert as_dict["model"] == "test-model"
    assert as_dict["total_cost"] == round(cost.total_cost, 6)
    assert set(as_dict) == {
        "model",
        "calls",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "input_cost",
        "output_cost",
        "cached_input_cost",
        "cache_write_cost",
        "total_cost",
        "cost_per_call",
    }
