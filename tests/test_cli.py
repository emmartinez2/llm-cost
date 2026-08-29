import json

import pytest

from llm_cost.cli import EXIT_BAD_INPUT, EXIT_OK, EXIT_UNKNOWN_MODEL, main
from llm_cost.pricing import default_pricing
from llm_cost.table import render_table


def test_estimate_json_shape(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "1000", "--output", "100", "--json"])
    assert code == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "claude-opus-5"
    assert out["input_tokens"] == 1000
    assert out["output_tokens"] == 100
    assert out["total_cost"] == pytest.approx(1000 * 5.00 / 1e6 + 100 * 25.00 / 1e6)


def test_estimate_table_output_has_expected_lines(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "1000", "--output", "100"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "claude-opus-5  (1 call, prices as of" in out
    assert "cost per call:" in out


def test_estimate_unknown_model_exits_with_unknown_model_code(capsys):
    code = main(["estimate", "--model", "no-such-model", "--input", "10"])
    assert code == EXIT_UNKNOWN_MODEL
    assert "no price for model" in capsys.readouterr().err


def test_estimate_negative_tokens_is_bad_input(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "-5"])
    assert code == EXIT_BAD_INPUT
    assert "must not be negative" in capsys.readouterr().err


def test_report_missing_file_is_bad_input(capsys):
    code = main(["report", "/no/such/file.jsonl"])
    assert code == EXIT_BAD_INPUT
    assert "cannot read" in capsys.readouterr().err


def test_report_json_shape(tmp_path, capsys):
    logfile = tmp_path / "usage.jsonl"
    logfile.write_text(
        '{"model": "claude-opus-5", "usage": {"input_tokens": 1000, "output_tokens": 100}}\n'
        '{"model": "internal-router-v3", "usage": {"input_tokens": 1, "output_tokens": 1}}\n',
        encoding="utf-8",
    )
    code = main(["report", str(logfile), "--json"])
    assert code == EXIT_OK
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["total_calls"] == 1
    assert out["unknown_models"] == ["internal-router-v3"]
    assert "skipped 1 record(s) with no price" in captured.err


def test_report_table_output(tmp_path, capsys):
    logfile = tmp_path / "usage.jsonl"
    logfile.write_text(
        '{"model": "claude-opus-5", "usage": {"input_tokens": 1000, "output_tokens": 100}}\n',
        encoding="utf-8",
    )
    code = main(["report", str(logfile)])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "claude-opus-5" in out


def test_report_strict_stops_on_malformed_line(tmp_path, capsys):
    logfile = tmp_path / "usage.jsonl"
    logfile.write_text("not json\n", encoding="utf-8")
    code = main(["report", str(logfile), "--strict"])
    assert code == EXIT_BAD_INPUT
    assert "line 1" in capsys.readouterr().err


def test_compare_json_shape(capsys):
    code = main(["compare", "--input", "1000", "--output", "100", "--provider", "anthropic", "--json"])
    assert code == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert all(entry["provider"] == "anthropic" for entry in out)
    assert out[0]["vs_cheapest"] == pytest.approx(1.0)


def test_compare_no_match_is_unknown_model_exit(capsys):
    code = main(["compare", "--provider", "no-such-provider"])
    assert code == EXIT_UNKNOWN_MODEL
    assert "no models matched" in capsys.readouterr().err


def test_models_json_shape(capsys):
    code = main(["models", "--json"])
    assert code == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert "claude-opus-5" in out["models"]
    assert out["source"] == "built-in"


def test_models_table_output_matches_price_table(capsys):
    code = main(["models"])
    assert code == EXIT_OK
    out = capsys.readouterr().out

    table = default_pricing()
    headers = ("model", "provider", "input", "output", "cached_input", "cache_write")
    rows = [
        (
            name,
            table.prices[name].provider,
            "%.2f" % table.prices[name].input,
            "%.2f" % table.prices[name].output,
            "%.2f" % table.prices[name].cached_input,
            "%.2f" % table.prices[name].cache_write,
        )
        for name in table.models()
    ]
    intro = "%d models, USD per 1M tokens, as of %s (source: %s)" % (
        len(table),
        table.as_of,
        table.source,
    )
    expected = "\n".join([intro, "", render_table(headers, rows)]) + "\n"
    assert out == expected


def test_no_command_prints_help_and_exits_bad_input(capsys):
    code = main([])
    assert code == EXIT_BAD_INPUT
    assert "usage:" in capsys.readouterr().out


def test_bad_pricing_override_is_bad_input(tmp_path, capsys):
    pricing_file = tmp_path / "prices.json"
    pricing_file.write_text("not json", encoding="utf-8")
    code = main(["--pricing", str(pricing_file), "models"])
    assert code == EXIT_BAD_INPUT
    assert "not valid JSON" in capsys.readouterr().err
