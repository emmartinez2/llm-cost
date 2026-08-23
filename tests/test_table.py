import os

from llm_cost.table import format_int, format_money, render_table

README_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")


def test_format_int_adds_thousands_separators():
    assert format_int(12000) == "12,000"
    assert format_int(0) == "0"
    assert format_int(800) == "800"


def test_format_int_coerces_floats():
    assert format_int(12000.0) == "12,000"


def test_format_money_default_decimals():
    assert format_money(0.08) == "$0.0800"
    assert format_money(0) == "$0.0000"


def test_format_money_custom_decimals():
    assert format_money(1234.5, decimals=2) == "$1234.50"


def test_render_table_pads_first_column_left_and_rest_right():
    out = render_table(("name", "count"), [("a", "1"), ("bb", "22")])
    lines = out.splitlines()
    assert lines[0] == "  ".join(["name", "count"])
    assert lines[1] == "  ".join(["----", "-----"])
    assert lines[2] == "  ".join(["a".ljust(4), "1".rjust(5)])
    assert lines[3] == "  ".join(["bb".ljust(4), "22".rjust(5)])


def test_render_table_widens_columns_to_the_longest_cell():
    out = render_table(("item",), [("cached input",), ("x",)])
    lines = out.splitlines()
    width = len("cached input")
    assert lines[0] == "item".ljust(width)
    assert lines[2] == "cached input"
    assert lines[3] == "x".ljust(width)


def test_render_table_with_no_rows_is_header_and_separator_only():
    out = render_table(("a", "bb"), [])
    assert out == "a  bb\n-  --"


def test_render_table_matches_readme_estimate_example_byte_for_byte():
    # Reproduces the exact rows llm_cost.cli._cmd_estimate builds for
    # `llm-cost estimate --model claude-opus-5 --input 12000 --output 800`,
    # checked against the README's --input 12000 --output 800 console block.
    headers = ("item", "tokens", "$/1M", "cost")
    rows = [
        ("input", "12,000", "5.00", "$0.0600"),
        ("cached input", "0", "0.50", "$0.0000"),
        ("cache write", "0", "6.25", "$0.0000"),
        ("output", "800", "25.00", "$0.0200"),
        ("total", "12,800", "", "$0.0800"),
    ]

    columns = [headers] + rows
    widths = [max(len(row[i]) for row in columns) for i in range(len(headers))]

    def line(cells, fill=None):
        padded = []
        for index, cell in enumerate(cells):
            cell = fill * widths[index] if fill else cell
            padded.append(cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index]))
        return "  ".join(padded)

    expected = "\n".join(
        [line(headers), line(headers, fill="-")] + [line(row) for row in rows]
    )
    assert render_table(headers, rows) == expected

    with open(README_PATH, encoding="utf-8") as handle:
        readme = handle.read()
    assert expected in readme
