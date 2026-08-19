"""The ``llm-cost`` command-line entry point."""

import argparse
import json
import sys

from .estimate import estimate_cost
from .pricing import UnknownModelError, default_pricing, load_pricing
from .report import build_report, compare_models
from .table import format_int, format_money, render_table
from .usage import load_usage

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_UNKNOWN_MODEL = 3

__all__ = ["main"]


def _common_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pricing", metavar="FILE", help="JSON file overriding the built-in price table"
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON instead of a table"
    )
    return parser


def _build_parser():
    common = _common_parser()
    parser = argparse.ArgumentParser(prog="llm-cost", parents=[common])
    subparsers = parser.add_subparsers(dest="command")

    estimate_parser = subparsers.add_parser(
        "estimate", parents=[common], help="price one call, or a batch of identical calls"
    )
    estimate_parser.add_argument("--model", required=True)
    estimate_parser.add_argument("--input", type=int, default=0, dest="input_tokens")
    estimate_parser.add_argument("--output", type=int, default=0, dest="output_tokens")
    estimate_parser.add_argument("--cached", type=int, default=0, dest="cached_input_tokens")
    estimate_parser.add_argument("--cache-write", type=int, default=0, dest="cache_write_tokens")
    estimate_parser.add_argument("--calls", type=int, default=1)

    report_parser = subparsers.add_parser(
        "report", parents=[common], help="cost a JSONL usage log"
    )
    report_parser.add_argument("logfile")
    report_parser.add_argument("--group-by", default="model")
    report_parser.add_argument("--strict", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare", parents=[common], help="rank models on the same call shape"
    )
    compare_parser.add_argument("--input", type=int, default=0, dest="input_tokens")
    compare_parser.add_argument("--output", type=int, default=0, dest="output_tokens")
    compare_parser.add_argument("--calls", type=int, default=1)
    compare_parser.add_argument("--provider")
    compare_parser.add_argument("--models", help="comma-separated shortlist of model names")

    subparsers.add_parser("models", parents=[common], help="list the price table")

    return parser


def _load_pricing_table(path):
    return default_pricing() if path is None else load_pricing(path)


def _cmd_estimate(args, table):
    try:
        price = table.resolve(args.model)
    except UnknownModelError as error:
        print(str(error), file=sys.stderr)
        return EXIT_UNKNOWN_MODEL

    try:
        cost = estimate_cost(
            price,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            cached_input_tokens=args.cached_input_tokens,
            cache_write_tokens=args.cache_write_tokens,
            calls=args.calls,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_INPUT

    if args.json:
        print(json.dumps(cost.to_dict()))
        return EXIT_OK

    print(
        "%s  (%d call%s, prices as of %s)"
        % (price.model, args.calls, "" if args.calls == 1 else "s", table.as_of)
    )
    print("")
    rows = [
        ("input", format_int(cost.input_tokens), "%.2f" % price.input, format_money(cost.input_cost)),
        (
            "cached input",
            format_int(cost.cached_input_tokens),
            "%.2f" % price.cached_input,
            format_money(cost.cached_input_cost),
        ),
        (
            "cache write",
            format_int(cost.cache_write_tokens),
            "%.2f" % price.cache_write,
            format_money(cost.cache_write_cost),
        ),
        ("output", format_int(cost.output_tokens), "%.2f" % price.output, format_money(cost.output_cost)),
        ("total", format_int(cost.total_tokens), "", format_money(cost.total_cost)),
    ]
    print(render_table(("item", "tokens", "$/1M", "cost"), rows))
    print("")
    print("cost per call: %s" % format_money(cost.cost_per_call))
    return EXIT_OK


def _cmd_report(args, table):
    try:
        with open(args.logfile, "r", encoding="utf-8") as handle:
            text = handle.read()
    except EnvironmentError as error:
        print("cannot read %s: %s" % (args.logfile, error), file=sys.stderr)
        return EXIT_BAD_INPUT

    try:
        records, problems = load_usage(text, strict=args.strict)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_INPUT

    report = build_report(records, table, group_by=args.group_by, problems=problems)

    if args.json:
        print(json.dumps(report.to_dict()))
        return EXIT_OK

    headers = (args.group_by, "calls", "input", "cached", "output", "cost", "$/call")
    rows = []
    for group in report.groups:
        per_call = group.total_cost / group.calls if group.calls else 0.0
        rows.append(
            (
                group.key,
                format_int(group.calls),
                format_int(group.input_tokens),
                format_int(group.cached_input_tokens),
                format_int(group.output_tokens),
                format_money(group.total_cost),
                format_money(per_call),
            )
        )
    rows.append(
        (
            "TOTAL",
            format_int(report.total_calls),
            format_int(sum(g.input_tokens for g in report.groups)),
            format_int(sum(g.cached_input_tokens for g in report.groups)),
            format_int(sum(g.output_tokens for g in report.groups)),
            format_money(report.total_cost),
            "",
        )
    )
    print(render_table(headers, rows))

    for problem in report.problems:
        print("skipped malformed record: %s" % problem, file=sys.stderr)
    if report.unknown_models:
        print(
            "skipped %d record(s) with no price: %s"
            % (len(report.unknown_models), ", ".join(sorted(report.unknown_models))),
            file=sys.stderr,
        )
    return EXIT_OK


def _cmd_compare(args, table):
    models = None
    if args.models:
        models = [name.strip() for name in args.models.split(",") if name.strip()]

    try:
        ranked = compare_models(
            table,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            calls=args.calls,
            provider=args.provider,
            models=models,
        )
    except UnknownModelError as error:
        print(str(error), file=sys.stderr)
        return EXIT_UNKNOWN_MODEL

    if not ranked:
        print("no models matched", file=sys.stderr)
        return EXIT_UNKNOWN_MODEL

    if args.json:
        print(
            json.dumps(
                [
                    dict(cost.to_dict(), provider=price.provider, vs_cheapest=round(multiple, 4))
                    for price, cost, multiple in ranked
                ]
            )
        )
        return EXIT_OK

    print(
        "%s in + %s out, %d call%s, prices as of %s"
        % (
            format_int(args.input_tokens),
            format_int(args.output_tokens),
            args.calls,
            "" if args.calls == 1 else "s",
            table.as_of,
        )
    )
    print("")
    headers = ("model", "provider", "$/1M in", "$/1M out", "cost", "vs cheapest")
    rows = []
    for price, cost, multiple in ranked:
        rows.append(
            (
                price.model,
                price.provider,
                "%.2f" % price.input,
                "%.2f" % price.output,
                format_money(cost.total_cost),
                "%.1fx" % multiple,
            )
        )
    print(render_table(headers, rows))
    return EXIT_OK


def _cmd_models(args, table):
    if args.json:
        print(
            json.dumps(
                {
                    "as_of": table.as_of,
                    "source": table.source,
                    "models": dict((name, table.prices[name].to_dict()) for name in table.models()),
                }
            )
        )
        return EXIT_OK

    print(
        "%d models, USD per 1M tokens, as of %s (source: %s)"
        % (len(table), table.as_of, table.source)
    )
    print("")
    headers = ("model", "provider", "input", "output", "cached_input", "cache_write")
    rows = []
    for name in table.models():
        price = table.prices[name]
        rows.append(
            (
                name,
                price.provider,
                "%.2f" % price.input,
                "%.2f" % price.output,
                "%.2f" % price.cached_input,
                "%.2f" % price.cache_write,
            )
        )
    print(render_table(headers, rows))
    return EXIT_OK


_HANDLERS = {
    "estimate": _cmd_estimate,
    "report": _cmd_report,
    "compare": _cmd_compare,
    "models": _cmd_models,
}


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return EXIT_BAD_INPUT

    try:
        table = _load_pricing_table(args.pricing)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_INPUT

    return _HANDLERS[args.command](args, table)


if __name__ == "__main__":
    sys.exit(main())
