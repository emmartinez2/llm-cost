"""Parsing usage logs and pricing them into grouped totals.

A log line can come from either provider's SDK. OpenAI's ``prompt_tokens``
includes the cached prefix; Anthropic's ``input_tokens`` does not. Both are
normalised here, once, so the rest of the package only ever sees the
exclusive form that :func:`llm_cost.estimate.estimate_cost` expects.
"""

import json

from .estimate import CostBreakdown, estimate_cost
from .pricing import UnknownModelError

__all__ = ["UsageRecord", "aggregate", "load_usage", "parse_record"]


class UsageRecord(object):
    """One priced API call, plus the raw JSON fields it was parsed from.

    ``fields`` keeps the original record around so ``--group-by`` can key on
    any top-level field (``team``, ``date``, ...), not just the ones this
    module knows how to price.
    """

    __slots__ = (
        "model",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "fields",
    )

    def __init__(
        self,
        model,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_write_tokens,
        fields,
    ):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.fields = fields

    def group_key(self, group_by):
        if group_by == "model":
            return self.model
        value = self.fields.get(group_by)
        if value is None:
            return "(none)"
        value = str(value)
        if group_by == "date" and len(value) >= 10:
            # Timestamps are full ISO 8601; group by the calendar day.
            return value[:10]
        return value


def parse_record(obj):
    """Turn one decoded JSON object into a :class:`UsageRecord`.

    Raises :class:`ValueError` if the record is missing a model name or its
    ``usage`` object doesn't match either provider's shape.
    """
    if not isinstance(obj, dict):
        raise ValueError("record is not a JSON object")
    model = obj.get("model")
    if not model:
        raise ValueError("record has no 'model' field")
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("record has no 'usage' object")

    if "input_tokens" in usage or "output_tokens" in usage:
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cached_input_tokens = int(usage.get("cache_read_input_tokens", 0))
        cache_write_tokens = int(usage.get("cache_creation_input_tokens", 0))
    elif "prompt_tokens" in usage or "completion_tokens" in usage:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        details = usage.get("prompt_tokens_details") or {}
        cached_input_tokens = int(details.get("cached_tokens", 0))
        input_tokens = prompt_tokens - cached_input_tokens
        if input_tokens < 0:
            raise ValueError("cached_tokens exceeds prompt_tokens")
        cache_write_tokens = 0
    else:
        raise ValueError("usage object has no recognised token fields")

    return UsageRecord(
        model=str(model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        fields=obj,
    )


def load_usage(text, strict=False):
    """Parse JSONL *text* into ``(records, problems)``.

    Blank lines are skipped. Malformed lines are collected in *problems* as
    ``"line N: message"`` strings and otherwise ignored, unless *strict* is
    set, in which case the first bad line raises :class:`ValueError`.
    """
    records = []
    problems = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            record = parse_record(obj)
        except ValueError as error:
            message = "line %d: %s" % (lineno, error)
            if strict:
                raise ValueError(message)
            problems.append(message)
            continue
        records.append(record)
    return records, problems


def _combine(a, b):
    return CostBreakdown(
        model=a.model if a.model == b.model else "(mixed)",
        calls=a.calls + b.calls,
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cached_input_tokens=a.cached_input_tokens + b.cached_input_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        input_cost=a.input_cost + b.input_cost,
        output_cost=a.output_cost + b.output_cost,
        cached_input_cost=a.cached_input_cost + b.cached_input_cost,
        cache_write_cost=a.cache_write_cost + b.cache_write_cost,
    )


def aggregate(records, price_table, group_by="model"):
    """Price every record and sum them by ``group_key(group_by)``.

    A group can span several models with different prices (grouping by
    ``team`` is the common case) -- each record is priced against its own
    model before the totals are summed, so that's fine. Records naming a
    model with no price are skipped and their names collected instead of
    being silently priced at zero.

    Returns ``(groups, unknown_models)``: a dict of group key to
    :class:`~llm_cost.estimate.CostBreakdown`, and a set of unpriced model
    names.
    """
    groups = {}
    unknown_models = set()
    for record in records:
        try:
            price = price_table.resolve(record.model)
        except UnknownModelError:
            unknown_models.add(record.model)
            continue
        cost = estimate_cost(
            price,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_input_tokens,
            cache_write_tokens=record.cache_write_tokens,
        )
        key = record.group_key(group_by)
        groups[key] = _combine(groups[key], cost) if key in groups else cost
    return groups, unknown_models
