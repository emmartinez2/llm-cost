"""Turning priced usage into a report, and ranking models against each other."""

from .estimate import estimate_cost
from .usage import aggregate

__all__ = ["GroupSummary", "Report", "build_report", "compare_models"]


class GroupSummary(object):
    """Totals for one ``--group-by`` bucket."""

    __slots__ = ("key", "cost")

    def __init__(self, key, cost):
        self.key = key
        self.cost = cost

    @property
    def calls(self):
        return self.cost.calls

    @property
    def input_tokens(self):
        return self.cost.input_tokens

    @property
    def cached_input_tokens(self):
        return self.cost.cached_input_tokens

    @property
    def output_tokens(self):
        return self.cost.output_tokens

    @property
    def total_cost(self):
        return self.cost.total_cost

    def to_dict(self):
        result = self.cost.to_dict()
        result["group"] = self.key
        return result


class Report(object):
    """A priced, grouped usage log."""

    __slots__ = ("group_by", "groups", "unknown_models", "problems")

    def __init__(self, group_by, groups, unknown_models, problems):
        self.group_by = group_by
        self.groups = groups
        self.unknown_models = unknown_models
        self.problems = problems

    @property
    def total_calls(self):
        return sum(group.calls for group in self.groups)

    @property
    def total_cost(self):
        return sum(group.total_cost for group in self.groups)

    def to_dict(self):
        return {
            "group_by": self.group_by,
            "groups": [group.to_dict() for group in self.groups],
            "total_calls": self.total_calls,
            "total_cost": round(self.total_cost, 6),
            "unknown_models": sorted(self.unknown_models),
            "problems": list(self.problems),
        }


def build_report(records, price_table, group_by="model", problems=None):
    """Aggregate *records* by *group_by* and price them against *price_table*."""
    groups, unknown_models = aggregate(records, price_table, group_by=group_by)
    summaries = [GroupSummary(key, cost) for key, cost in groups.items()]
    summaries.sort(key=lambda group: group.key)
    return Report(group_by, summaries, unknown_models, list(problems or []))


def compare_models(price_table, input_tokens, output_tokens, calls=1, provider=None, models=None):
    """Price one call shape against every model, cheapest first.

    Returns a list of ``(price, cost, multiple)`` triples, where *multiple*
    is the cost relative to the cheapest matching model. Ties sort by model
    name so the ranking is stable.
    """
    names = list(models) if models else price_table.models()

    priced = []
    for name in names:
        price = price_table.resolve(name)
        if provider and price.provider != provider:
            continue
        cost = estimate_cost(
            price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            calls=calls,
        )
        priced.append((price, cost))

    priced.sort(key=lambda pair: (pair[1].total_cost, pair[0].model))
    if not priced:
        return []

    cheapest = priced[0][1].total_cost
    return [
        (price, cost, cost.total_cost / cheapest if cheapest else 0.0)
        for price, cost in priced
    ]
