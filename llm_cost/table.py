"""Number formatting and the fixed-width table renderer shared by the CLI."""

__all__ = ["format_int", "format_money", "render_table"]


def format_int(value):
    return "{:,}".format(int(value))


def format_money(value, decimals=4):
    return "$%.*f" % (decimals, value)


def render_table(headers, rows):
    """Render *headers* and *rows* as a plain-text table.

    The first column is left-aligned (it holds names), the rest are
    right-aligned (they hold numbers). Cells are already-formatted strings;
    this function only measures and pads them.
    """
    headers = [str(cell) for cell in headers]
    body = [[str(cell) for cell in row] for row in rows]

    widths = [len(cell) for cell in headers]
    for row in body:
        for index, cell in enumerate(row):
            if len(cell) > widths[index]:
                widths[index] = len(cell)

    def pad(cell, index):
        width = widths[index]
        return cell.ljust(width) if index == 0 else cell.rjust(width)

    lines = ["  ".join(pad(cell, i) for i, cell in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in body:
        lines.append("  ".join(pad(cell, i) for i, cell in enumerate(row)))
    return "\n".join(lines)
