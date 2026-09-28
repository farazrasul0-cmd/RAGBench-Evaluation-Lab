"""Terminal output and tabular formatting utilities for RAGBench CLI."""

from typing import Any


def render_table(
    headers: list[str],
    rows: list[list[Any]],
    title: str | None = None,
) -> str:
    """Render a clean ASCII table with headers, borders, and aligned columns."""
    if not headers and not rows:
        return ""

    str_rows = [[str(cell) for cell in row] for row in rows]
    num_cols = len(headers)

    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for idx in range(min(len(row), num_cols)):
            col_widths[idx] = max(col_widths[idx], len(row[idx]))

    sep_line = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    lines = []
    if title:
        title_width = sum(col_widths) + 3 * (num_cols - 1) + 2
        lines.append("+" + "-" * (title_width) + "+")
        lines.append(f"| {title.center(title_width - 2)} |")
    lines.append(sep_line)
    lines.append(header_line)
    lines.append(sep_line)

    for row in str_rows:
        row_cells = []
        for i in range(num_cols):
            val = row[i] if i < len(row) else ""
            row_cells.append(val.ljust(col_widths[i]))
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append(sep_line)
    return "\n".join(lines)


def format_delta(val1: float, val2: float, higher_is_better: bool = True) -> str:
    """Format comparative delta between two numeric metrics."""
    delta = val2 - val1
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}"
