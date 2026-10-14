"""Deterministic serializers for every derived string in the schema.

All functions are pure. They are the *only* place that produces a derived
string, so the exporter and the frontend preview are guaranteed identical.
"""
from __future__ import annotations

from app.schemas.doc_v2 import (
    CheckboxOption,
    FormFieldEntry,
    ListItem,
    TableCell,
)

# Markers that render as a plain hyphen in list serialization (§2.7 of the prompt).
_BULLET_MARKERS: frozenset[str] = frozenset(
    {"•", "●", "◦", "▪", "‣", "·", "*", "-"}
)


def serialize_checkbox_value(options: list[CheckboxOption]) -> str:
    """``"[x] A [ ] B"`` — options joined by single spaces, in order."""
    return " ".join(
        f"[{'x' if opt.checked else ' '}] {opt.text}" for opt in options
    )


def serialize_form_field(fields: list[FormFieldEntry]) -> str:
    """One line per field: ``label.text + " " + value.text`` (just the label
    when ``value`` is null). Joined with ``\\n``."""
    out: list[str] = []
    for f in fields:
        if f.value is None:
            out.append(f.label.text)
        else:
            out.append(f"{f.label.text} {f.value.text}")
    return "\n".join(out)


def serialize_list(items: list[ListItem]) -> str:
    """Each item on its own line, indented 2 spaces per level.

    Bullet markers become ``-``; non-bullet markers (``१.``, ``क)``) are kept
    verbatim. An empty marker produces a bare line with no leading separator.
    """
    out: list[str] = []
    for item in items:
        indent = "  " * item.level
        marker = "-" if item.marker in _BULLET_MARKERS else item.marker
        if marker:
            out.append(f"{indent}{marker} {item.text}")
        else:
            out.append(f"{indent}{item.text}")
    return "\n".join(out)


def serialize_table_html(cells: list[TableCell]) -> str:
    """Build the table HTML from the cells.

    Rules (verified against both sample tables):
      * ``<th>`` for header cells, ``<td>`` otherwise.
      * ``colspan``/``rowspan`` attributes only when > 1.
      * No whitespace between tags.
      * Cells covered by a span above/left are skipped.
    """
    if not cells:
        return "<table></table>"

    max_row = max(c.row + c.rowspan for c in cells)
    max_col = max(c.col + c.colspan for c in cells)

    grid: dict[tuple[int, int], TableCell] = {}
    covered: set[tuple[int, int]] = set()
    for c in cells:
        grid[(c.row, c.col)] = c
        for dr in range(c.rowspan):
            for dc in range(c.colspan):
                if dr == 0 and dc == 0:
                    continue
                covered.add((c.row + dr, c.col + dc))

    parts: list[str] = ["<table>"]
    for r in range(max_row):
        parts.append("<tr>")
        for c in range(max_col):
            if (r, c) in covered:
                continue
            cell = grid.get((r, c))
            if cell is None:
                continue
            tag = "th" if cell.is_header else "td"
            attrs = ""
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            parts.append(f"<{tag}{attrs}>{cell.text}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)