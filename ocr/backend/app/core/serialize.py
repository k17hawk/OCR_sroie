"""Deterministic serializers for every derived string in the schema.

All functions are pure. They are the *only* place that produces a derived
string, so the exporter and the frontend preview are guaranteed identical.
"""
from __future__ import annotations

from typing import Callable

from app.schemas.doc_v2 import (
    FormFieldEntry,
    ListItem,
    TableCell,
)

# Markers that render as a plain hyphen in list serialization (§2.7 of the prompt).
_BULLET_MARKERS: frozenset[str] = frozenset(
    {"•", "●", "◦", "▪", "‣", "·", "*", "-"}
)


def serialize_form_field(
    fields: list[FormFieldEntry],
    line_text: "Callable[[int], str | None] | None" = None,
) -> str:
    """One line per field: the label, then the value text, or for a checkbox
    field the option words in ``options[]`` order. Joined with ``\\n``.
    ``line_text(id)`` gives the text of an option line."""
    out: list[str] = []
    for f in fields:
        parts = [f.label.text] if f.label is not None and f.label.text else []
        if f.options:
            for o in f.options:
                t = line_text(o.line_id) if line_text is not None else None
                if t:
                    parts.append(t)
        elif f.value is not None and f.value.text:
            parts.append(f.value.text)
        out.append(" ".join(parts))
    return "\n".join(out)


def serialize_list(
    items: list[ListItem],
    child_text: "Callable[[int], str | None] | None" = None,
) -> str:
    """Each item on its own line, indented 2 spaces per level.

    Bullet markers become ``-``; non-bullet markers (``१.``, ``क)``) are kept
    verbatim. An item's ``children`` (a note, a figure, … inside the item)
    follow it, one level deeper; ``child_text(id)`` gives each child's text.
    """
    out: list[str] = []
    for item in items:
        indent = "  " * item.level
        marker = "-" if item.marker in _BULLET_MARKERS else item.marker
        first, *rest = item.text.split("\n")
        out.append(f"{indent}{marker} {first}")
        # an item printed on several lines: its further lines under it
        for part in rest:
            out.append(f"{indent}  {part}")
        if child_text is not None:
            for cid in item.children or []:
                text = child_text(cid)
                if not text:
                    continue
                for part in text.split("\n"):
                    out.append(f"{indent}  {part}")
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
            # a line break inside a cell is written as <br>
            parts.append(f"<{tag}{attrs}>{cell.text.replace(chr(10), '<br>')}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)
