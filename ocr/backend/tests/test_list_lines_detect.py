"""Detect lines on a bulleted list: every printed row gets its own measured
box, tied to the bullet it belongs to (item_index / list_level)."""
from __future__ import annotations

from app.services.lines import detect_page_lines
from conftest import blank, make_page, write

BOX = [100, 100, 900, 420]
ITEMS = [
    {"marker": "•", "level": 0, "text": "First point on one row"},
    {"marker": "•", "level": 0, "text": "Second point that wraps\nonto a second row"},
    {"marker": "•", "level": 0, "text": "Third point"},
]
TEXT = "• First point on one row\n• Second point that wraps\n  onto a second row\n• Third point"


def _page(typ):
    g = blank()
    rows = [write(g, 130, 120, "• First point on one row"),
            write(g, 130, 190, "• Second point that wraps"),
            write(g, 160, 260, "onto a second row"),
            write(g, 130, 330, "• Third point")]
    rec = make_page(layout=[{"id": 0, "type": typ, "bbox": BOX, "reading_order": 0, "text": TEXT,
                             "list_style": "bullet", "items": [dict(i) for i in ITEMS]}])
    return g, rec, rows


def test_each_row_of_a_sublist_is_measured_and_tied_to_its_bullet():
    g, rec, rows = _page("sublist")
    rec, warnings = detect_page_lines(g, rec)
    assert warnings == []
    lines = [ln for ln in rec["lines"] if ln["block_id"] == 0]
    assert [ln["text"] for ln in lines] == ["• First point on one row", "• Second point that wraps",
                                           "onto a second row", "• Third point"]
    assert [ln["item_index"] for ln in lines] == [0, 1, 1, 2]
    assert all(ln["list_level"] == 0 for ln in lines)
    # each box is that row's own ink, bullet included
    for ln, ink in zip(lines, rows):
        assert abs(ln["bbox"][0] - ink[0]) <= 8 and abs(ln["bbox"][1] - ink[1]) <= 8


def test_plain_list_rows_are_tied_too():
    g, rec, _ = _page("list")
    rec, _ = detect_page_lines(g, rec)
    assert [ln.get("item_index") for ln in rec["lines"] if ln["block_id"] == 0] == [0, 1, 1, 2]
