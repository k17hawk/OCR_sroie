"""Step 3: table grid snapped to the printed ruling lines."""
from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.schemas.doc_v2 import Doc
from app.services.table_snap import cells_to_html, parse_table_html, snap_tables
from conftest import blank, ink, make_page, write

XS = [100, 300, 500, 700]          # column rules
YS = [100, 160, 300, 380]          # uneven rows: 60, 140, 80 px

# header row: one <th> over columns 0-1 -> no rule at x=300 in row 0
HTML = ('<table><tr><th colspan="2">शीर्षक</th><th>ख</th></tr>'
        r'<tr><td>a\nb</td><td>c</td><td></td></tr>'      # literal backslash-n
        '<tr><td>d</td><td>e</td><td>f</td></tr></table>')


def _scan():
    g = blank()
    for y in YS:                                   # full-width rows
        ink(g, (XS[0] - 1, y - 1, XS[-1] + 2, y + 2))
    for x in (100, 500, 700):                      # full-height columns
        ink(g, (x - 1, YS[0] - 1, x + 2, YS[-1] + 2))
    ink(g, (299, YS[1] - 1, 302, YS[-1] + 2))      # x=300 only below the header: partial
    # two printed lines in cell r1c0
    write(g, 130, 180, "line a", size=24)
    write(g, 130, 230, "line b", size=24)
    # a Devanagari headline (shirorekha) in r1c1: long, horizontal, touches no rule
    ink(g, (320, 229, 480, 232))
    return g


def _page(html=HTML):
    table = {"bbox": [90, 90, 710, 390], "quad": None, "num_rows": 1, "num_cols": 1,
             "header_rows": 0, "ruling_style": "full", "theme": None, "col_types": None,
             "html": html, "reading_order": [0],
             "cells": [{"id": 0, "row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "",
                        "is_header": False, "handwritten": False, "empty": True,
                        "bbox": [90, 90, 710, 390]}]}
    other_line = {"text": "बाहिर", "bbox": [100, 500, 300, 530], "block_id": 1, "line_in_block": 0,
                  "is_heading": False, "font_size": 24.0, "font_id": None}
    return make_page(
        layout=[{"id": 0, "type": "table", "bbox": [90, 90, 710, 390], "text": html, "table_index": 0},
                {"id": 1, "type": "paragraph", "bbox": [100, 500, 300, 530], "text": "बाहिर"}],
        blocks=[{"type": "table", "bbox": [90, 90, 710, 390], "block_id": 0}],
        tables=[table], lines=[other_line],
    )


def _cell(t, row, col):
    return next(c for c in t["cells"] if c["row"] == row and c["col"] == col)


def test_cells_follow_the_printed_grid_with_uneven_rows_and_colspan():
    rec, warnings, info = snap_tables(_scan(), _page())
    t = rec["tables"][0]
    assert info and "3x3 snapped" in info[0], (warnings, info)
    assert (t["num_rows"], t["num_cols"], t["header_rows"]) == (3, 3, 1)
    head = _cell(t, 0, 0)
    assert head["colspan"] == 2 and head["is_header"]
    assert head["bbox"] == [100, 100, 500, 160]                   # spans two columns
    assert _cell(t, 0, 2)["bbox"] == [500, 100, 700, 160]
    assert _cell(t, 1, 0)["bbox"] == [100, 160, 300, 300]         # the 140 px row
    assert _cell(t, 2, 2)["bbox"] == [500, 300, 700, 380]         # the 80 px row
    empty = _cell(t, 1, 2)
    assert empty["empty"] is True and empty["text"] == ""         # empty cell kept
    assert t["bbox"] == [100, 100, 700, 380]
    assert rec["layout"][0]["bbox"] == t["bbox"] and rec["blocks"][0]["bbox"] == t["bbox"]
    Doc.model_validate(rec)


def test_literal_backslash_n_becomes_br_and_two_lines():
    rec, _, _ = snap_tables(_scan(), _page())
    t = rec["tables"][0]
    assert _cell(t, 1, 0)["text"] == "a\nb"
    assert "<td>a<br>b</td>" in t["html"]
    two = [ln for ln in rec["lines"] if ln.get("cell_id") == _cell(t, 1, 0)["id"]]
    assert [ln["text"] for ln in two] == ["a", "b"]
    assert two[0]["bbox"][1] < two[1]["bbox"][1]
    assert all(ln["table_index"] == 0 and ln["block_id"] == 0 for ln in two)


def test_partial_vertical_rule_stays_partial():
    rec, _, _ = snap_tables(_scan(), _page())
    v300 = [r for r in rec["ruling_lines"] if r["orientation"] == "v" and abs(r["p0"][0] - 300) <= 2]
    assert len(v300) == 1
    assert abs(v300[0]["p0"][1] - 160) <= 3 and abs(v300[0]["p1"][1] - 380) <= 3
    assert rec["tables"][0]["ruling_style"] == "partial"


def test_shirorekha_is_not_a_row_rule():
    rec, _, _ = snap_tables(_scan(), _page())
    hs = sorted(r["p0"][1] for r in rec["ruling_lines"] if r["orientation"] == "h")
    assert all(abs(y - 230) > 5 for y in hs)
    assert rec["tables"][0]["num_rows"] == 3


def test_other_regions_lines_are_untouched():
    rec, _, _ = snap_tables(_scan(), _page())
    assert _page()["lines"][0] in rec["lines"]


def test_shape_mismatch_leaves_the_table_unchanged_with_a_warning():
    two_rows = '<table><tr><td>a</td><td>b</td><td>c</td></tr><tr><td>d</td><td>e</td><td>f</td></tr></table>'
    before = _page(two_rows)
    rec, warnings, info = snap_tables(_scan(), copy.deepcopy(before))
    assert rec == before and info == []
    assert warnings == ["block 0: HTML has 2 rows x 3 cols, but the scan shows 3 x 3 - check the HTML; skipped"]


def test_no_rules_found_is_a_warning():
    rec, warnings, _ = snap_tables(blank(), _page())
    assert "ruling lines - skipped" in warnings[0]
    assert rec == _page()


def test_html_round_trip_escapes_text():
    cells, nr, nc = parse_table_html("<table><tr><td>a &amp; b &lt;c&gt;</td></tr></table>")
    assert cells[0]["text"] == "a & b <c>" and (nr, nc) == (1, 1)
    assert cells_to_html(cells, nr) == "<table><tr><td>a &amp; b &lt;c&gt;</td></tr></table>"


def test_endpoint_uses_the_typed_html(pagestore):
    from app.main import app

    rec = _page("<table><tr><td>पुरानो</td></tr></table>")
    pagestore(rec, _scan())
    r = TestClient(app).post("/documents/DOC-0001/snap-table",
                             json={"doc": rec, "element_id": 0, "html": HTML})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["record"]["tables"][0]["num_rows"] == 3 and body["info"]


def test_endpoint_rejects_a_non_table(pagestore):
    from app.main import app

    pagestore(_page(), _scan())
    r = TestClient(app).post("/documents/DOC-0001/snap-table", json={"element_id": 1})
    assert r.status_code == 422


def test_cell_lines_follow_their_own_ink():
    g = _scan()
    a = write(g, 160, 264, "Ty", size=20)  # a short third line, indented, low in cell r1c0
    rec, _, _ = snap_tables(g, _page(HTML.replace(r"a\nb", r"a\nb\nc")))
    t = rec["tables"][0]
    cell = _cell(t, 1, 0)
    three = [ln for ln in rec["lines"] if ln.get("cell_id") == cell["id"]]
    assert [ln["text"] for ln in three] == ["a", "b", "c"]
    assert abs((three[2]["bbox"][0] + 6) - a[0]) <= 3 and abs((three[2]["bbox"][2] - 6) - a[2]) <= 3
    assert all(cell["bbox"][0] <= ln["bbox"][0] and ln["bbox"][2] <= cell["bbox"][2] for ln in three)
