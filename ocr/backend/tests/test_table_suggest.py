"""Suggest a table layout from a drawn box (app/services/table_suggest.py on measure.py).

Synthetic scans with known rows / columns / merged cells: clean, light grey rules, tilted both
ways, bleed-through, a gap in a rule, borderless, half-size page.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.services.table_suggest import suggest_table
from conftest import make_page

COLS = [140, 520, 260, 260, 260, 360]
ROWS = [90, 70, 80, 120, 80, 80, 80, 120, 80]
# (row, col, rowspan, colspan, text): 2 header rows with merged cells
CELLS = [(0, 0, 2, 1, "No."), (0, 1, 2, 1, "Occupation"), (0, 2, 1, 3, "Family members"),
         (0, 5, 2, 1, "Remarks"), (1, 2, 1, 1, "Male"), (1, 3, 1, 1, "Female"), (1, 4, 1, 1, "Total")]
for _r in range(7):
    for _c in range(6):
        CELLS.append((2 + _r, _c, 1, 1, ["a)", "Farming", "2", "3", "5", ""][_c] if _r % 2 == 0 else ""))
WANT = {(r, c, rs, cs) for r, c, rs, cs, _ in CELLS}


def draw(scale=1.0, gray=0, skew=0.0, bleed=False, gap=False, borderless=False):
    W, H = int(2481 * scale), int(1700 * scale)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=int(30 * scale))
    xs, ys = [int(300 * scale)], [int(400 * scale)]
    for w in COLS:
        xs.append(xs[-1] + int(w * scale))
    for h in ROWS:
        ys.append(ys[-1] + int(h * scale))
    occ = {}
    for i, (r, c, rs, cs, _) in enumerate(CELLS):
        for rr in range(r, r + rs):
            for cc in range(c, c + cs):
                occ[(rr, cc)] = i
    lw = max(1, round(3 * scale))
    nr, nc = len(ROWS), len(COLS)
    for j, y in enumerate(ys):
        for c in range(nc):
            if j in (0, nr) or occ[(j - 1, c)] != occ[(j, c)]:
                d.line([(xs[c], y), (xs[c + 1], y)], fill=gray, width=lw)
    if not borderless:
        for i, x in enumerate(xs):
            for r in range(nr):
                if i in (0, nc) or occ[(r, i - 1)] != occ[(r, i)]:
                    y1 = ys[r + 1] if not (gap and i == 3 and r == 5) else ys[r] + (ys[r + 1] - ys[r]) // 3
                    d.line([(x, ys[r]), (x, y1)], fill=gray, width=lw)
    if bleed:   # faint rule shining through from the back of the page
        d.line([(xs[1] + int(180 * scale), ys[2] + 10), (xs[1] + int(180 * scale), ys[-1] - 40)], fill=205, width=lw)
    for r, c, rs, cs, t in CELLS:
        if t:
            d.text((xs[c] + int(18 * scale), (ys[r] + ys[r + rs]) / 2 - 18 * scale), t, fill=0, font=font)
    g = np.array(img)
    if skew:
        g = cv2.warpAffine(g, cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1), (W, H),
                           flags=cv2.INTER_CUBIC, borderValue=255)
    m = int(25 * scale)
    return g, [xs[0] - m, ys[0] - m + 8, xs[-1] + m - 10, ys[-1] + m], (xs, ys)


def spans(s):
    return {(c["row"], c["col"], c["rowspan"], c["colspan"]) for c in s["cells"]}


@pytest.mark.parametrize("kw", [dict(), dict(gray=185), dict(skew=0.8), dict(skew=-1.5, gray=120),
                                dict(scale=1191 / 2481)],
                         ids=["clean", "light_grey", "tilt_0.8", "tilt_-1.5", "half_size"])
def test_grid_and_merged_cells_from_the_box(kw):
    g, box, _ = draw(**kw)
    s = suggest_table(g, box)
    assert s["ok"], s["warnings"]
    assert (s["num_rows"], s["num_cols"]) == (9, 6)
    assert spans(s) == WANT
    assert s["warnings"] == []
    if kw.get("skew"):
        assert abs(s["skew_deg"] - kw["skew"]) < 0.15   # the page tilt, OpenCV sign (+ = counter-clockwise)


def test_cell_boxes_follow_the_printed_rules():
    g, box, (xs, ys) = draw()
    s = suggest_table(g, box)
    cell = next(c for c in s["cells"] if (c["row"], c["col"]) == (0, 2))     # the colspan-3 header
    assert all(abs(a - b) <= 4 for a, b in zip(cell["bbox"], [xs[2], ys[0], xs[5], ys[1]]))
    assert s["html"].startswith("<table><tr><td rowspan=\"2\">")


def test_tilted_cells_are_quads_in_the_original_image():
    g, box, (xs, ys) = draw(skew=0.8)
    W, H = g.shape[1], g.shape[0]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0.8, 1)
    s = suggest_table(g, box)
    c = next(c for c in s["cells"] if (c["row"], c["col"]) == (8, 5))        # far corner: most tilt
    want = np.c_[[[xs[5], ys[8]], [xs[6], ys[8]], [xs[6], ys[9]], [xs[5], ys[9]]], np.ones(4)] @ M.T
    assert np.abs(np.asarray(c["quad"], float) - want).max() <= 6


def test_faint_bleed_through_line_is_not_a_column():
    g, box, _ = draw(bleed=True)
    s = suggest_table(g, box)
    assert s["num_cols"] == 6 and spans(s) == WANT
    assert any("faint" in w for w in s["warnings"])


def test_gap_in_a_rule_is_flagged():
    g, box, _ = draw(gap=True)
    s = suggest_table(g, box)
    assert s["ok"]
    assert any("row 6, columns 3-4" in w and "printed" in w for w in s["warnings"])


def test_borderless_table_says_why():
    g, box, _ = draw(borderless=True)
    s = suggest_table(g, box)
    assert s["ok"] is False
    assert "borderless" in s["warnings"][0]


def test_endpoint(pagestore):
    from app.main import app

    g, box, _ = draw()
    rec = make_page(width=g.shape[1], height=g.shape[0])
    pagestore(rec, g)
    r = TestClient(app).post("/documents/DOC-0001/suggest-table", json={"bbox": box})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and (body["num_rows"], body["num_cols"]) == (9, 6)
    assert len(body["cells"]) == len(CELLS)
    assert all(type(v) is int for c in body["cells"] for v in c["bbox"])   # plain JSON ints
