"""Table structure detection on drawn tables (no models needed).

Run:  cd ocr/backend && python -m pytest tests/test_table_structure.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.pipeline.adapters import mock
from app.pipeline.table_ocr import recognize_table
from app.pipeline.table_structure import html_layout
from app.schemas.doc_v2 import Doc

FONTS = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int):
    for f in FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _rot(p, c, deg):
    a = math.radians(deg)
    dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(a) - dy * math.sin(a), c[1] + dx * math.sin(a) + dy * math.cos(a))


def make_page(spec, *, rules="full", skew=0.0, row_lines=False, x0=100, y0=200,
              col_w=(80, 260, 220), row_h=36, W=1000, H=900):
    """spec: rows of cells; a cell is text or (text, rowspan, colspan).
    Returns (image, doc dict, table quad, expected html)."""
    # place the cells on the grid
    nrows = len(spec)
    ncols = len(col_w)
    taken = set()
    placed = []
    for r, row in enumerate(spec):
        c = 0
        for cell in row:
            text, rs, cs = (cell, 1, 1) if isinstance(cell, str) else cell
            while (r, c) in taken:
                c += 1
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            placed.append((r, c, rs, cs, text))
            c += cs
    xs = [x0]
    for w in col_w:
        xs.append(xs[-1] + w)
    ys = [y0 + i * row_h for i in range(nrows + 1)]
    centre = ((xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2)
    R = lambda p: _rot(p, centre, skew)  # noqa: E731

    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    f = _font(int(row_h * 0.45))
    lines = []

    def text_at(x, y, t, hw=False):
        # draw word-by-word along the skewed baseline
        bw = dr.textlength(t, font=f)
        p = R((x, y))
        layer = Image.new("L", (int(bw) + 8, row_h), 0)
        ImageDraw.Draw(layer).text((2, 2), t, font=f, fill=255)
        layer = layer.rotate(-skew, expand=True, resample=Image.BICUBIC)
        img.paste((20, 20, 20), (int(p[0]), int(p[1])), layer)
        q = [R((x, y)), R((x + bw, y)), R((x + bw, y + row_h * 0.6)), R((x, y + row_h * 0.6))]
        bx = [min(a[0] for a in q), min(a[1] for a in q), max(a[0] for a in q), max(a[1] for a in q)]
        lines.append({
            "text": t, "bbox": [math.floor(bx[0]), math.floor(bx[1]), math.ceil(bx[2]), math.ceil(bx[3])],
            "block_id": 0, "line_in_block": len(lines), "is_heading": False, "font_size": 16.0,
            "font_id": None, "quad": [[round(a[0], 1), round(a[1], 1)] for a in q],
            **({"handwritten": True} if hw else {}),
        })

    for (r, c, rs, cs, t) in placed:
        if t:
            text_at(xs[c] + 6, ys[r] + (row_h * rs) / 2 - row_h * 0.3, t)
    if row_lines:
        # page OCR read each row as ONE line across the columns
        merged = []
        for r in range(nrows):
            mine = [ln for ln in lines if ys[r] <= (ln["bbox"][1] + ln["bbox"][3]) / 2 < ys[r + 1]]
            if not mine:
                continue
            mine.sort(key=lambda ln: ln["bbox"][0])
            b = [min(l["bbox"][0] for l in mine), min(l["bbox"][1] for l in mine),
                 max(l["bbox"][2] for l in mine), max(l["bbox"][3] for l in mine)]
            merged.append({**mine[0], "text": " ".join(l["text"] for l in mine), "bbox": b,
                           "quad": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]})
        lines[:] = merged

    def seg(a, b):
        dr.line([R(a), R(b)], fill=(0, 0, 0), width=2)

    if rules in ("full", "horizontal"):
        # horizontal rules, interrupted under row-spanning cells
        for i, y in enumerate(ys):
            for c in range(ncols):
                spanned = any(r < i < r + rs and c0 <= c < c0 + cs for (r, c0, rs, cs, _) in placed)
                if not spanned:
                    seg((xs[c], y), (xs[c + 1], y))
    if rules == "full":
        for j, x in enumerate(xs):
            for r in range(nrows):
                spanned = any(c < j < c + cs and r0 <= r < r0 + rs for (r0, c, rs, cs, _) in placed)
                if not spanned:
                    seg((x, ys[r]), (x, ys[r + 1]))

    quad = [list(R((xs[0], ys[0]))), list(R((xs[-1], ys[0]))), list(R((xs[-1], ys[-1]))), list(R((xs[0], ys[-1])))]
    rows_html = {}
    for (r, c, rs, cs, t) in sorted(placed):
        tag = "th" if r == 0 else "td"
        a = (f' colspan="{cs}"' if cs > 1 else "") + (f' rowspan="{rs}"' if rs > 1 else "")
        rows_html.setdefault(r, []).append(f"<{tag}{a}>{t}</{tag}>")
    html = "<table>" + "".join("<tr>" + "".join(rows_html.get(r, [])) + "</tr>" for r in range(nrows)) + "</table>"
    doc = {
        "v": 2, "image": "x.jpg", "width": W, "height": H, "document_id": "doc_test",
        "page_number": 1, "page_count": 1, "language": "ne", "digit_style": "dev",
        "doc_type": "other", "text": "", "fonts": [], "font_ids": [],
        "layout": [], "blocks": [], "lines": lines, "tables": [], "ruling_lines": [],
        "extra_regions": [],
        "quality": {"is_grayscale": False, "is_binarized": False, "perspective": False,
                    "is_clean": True, "has_stamp": False, "has_signature": False,
                    "has_handwriting": False, "has_fold": False, "has_bleedthrough": False,
                    "raw_lengths": {"paragraph_chars": 0, "list_items": 0, "form_fields": 0, "table_cells": 0}},
        "annotation": {"source": "real", "tool": "t", "status": "in_progress"},
    }
    return img, Doc.model_validate(doc), quad, html


BASIC = [
    ["क्र.सं.", "निर्वाचन गराउन", "एउटा"],
    ["१", "१७,६३९", "८२,५३९"],
    ["२", "४६,७२६", "३३,९४०"],
    ["३", "८५,७९५", "५२,५१५"],
    ["४", "९४,१११", "९२,७२०"],
    ["५", "७४,३२७", "७,१०६"],
]


def run(img, doc, quad, adapter=None):
    return recognize_table(doc=doc, img=img, quad=quad, adapter=adapter or mock.MockTable())


@pytest.mark.parametrize("skew", [0.0, 0.8, -1.2])
def test_full_rules_every_td_found(skew):
    img, doc, quad, html = make_page(BASIC, skew=skew)
    t = run(img, doc, quad)
    assert (t.num_rows, t.num_cols, len(t.cells)) == (6, 3, 18)
    assert t.html == html
    assert t.ruling_style == "full"


def test_drawn_box_on_skewed_page():
    """User drags an axis-aligned box around a slightly rotated table."""
    img, doc, quad, html = make_page(BASIC, skew=1.0)
    xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
    box = [[min(xs) - 4, min(ys) - 4], [max(xs) + 4, min(ys) - 4], [max(xs) + 4, max(ys) + 4], [min(xs) - 4, max(ys) + 4]]
    t = run(img, doc, box)
    assert (t.num_rows, t.num_cols) == (6, 3)
    assert t.html == html


def test_horizontal_rules_only():
    img, doc, quad, html = make_page(BASIC, rules="horizontal")
    t = run(img, doc, quad)
    assert (t.num_rows, t.num_cols, len(t.cells)) == (6, 3, 18)
    assert t.html == html
    assert t.ruling_style == "horizontal"


def test_borderless():
    img, doc, quad, html = make_page(BASIC, rules="none")
    t = run(img, doc, quad)
    assert (t.num_rows, t.num_cols, len(t.cells)) == (6, 3, 18)
    assert t.html == html
    assert t.ruling_style == "none"


def test_spans():
    spec = [
        ["क्र.सं.", ("विवरण", 1, 2)],
        ["१", ("बाग्मती", 2, 1), "५१"],
        ["२", "६२"],
        ["३", "गण्डकी", "७३"],
    ]
    img, doc, quad, html = make_page(spec)
    t = run(img, doc, quad)
    assert (t.num_rows, t.num_cols) == (4, 3)
    assert t.html == html


def test_row_read_as_one_line_is_split_not_copied():
    img, doc, quad, html = make_page(BASIC, row_lines=True)
    t = run(img, doc, quad)
    assert (t.num_rows, t.num_cols, len(t.cells)) == (6, 3, 18)
    body = [c for c in t.cells if c.row == 1]
    assert [c.text for c in sorted(body, key=lambda c: c.col)] == ["१", "१७,६३९", "८२,५३९"]


def test_html_layout_spans():
    hl = html_layout('<table><tr><th rowspan="2">a</th><th colspan="2">b</th></tr><tr><td>c</td><td>d</td></tr></table>')
    assert hl["rows"] == 2 and hl["cols"] == 3
    assert [(c["row"], c["col"]) for c in hl["cells"]] == [(0, 0), (0, 1), (1, 1), (1, 2)]


class _HtmlOnlyAdapter:
    """Paddle-style: HTML with <th>, no geometry."""
    name = "html-only"

    def __init__(self, html):
        self.html = html

    def recognize(self, img, quad):
        return {"html": self.html, "rows": 0, "cols": 0, "cells": []}


def test_model_html_header_rows_used_when_grid_matches():
    img, doc, quad, _ = make_page(BASIC)
    html = "<table>" + "<tr><th></th><th></th><th></th></tr>" * 2 + "<tr><td></td><td></td><td></td></tr>" * 4 + "</table>"
    t = run(img, doc, quad, _HtmlOnlyAdapter(html))
    assert t.header_rows == 2
