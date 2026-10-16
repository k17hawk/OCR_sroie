"""Table grid snapped to the scan (port of snap_table.py).

For a table region: find the printed ruling lines on the page image, build
the row / column grid from them, rebuild the cells from the table HTML
(rows, cols, rowspan, colspan, <th>, <br>) with boxes taken from the grid,
one text line per printed line inside each cell, and ruling_lines from the
segments actually printed (a partial rule stays partial).

If the HTML's rows / columns don't match the grid, or the grid can't be
found, the table is left unchanged and a warning says why. Nothing outside
the table (its region, block, lines, ruling lines, cells) is touched.
"""
from __future__ import annotations

import html as _html
import unicodedata
from html.parser import HTMLParser
from typing import Any, Optional

import cv2
import numpy as np

from app.services.geometry import bbox_to_quad
from app.services.lines import ink_bands, runs, union


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ---------------------------------------------------------------------------
# HTML -> cells
# ---------------------------------------------------------------------------

class _TableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self.cur: Optional[list[dict[str, Any]]] = None
        self.cell: Optional[dict[str, Any]] = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.cur = []
            self.rows.append(self.cur)
        elif tag in ("td", "th"):
            if self.cur is None:
                self.cur = []
                self.rows.append(self.cur)
            self.cell = {"header": tag == "th", "rowspan": _span(a.get("rowspan")),
                         "colspan": _span(a.get("colspan")), "text": ""}
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += "\n"

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.cur is not None:
            self.cur.append(self.cell)
            self.cell = None
        elif tag == "tr":
            self.cur = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data


def _span(v: Any) -> int:
    try:
        return max(1, int(v or 1))
    except (TypeError, ValueError):
        return 1


def parse_table_html(html: str) -> tuple[list[dict[str, Any]], int, int]:
    """Return ``(cells, n_rows, n_cols)``. A literal ``\\n`` typed in the HTML is a <br>."""
    html = (html or "").replace("\\n", "<br>")
    p = _TableHTML()
    p.feed(html)
    p.close()
    occupied: set[tuple[int, int]] = set()
    cells = []
    for r, row in enumerate(p.rows):
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    occupied.add((r + dr, c + dc))
            text = "\n".join(" ".join(part.split()) for part in cell["text"].split("\n"))
            cells.append({"row": r, "col": c, "rowspan": cell["rowspan"], "colspan": cell["colspan"],
                          "text": _nfc(text.strip("\n")), "is_header": cell["header"]})
            c += cell["colspan"]
    n_rows = max(r for r, _ in occupied) + 1 if occupied else 0
    n_cols = max(c for _, c in occupied) + 1 if occupied else 0
    return cells, n_rows, n_cols


def cells_to_html(cells: list[dict[str, Any]], n_rows: int) -> str:
    rows: dict[int, list[str]] = {r: [] for r in range(n_rows)}
    for c in sorted(cells, key=lambda c: (c["row"], c["col"])):
        tag = "th" if c["is_header"] else "td"
        attrs = (f' colspan="{c["colspan"]}"' if c["colspan"] > 1 else "") + \
                (f' rowspan="{c["rowspan"]}"' if c["rowspan"] > 1 else "")
        text = _html.escape(c["text"], quote=False).replace("\n", "<br>")
        rows[c["row"]].append(f"<{tag}{attrs}>{text}</{tag}>")
    return "<table>" + "".join("<tr>" + "".join(rows[r]) + "</tr>" for r in range(n_rows)) + "</table>"


# ---------------------------------------------------------------------------
# image -> grid
# ---------------------------------------------------------------------------

def _cluster(vals: list[int], tol: int = 10) -> list[int]:
    groups: list[list[int]] = []
    for v in sorted(vals):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(round(sum(g) / len(g))) for g in groups]


def find_grid(gray: np.ndarray, box, margin: int = 40, min_v: int = 100,
              min_h: int = 150, touch: int = 14, max_thick: int = 12):
    """Ruling lines inside ``box`` (+``margin``).

    Returns ``(xs, ys, vsegs, hsegs)`` in page pixels: column x's, row y's,
    vertical segments (x, ya, yb) and horizontal segments (y, xa, xb).
    Raises ``ValueError`` when fewer than 2 rules are found on an axis.
    """
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    X0, Y0 = max(0, x0 - margin), max(0, y0 - margin)
    X1, Y1 = min(W, x1 + margin), min(H, y1 + margin)
    sub = np.ascontiguousarray(gray[Y0:Y1, X0:X1])
    if sub.size == 0:
        raise ValueError("table box is outside the image")
    b = cv2.adaptiveThreshold(255 - sub, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -10)
    ver = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v)))
    hor = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_h, 1)))

    vsegs = []
    for xa, xb in runs(ver.any(0)):
        if xb - xa > max_thick:                 # too thick to be a ruling line (a black bar)
            continue
        for ya, yb in runs(ver[:, xa:xb + 1].any(1), gap=8):
            if yb - ya >= min_v:
                vsegs.append(((xa + xb) // 2 + X0, ya + Y0, yb + Y0))
    vx = _cluster([v[0] for v in vsegs])

    hsegs = []
    for ya, yb in runs(hor.any(1)):
        if yb - ya > max_thick:
            continue
        for xa, xb in runs(hor[ya:yb + 1].any(0), gap=8):
            if xb - xa < min_h:
                continue
            xa, xb = xa + X0, xb + X0
            # keep only lines that end on vertical rules: Devanagari headlines
            # (shirorekha) are long and horizontal too, but never touch cell borders
            if any(abs(xa - x) <= touch for x in vx) and any(abs(xb - x) <= touch for x in vx):
                hsegs.append(((ya + yb) // 2 + Y0, xa, xb))
    hy = _cluster([h[0] for h in hsegs])
    if len(vx) < 2 or len(hy) < 2:
        raise ValueError(f"found {len(vx)} vertical / {len(hy)} horizontal ruling lines")
    return vx, hy, vsegs, hsegs


def _covered(segs: list[tuple[int, int]], lo: int, hi: int) -> float:
    """Fraction of [lo, hi] covered by the (a, b) segments."""
    if hi <= lo:
        return 1.0
    got, cur = 0, lo
    for a, b in sorted(segs):
        a, b = max(a, cur), min(b, hi)
        if b > a:
            got += b - a
            cur = b
    return got / (hi - lo)


def _ruling_style(xs, ys, vsegs, hsegs, tol: int = 14) -> str:
    full_v = all(_covered([(a, b) for x, a, b in vsegs if abs(x - X) <= tol], ys[0], ys[-1]) >= 0.95
                 for X in xs)
    full_h = all(_covered([(a, b) for y, a, b in hsegs if abs(y - Y) <= tol], xs[0], xs[-1]) >= 0.95
                 for Y in ys)
    return "full" if full_v and full_h else "partial"


# ---------------------------------------------------------------------------
# snap
# ---------------------------------------------------------------------------

def _snap_one(gray, rec, blk, html_override, log, info) -> None:
    tables = rec.setdefault("tables", [])
    rec.setdefault("ruling_lines", [])
    bid = blk.get("id")
    ti = blk.get("table_index")
    t = tables[ti] if isinstance(ti, int) and 0 <= ti < len(tables) else None

    html = html_override if html_override is not None else ((t or {}).get("html") or blk.get("text") or "")
    cells, n_rows, n_cols = parse_table_html(html)
    if not cells:
        log.append(f"block {bid}: no table HTML, skipped")
        return
    try:
        xs, ys, vsegs, hsegs = find_grid(gray, blk["bbox"])
    except ValueError as e:
        log.append(f"block {bid}: {e} - skipped")
        return
    if len(xs) != n_cols + 1 or len(ys) != n_rows + 1:
        log.append(f"block {bid}: HTML has {n_rows} rows x {n_cols} cols, but the scan shows "
                   f"{len(ys) - 1} x {len(xs) - 1} - check the HTML; skipped")
        return

    # from here on the table is changed
    if t is None:
        ti = len(tables)
        t = {}
        tables.append(t)
        blk["table_index"] = ti
    old_hw = {(c.get("row"), c.get("col")): bool(c.get("handwritten")) for c in t.get("cells") or []}

    cells.sort(key=lambda c: (c["row"], c["col"]))
    for cid, c in enumerate(cells):
        c["id"] = cid
        c["bbox"] = [xs[c["col"]], ys[c["row"]], xs[c["col"] + c["colspan"]], ys[c["row"] + c["rowspan"]]]
        c["quad"] = bbox_to_quad(c["bbox"])
        c["empty"] = not c["text"].strip()
        c["handwritten"] = old_hw.get((c["row"], c["col"]), False)

    # text lines: one record per printed line inside each cell
    old = [ln for ln in rec.get("lines") or [] if ln.get("block_id") == bid]
    rest = [ln for ln in rec.get("lines") or [] if ln.get("block_id") != bid]
    new_lines = []
    for c in cells:
        bands = ink_bands(gray, c["bbox"], inset=9)
        texts = [x for x in c["text"].split("\n") if x.strip()]
        if not texts:
            if bands:
                log.append(f"  cell r{c['row']}c{c['col']}: ink found but cell text is empty "
                           f"(handwriting or a missed value?)")
            continue
        if len(bands) != len(texts):
            log.append(f"  cell r{c['row']}c{c['col']}: {len(texts)} text line(s) but {len(bands)} "
                       f"printed line(s) found - one line record used; split the text with <br>")
            pairs = [(" ".join(texts), union(bands) if bands else list(c["bbox"]))]
        else:
            pairs = list(zip(texts, bands))
        for text, box in pairs:
            ln = {"text": _nfc(text), "bbox": box, "block_id": bid,
                  "line_in_block": len(new_lines), "is_heading": False,
                  "font_size": round((box[3] - box[1]) * 0.8, 1), "font_id": None,
                  "table_index": ti, "cell_id": c["id"]}
            if c["handwritten"]:
                ln["handwritten"] = True
            ln["quad"] = bbox_to_quad(box)
            new_lines.append(ln)
    rec["lines"] = rest + new_lines

    # ruling lines: what is actually printed
    rec["ruling_lines"] = [r for r in rec["ruling_lines"] if r.get("table_index") != ti]
    for y, xa, xb in hsegs:
        rec["ruling_lines"].append({"table_index": ti, "orientation": "h", "width": 2,
                                    "p0": [xa, y], "p1": [xb, y], "bbox": [xa, y - 1, xb, y + 1]})
    for x, ya, yb in vsegs:
        rec["ruling_lines"].append({"table_index": ti, "orientation": "v", "width": 2,
                                    "p0": [x, ya], "p1": [x, yb], "bbox": [x - 1, ya, x + 1, yb]})

    frame = [xs[0], ys[0], xs[-1], ys[-1]]
    header_rows = 0
    while header_rows < n_rows:
        row = [c for c in cells if c["row"] == header_rows]
        if not row or not all(c["is_header"] for c in row):
            break
        header_rows += 1
    new_html = cells_to_html(cells, n_rows)
    t.update({"bbox": frame, "quad": bbox_to_quad(frame), "num_rows": n_rows, "num_cols": n_cols,
              "header_rows": header_rows, "html": new_html,
              "ruling_style": _ruling_style(xs, ys, vsegs, hsegs),
              "reading_order": [c["id"] for c in cells], "cells": cells})
    t.setdefault("theme", None)
    t.setdefault("col_types", None)
    blk.update({"bbox": frame, "quad": bbox_to_quad(frame), "text": new_html})
    for b in rec.get("blocks") or []:
        if b.get("block_id") == bid:
            b.update({"bbox": frame, "quad": bbox_to_quad(frame)})
    info.append(f"block {bid}: table {n_rows}x{n_cols} snapped to the scan "
               f"(rows at y={ys}, cols at x={xs}); {len(cells)} cells, "
               f"{len(new_lines)} text lines (was {len(old)})")


def snap_tables(gray: np.ndarray, rec: dict[str, Any], element_id: Optional[int] = None,
                html: Optional[str] = None) -> tuple[dict[str, Any], list[str], list[str]]:
    """Snap every table region (or only ``element_id``) to the scan.

    ``html`` overrides the stored HTML (the editor's typed table) and needs
    ``element_id``. Returns ``(rec, warnings, info)``; ``rec`` is changed in
    place. ``info`` has one summary per table that was snapped.
    """
    log: list[str] = []
    info: list[str] = []
    for blk in rec.get("layout") or []:
        if not isinstance(blk, dict) or blk.get("type") != "table":
            continue
        if element_id is not None and blk.get("id") != element_id:
            continue
        _snap_one(gray, rec, blk, html if element_id is not None else None, log, info)
    return rec, log, info
