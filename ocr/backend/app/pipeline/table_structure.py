"""Table structure from the page image + the page's text lines.

Why this exists: the table adapters alone missed cells. Mock / tesseract /
surya have no table model and returned a 2x2 placeholder grid; Paddle only
returned HTML (no geometry), which the old code ignored -> 2x2 again; TATR
was fed the whole page and dropped rows/columns it was unsure of. Every
missed row or column became a missing ``<td>``.

This module finds the grid itself, one axis at a time:

  rows    : horizontal ruling lines   -> adapter rows   -> text-line rows
  columns : vertical ruling lines     -> adapter cols   -> gaps between text

The table region is deskewed first (angle from the quad, refined from the
ink), so slightly rotated scans still give straight rows. Cell quads are
rotated back, so they follow the page like the reference data does.

Spans: in a ruled table a missing rule between two grid cells merges them
(colspan / rowspan). Header rows: ``<th>`` from adapter HTML if it fits the
grid, else the first row plus any row a first-row cell spans down into.

Pure functions; numpy + Pillow only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

Pt = tuple[float, float]


# ---------------------------------------------------------------------------
# Small geometry
# ---------------------------------------------------------------------------

def _quad_angle_deg(quad: Sequence[Sequence[float]]) -> float:
    """Angle of the quad's top and bottom edges (image coords, y down)."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = [tuple(p) for p in quad]
    a_top = math.degrees(math.atan2(y1 - y0, x1 - x0))
    a_bot = math.degrees(math.atan2(y2 - y3, x2 - x3))
    a = (a_top + a_bot) / 2
    return a if abs(a) < 15 else 0.0


@dataclass
class Frame:
    """Deskewed crop of the table and the mapping back to page coords."""
    img: Image.Image          # deskewed crop, RGB
    ox: float                 # crop origin on the page
    oy: float
    cx: float                 # rotation centre (crop coords)
    cy: float
    angle: float              # degrees the page was rotated to level the table
    region: tuple[int, int, int, int]  # table extent inside the crop (x0,y0,x1,y1)

    def to_page(self, x: float, y: float) -> Pt:
        # undo the rotation (PIL rotates the content by `angle`, so points
        # come back by rotating with the opposite sign about the same centre)
        a = math.radians(self.angle)
        dx, dy = x - self.cx, y - self.cy
        px = self.cx + dx * math.cos(a) - dy * math.sin(a)
        py = self.cy + dx * math.sin(a) + dy * math.cos(a)
        return (px + self.ox, py + self.oy)

    def to_crop(self, x: float, y: float) -> Pt:
        a = math.radians(self.angle)
        dx, dy = x - self.ox - self.cx, y - self.oy - self.cy
        return (self.cx + dx * math.cos(a) + dy * math.sin(a),
                self.cy - dx * math.sin(a) + dy * math.cos(a))

    def quad_to_page(self, b: Sequence[float]) -> list[list[float]]:
        x0, y0, x1, y1 = b
        return [[round(v, 1) for v in self.to_page(x, y)]
                for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]

    def box_to_crop(self, b: Sequence[float]) -> tuple[float, float, float, float]:
        pts = [self.to_crop(x, y) for x, y in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))


def _ink(img: Image.Image) -> np.ndarray:
    """Boolean ink mask (Otsu threshold on grey)."""
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    total = g.size
    if total == 0:
        return np.zeros_like(g, dtype=bool)
    w = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w - mu * total) ** 2 / (w * (total - w))
    between[~np.isfinite(between)] = 0
    t = int(np.argmax(between))
    t = min(t, 200)  # a near-blank crop must not turn paper grain into ink
    return g <= t


def _refine_angle(ink: np.ndarray, base: float) -> float:
    """Extra rotation (degrees) that makes rows of ink sharpest.

    Classic projection-profile deskew: the angle whose row sums have the
    largest energy. Grey values are used (not re-thresholded) so thin rules
    survive the downscale."""
    h, w = ink.shape
    if h < 10 or w < 10 or not ink.any():
        return 0.0
    small = Image.fromarray((ink * 255).astype(np.uint8))
    scale = min(1.0, 800 / max(w, h))
    if scale < 1:
        small = small.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BOX)

    def score(d: float) -> float:
        r = np.asarray(small.rotate(float(d), resample=Image.BILINEAR, fillcolor=0), dtype=np.float64)
        prof = r.sum(axis=1)
        return float((prof ** 2).sum())

    coarse = max(np.arange(-2.0, 2.01, 0.25), key=score)
    fine = max(np.arange(coarse - 0.2, coarse + 0.201, 0.05), key=score)
    return float(round(fine, 3))


def make_frame(page: Image.Image, quad: Sequence[Sequence[float]], margin: int = 8) -> Frame:
    xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
    W, H = page.size
    x0 = max(0, int(math.floor(min(xs))) - margin)
    y0 = max(0, int(math.floor(min(ys))) - margin)
    x1 = min(W, int(math.ceil(max(xs))) + margin)
    y1 = min(H, int(math.ceil(max(ys))) + margin)
    crop = page.convert("RGB").crop((x0, y0, x1, y1))
    cx, cy = (x1 - x0) / 2, (y1 - y0) / 2

    angle = _quad_angle_deg(quad)
    # PIL.rotate(+a) turns the content counter-clockwise on screen; a table
    # whose right side is higher (negative image angle) needs clockwise.
    first = crop.rotate(angle, resample=Image.BICUBIC, center=(cx, cy), fillcolor=(255, 255, 255)) if abs(angle) > 0.05 else crop
    extra = _refine_angle(_ink(first), angle)
    total = angle + extra
    img = crop.rotate(total, resample=Image.BICUBIC, center=(cx, cy), fillcolor=(255, 255, 255)) if abs(total) > 0.05 else crop

    f = Frame(img=img, ox=x0, oy=y0, cx=cx, cy=cy, angle=total, region=(0, 0, img.width, img.height))
    pts = [f.to_crop(px, py) for px, py in quad]
    rx0 = max(0, int(round(min(p[0] for p in pts)))); ry0 = max(0, int(round(min(p[1] for p in pts))))
    rx1 = min(img.width, int(round(max(p[0] for p in pts)))); ry1 = min(img.height, int(round(max(p[1] for p in pts))))
    f.region = (rx0, ry0, max(rx0 + 1, rx1), max(ry0 + 1, ry1))
    return f


# ---------------------------------------------------------------------------
# Ruling lines
# ---------------------------------------------------------------------------

def _close1d(mask: np.ndarray, axis: int, gap: int) -> np.ndarray:
    """Fill gaps of up to `gap` px along `axis` (broken / dotted scan lines)."""
    if gap <= 0:
        return mask
    # a pixel is filled when there is ink within `gap` px on BOTH sides
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    for t in range(1, gap + 1):
        left |= np.roll(mask, t, axis=axis)
        right |= np.roll(mask, -t, axis=axis)
    return mask | (left & right)


def _runs(v: np.ndarray, minlen: int) -> list[tuple[int, int]]:
    """[start, end) of True runs at least `minlen` long."""
    if not v.any():
        return []
    d = np.diff(np.concatenate(([0], v.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)
    keep = (ends - starts) >= minlen
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


@dataclass
class Rule:
    pos: float                 # y of a horizontal rule / x of a vertical one
    segs: list[tuple[int, int]] = field(default_factory=list)  # covered extent along the rule

    def covers(self, a: float, b: float, frac: float = 0.6) -> bool:
        need = max(1.0, (b - a) * frac)
        have = 0.0
        for s, e in self.segs:
            have += max(0.0, min(b, e) - max(a, s))
        return have >= need


def _group_rules(cands: dict[int, list[tuple[int, int]]], merge: int = 3) -> list[Rule]:
    """Neighbouring pixel rows/cols (a 2-3 px thick rule) become one Rule."""
    groups: list[list[int]] = []
    for p in sorted(cands):
        if groups and p - groups[-1][-1] <= merge:
            groups[-1].append(p)
        else:
            groups.append([p])
    out = []
    for ps in groups:
        segs = sorted(s for p in ps for s in cands[p])
        merged: list[tuple[int, int]] = []
        for s, e in segs:
            if merged and s <= merged[-1][1] + 2:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        out.append(Rule(pos=(min(ps) + max(ps)) / 2, segs=merged))
    return out


def find_h_rules(ink: np.ndarray, region: tuple[int, int, int, int]) -> list[Rule]:
    """Thin horizontal strokes much longer than a word.

    A Devanagari headline (शिरोरेखा) is also a long horizontal stroke, so a
    candidate must be thin AND have clear paper right above or right below
    along most of its length (letters hang below a headline, never both).
    """
    x0, y0, x1, y1 = region
    W = x1 - x0
    # a rule that is a pixel off level (residual skew, aliasing steps) still
    # reads as one run: look at y-1..y+1 together
    m = _close1d(ink | np.roll(ink, 1, axis=0) | np.roll(ink, -1, axis=0), axis=1, gap=2)
    minlen = max(30, int(0.12 * W))
    cands: dict[int, list[tuple[int, int]]] = {}
    H = ink.shape[0]
    for y in range(max(0, y0 - 4), min(H, y1 + 5)):
        segs = _runs(m[y, x0:x1], minlen)
        good = []
        for s, e in segs:
            s += x0; e += x0
            long_full = (e - s) >= 0.5 * W
            # paper a few px above and below the stroke
            above = ink[max(0, y - 6):max(0, y - 3), s:e]
            below = ink[min(H, y + 4):min(H, y + 7), s:e]
            fa = above.mean() if above.size else 0.0
            fb = below.mean() if below.size else 0.0
            if long_full and min(fa, fb) < 0.35:
                good.append((s, e))
            elif fa < 0.18 and fb < 0.18:
                good.append((s, e))
        if good:
            cands[y] = good
    return _group_rules(cands)


def find_v_rules(ink: np.ndarray, region: tuple[int, int, int, int],
                 bands: Optional[list[tuple[float, float]]] = None) -> list[Rule]:
    """Vertical rules. With row bands known, a column rule must cross most of
    a band's height (text strokes don't), else it must be long globally."""
    x0, y0, x1, y1 = region
    Hh = y1 - y0
    m = _close1d(ink | np.roll(ink, 1, axis=1) | np.roll(ink, -1, axis=1), axis=0, gap=2)
    cands: dict[int, list[tuple[int, int]]] = {}
    Wimg = ink.shape[1]
    for x in range(max(0, x0 - 4), min(Wimg, x1 + 5)):
        col = m[:, x]
        good: list[tuple[int, int]] = []
        if bands:
            for a, b in bands:
                ia, ib = int(math.ceil(a)) + 2, int(math.floor(b)) - 2
                if ib - ia < 4:
                    continue
                seg = col[ia:ib]
                if seg.mean() >= 0.9:
                    good.append((int(a), int(b)))
        else:
            for s, e in _runs(col[y0:y1], max(20, int(0.5 * Hh))):
                good.append((s + y0, e + y0))
        if good:
            # thin-ness: paper on both sides
            l = ink[y0:y1, max(0, x - 6):max(0, x - 3)]
            r = ink[y0:y1, min(Wimg, x + 4):min(Wimg, x + 7)]
            if (l.mean() if l.size else 0) < 0.5 or (r.mean() if r.size else 0) < 0.5:
                cands[x] = good
    return _group_rules(cands)


# ---------------------------------------------------------------------------
# Text-line helpers
# ---------------------------------------------------------------------------

@dataclass
class TLine:
    text: str
    box: tuple[float, float, float, float]   # in crop (deskewed) coords
    handwritten: bool = False
    idx: int = -1


def _rows_from_lines(lines: list[TLine]) -> list[tuple[float, float]]:
    if not lines:
        return []
    hs = sorted(l.box[3] - l.box[1] for l in lines)
    tol = max(4.0, hs[len(hs) // 2] * 0.5)
    rows: list[list[TLine]] = []
    for l in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        c = (l.box[1] + l.box[3]) / 2
        if rows:
            r = rows[-1]
            rc = sum((x.box[1] + x.box[3]) / 2 for x in r) / len(r)
            if abs(c - rc) <= tol:
                r.append(l); continue
        rows.append([l])
    return [(min(x.box[1] for x in r), max(x.box[3] for x in r)) for r in rows]


def _cols_from_lines(lines: list[TLine], x0: float, x1: float,
                     row_bands: list[tuple[float, float]]) -> list[float]:
    """Column separators = x ranges no (or almost no) row puts text in."""
    if not lines or x1 - x0 < 4:
        return []
    n = int(math.ceil(x1 - x0)) + 1
    cover = np.zeros(n, dtype=np.int32)
    nrows = max(1, len(row_bands))
    for a, b in row_bands or [(-1e9, 1e9)]:
        occ = np.zeros(n, dtype=bool)
        for l in lines:
            cy = (l.box[1] + l.box[3]) / 2
            if not (a - 1 <= cy <= b + 1):
                continue
            s = max(0, int(l.box[0] - x0)); e = min(n, int(math.ceil(l.box[2] - x0)))
            occ[s:e] = True
        cover += occ
    allowed = max(0, int(0.1 * nrows))
    free = cover <= allowed
    seps: list[float] = []
    for s, e in _runs(free, 8):
        if s == 0 or e >= n:      # margins, not separators
            continue
        seps.append(x0 + (s + e) / 2)
    return seps


# ---------------------------------------------------------------------------
# Adapter output -> boundaries
# ---------------------------------------------------------------------------

def _bounds_from_boxes(boxes: list[Sequence[float]], axis: int) -> list[float]:
    """Separators between consecutive row (axis=1) / column (axis=0) boxes."""
    if len(boxes) < 2:
        return []
    bs = sorted(boxes, key=lambda b: (b[axis] + b[axis + 2]) / 2)
    out = []
    for a, b in zip(bs, bs[1:]):
        out.append((a[axis + 2] + b[axis]) / 2)
    return out


class _HtmlGrid(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict[str, Any]]] = []
        self._cell: Optional[dict[str, Any]] = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "br":
            if self._cell is not None:
                self._cell["text"] += "\n"
            return
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self._cell = {
                "th": tag == "th",
                "hw": "data-handwritten" in a or "handwritten" in (a.get("class") or ""),
                "colspan": max(1, int(a.get("colspan") or 1)),
                "rowspan": max(1, int(a.get("rowspan") or 1)),
                "text": "",
            }
            self.rows[-1].append(self._cell)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"] += data


def html_layout(html: str) -> Optional[dict[str, Any]]:
    """Place HTML cells on a grid: [{row, col, rowspan, colspan, th, text}]."""
    try:
        p = _HtmlGrid(); p.feed(html)
    except Exception:  # noqa: BLE001
        return None
    taken: set[tuple[int, int]] = set()
    cells = []
    ncols = 0
    for r, row in enumerate(p.rows):
        c = 0
        for cell in row:
            while (r, c) in taken:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    taken.add((r + dr, c + dc))
            cells.append({**cell, "row": r, "col": c})
            c += cell["colspan"]
            ncols = max(ncols, c)
    if not cells:
        return None
    for c in cells:
        # whitespace inside a cell collapses (as in a browser); <br> stays a line break
        c["text"] = "\n".join(" ".join(part.split()) for part in c["text"].split("\n")).strip()
    nrows = max(r for r, _ in taken) + 1
    return {"rows": nrows, "cols": ncols, "cells": cells}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    box: tuple[float, float, float, float]   # crop coords
    is_header: bool = False
    text: str = ""
    handwritten: bool = False
    line_ids: list[int] = field(default_factory=list)


@dataclass
class Structure:
    xs: list[float]
    ys: list[float]
    cells: list[Cell]
    header_rows: int
    ruling_style: str
    source: str


def _dedupe(vals: list[float], lo: float, hi: float, min_gap: float) -> list[float]:
    out: list[float] = []
    for v in sorted(vals):
        if v <= lo + min_gap or v >= hi - min_gap:
            continue
        if out and v - out[-1] < min_gap:
            out[-1] = (out[-1] + v) / 2
        else:
            out.append(v)
    return out


def detect(frame: Frame, lines: list[TLine], adapter_raw: Optional[dict[str, Any]] = None) -> Structure:
    ink = _ink(frame.img)
    rx0, ry0, rx1, ry1 = frame.region
    W, H = rx1 - rx0, ry1 - ry0
    raw = adapter_raw or {}

    # ---- rows -------------------------------------------------------
    h_rules = [r for r in find_h_rules(ink, frame.region)
               if ry0 - 6 <= r.pos <= ry1 + 6 and sum(e - s for s, e in r.segs) >= 0.12 * W]
    h_full = [r for r in h_rules if r.covers(rx0, rx1, 0.5)]
    top = min([r.pos for r in h_full], default=ry0)
    bot = max([r.pos for r in h_full], default=ry1)
    # the table body is between the outermost full rules when there are some
    ty0 = top if h_full and top - ry0 < 0.25 * H else ry0
    ty1 = bot if h_full and ry1 - bot < 0.25 * H else ry1

    lines_in = [l for l in lines
                if rx0 - 2 <= (l.box[0] + l.box[2]) / 2 <= rx1 + 2
                and ty0 - 2 <= (l.box[1] + l.box[3]) / 2 <= ty1 + 2]
    med_h = sorted(l.box[3] - l.box[1] for l in lines_in)[len(lines_in) // 2] if lines_in else 20.0
    min_row = max(6.0, 0.6 * med_h)

    row_seps = _dedupe([r.pos for r in h_rules], ty0, ty1, min_row)
    row_src = "rules"
    if not row_seps:
        rb = raw.get("row_boxes") or []
        row_seps = _dedupe(_bounds_from_boxes(rb, 1), ty0, ty1, min_row) if rb else []
        row_src = "model"
    if not row_seps:
        bands = _rows_from_lines(lines_in)
        row_seps = _dedupe([(a[1] + b[0]) / 2 for a, b in zip(bands, bands[1:])], ty0, ty1, min_row)
        row_src = "text"
    ys = [ty0, *row_seps, ty1]
    bands = list(zip(ys, ys[1:]))

    # ---- columns ----------------------------------------------------
    v_rules = find_v_rules(ink, frame.region, bands)
    v_rules = [r for r in v_rules if rx0 - 6 <= r.pos <= rx1 + 6]
    v_major = [r for r in v_rules
               if sum(1 for a, b in bands if r.covers(a, b, 0.7)) >= max(1, int(0.3 * len(bands)))]
    left = min([r.pos for r in v_major], default=rx0)
    right = max([r.pos for r in v_major], default=rx1)
    tx0 = left if v_major and left - rx0 < 0.2 * W else rx0
    tx1 = right if v_major and rx1 - right < 0.2 * W else rx1

    min_col = max(8.0, 0.8 * med_h)
    col_seps = _dedupe([r.pos for r in v_rules], tx0, tx1, min_col)
    col_src = "rules"
    if not col_seps:
        cb = raw.get("col_boxes") or []
        col_seps = _dedupe(_bounds_from_boxes(cb, 0), tx0, tx1, min_col) if cb else []
        col_src = "model"
    if not col_seps:
        col_seps = _dedupe(_cols_from_lines(lines_in, tx0, tx1, bands[1:] if len(bands) > 2 else bands), tx0, tx1, min_col)
        col_src = "text"
    xs = [tx0, *col_seps, tx1]

    nr, nc = len(ys) - 1, len(xs) - 1

    # ---- spans (only where the rules tell us) -----------------------
    v_by_sep = {i: _nearest(v_rules, xs[i]) for i in range(1, nc)}
    h_by_sep = {i: _nearest(h_rules, ys[i]) for i in range(1, nr)}
    ruled = row_src == "rules" and col_src == "rules"

    owner = [[(r, c) for c in range(nc)] for r in range(nr)]
    spans: dict[tuple[int, int], list[int]] = {}

    if ruled:
        # colspans: no rule between (r,c) and (r,c+1)
        for r in range(nr):
            c = 0
            while c < nc:
                c2 = c
                while c2 + 1 < nc:
                    rule = v_by_sep.get(c2 + 1)
                    if rule is not None and rule.covers(ys[r], ys[r + 1], 0.5):
                        break
                    c2 += 1
                spans[(r, c)] = [1, c2 - c + 1]
                for k in range(c + 1, c2 + 1):
                    owner[r][k] = (r, c)
                c = c2 + 1
        # rowspans: same column run below and no rule between them
        for r in range(nr - 1, -1, -1):
            for c in range(nc):
                if owner[r][c] != (r, c):
                    continue
                cs = spans[(r, c)][1]
                r2 = r
                while r2 + 1 < nr and owner[r2 + 1][c] == (r2 + 1, c) and spans.get((r2 + 1, c), [1, 0])[1] == cs:
                    rule = h_by_sep.get(r2 + 1)
                    if rule is not None and rule.covers(xs[c], xs[c + cs], 0.5):
                        break
                    r2 += 1
                if r2 > r:
                    # absorb rows r+1..r2 (they must not have absorbed others)
                    total = 1
                    for rr in range(r + 1, r2 + 1):
                        total += spans.pop((rr, c), [1, cs])[0]
                        for k in range(c, c + cs):
                            owner[rr][k] = (r, c)
                    spans[(r, c)][0] = total
    else:
        for r in range(nr):
            for c in range(nc):
                spans[(r, c)] = [1, 1]

    # ---- adapter HTML: header cells / spans when it fits the grid ----
    th_cells: set[tuple[int, int]] = set()
    hl = html_layout(raw["html"]) if raw.get("html") else None
    if hl and hl["rows"] == nr and hl["cols"] == nc:
        if not ruled:
            spans = {}
            owner = [[(r, c) for c in range(nc)] for r in range(nr)]
            for hc in hl["cells"]:
                spans[(hc["row"], hc["col"])] = [hc["rowspan"], hc["colspan"]]
                for dr in range(hc["rowspan"]):
                    for dc in range(hc["colspan"]):
                        if hc["row"] + dr < nr and hc["col"] + dc < nc:
                            owner[hc["row"] + dr][hc["col"] + dc] = (hc["row"], hc["col"])
        th_cells = {(hc["row"], hc["col"]) for hc in hl["cells"] if hc["th"]}

    cells: list[Cell] = []
    for (r, c), (rs, cs) in sorted(spans.items()):
        rs = min(rs, nr - r); cs = min(cs, nc - c)
        cells.append(Cell(row=r, col=c, rowspan=rs, colspan=cs,
                          box=(xs[c], ys[r], xs[c + cs], ys[r + rs])))

    # ---- header rows ------------------------------------------------
    if th_cells:
        hr = 0
        for r in range(nr):
            if all((cc.row, cc.col) in th_cells for cc in cells if cc.row == r):
                hr = r + 1
            else:
                break
    else:
        hr = 1 if nr >= 2 else 0
        if hr:
            hr = max([cc.rowspan for cc in cells if cc.row == 0], default=1)
            hr = min(hr, nr - 1) if nr > 1 else hr
    for cc in cells:
        cc.is_header = cc.row < hr or (cc.row, cc.col) in th_cells

    # ---- ruling style -----------------------------------------------
    has_h = bool(h_rules); has_v = bool(v_rules)
    if has_h and has_v:
        style = "full" if col_src == "rules" and row_src == "rules" and all(
            v_by_sep.get(i) is not None for i in range(1, nc)) else "partial"
    elif has_h:
        style = "horizontal"
    elif has_v:
        style = "partial"
    else:
        style = "none"

    _fill_text(cells, lines_in, ink, xs)
    return Structure(xs=xs, ys=ys, cells=cells, header_rows=hr, ruling_style=style,
                     source=f"rows:{row_src} cols:{col_src}")


def _nearest(rules: list[Rule], pos: float, tol: float = 6.0) -> Optional[Rule]:
    best = None
    for r in rules:
        if abs(r.pos - pos) <= tol and (best is None or abs(r.pos - pos) < abs(best.pos - pos)):
            best = r
    return best


def _ov(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _word_slots(l: "TLine", ink: Optional[np.ndarray], seps: list[float]) -> list[float]:
    """x centres of the ink 'words' inside a line box (rules masked out)."""
    if ink is None:
        return []
    x0, y0, x1, y1 = [int(round(v)) for v in l.box]
    h = max(1, y1 - y0)
    ya, yb = y0 + int(0.15 * h), y1 - int(0.15 * h)
    x0 = max(0, x0); x1 = min(ink.shape[1], x1)
    if yb <= ya or x1 <= x0:
        return []
    prof = ink[max(0, ya):yb, x0:x1].any(axis=0)
    for sx in seps:
        k = int(round(sx)) - x0
        prof[max(0, k - 3):max(0, k + 4)] = False
    gap = max(2, int(0.3 * h))
    runs = _runs(prof, 1)
    clusters: list[list[int]] = []
    for s_, e_ in runs:
        if clusters and s_ - clusters[-1][1] <= gap:
            clusters[-1][1] = e_
        else:
            clusters.append([s_, e_])
    return [x0 + (s_ + e_) / 2 for s_, e_ in clusters if e_ - s_ >= 2]


def _fill_text(cells: list[Cell], lines: list[TLine], ink: Optional[np.ndarray] = None,
               col_xs: Optional[list[float]] = None) -> None:
    """Every line goes to ONE cell (most overlap). A line that clearly runs
    across several cells (page OCR read a whole row as one line) is split
    word by word, using where the ink actually is, instead of being copied
    into each cell."""
    parts: dict[int, list[tuple[float, float, str, TLine]]] = {i: [] for i in range(len(cells))}
    seps = list(col_xs or [])
    for l in lines:
        la = max(1.0, (l.box[2] - l.box[0]) * (l.box[3] - l.box[1]))
        hits = [(i, _ov(l.box, c.box)) for i, c in enumerate(cells)]
        hits = [(i, o) for i, o in hits if o > 0.05 * la]
        if not hits:
            continue
        big = [i for i, o in hits if o >= 0.15 * la]
        words = l.text.split()
        cyy = (l.box[1] + l.box[3]) / 2

        def cell_at(x: float) -> int:
            inside = [i for i, _ in hits if cells[i].box[0] <= x <= cells[i].box[2]
                      and cells[i].box[1] - 2 <= cyy <= cells[i].box[3] + 2]
            return inside[0] if inside else max(hits, key=lambda t: t[1])[0]

        if len(big) >= 2 and len(words) >= 2:
            slots = _word_slots(l, ink, seps)
            targets: list[tuple[int, float]] = []
            if len(slots) == len(words):
                targets = [(cell_at(x), x) for x in slots]
            elif slots:
                # fewer/more ink blobs than words: share the words between the
                # cells that have ink, by how much ink each has
                per: dict[int, list[float]] = {}
                for x in slots:
                    per.setdefault(cell_at(x), []).append(x)
                order = sorted(per, key=lambda i: cells[i].box[0])
                if len(order) <= len(words):
                    weights = [len(per[i]) for i in order]
                    tot = sum(weights)
                    k = 0
                    for j, i in enumerate(order):
                        n = len(words) - k - (len(order) - j - 1) if j == len(order) - 1 else max(1, round(len(words) * weights[j] / tot))
                        n = min(n, len(words) - k - (len(order) - j - 1))
                        for w_ in range(n):
                            targets.append((i, per[i][0] + w_))
                        k += n
            if len(targets) != len(words):
                # no usable ink: spread words over the line by character count
                total = sum(len(w) for w in words) + len(words) - 1
                pos = 0
                lw = l.box[2] - l.box[0]
                targets = []
                for w in words:
                    cx = l.box[0] + lw * (pos + len(w) / 2) / max(1, total)
                    pos += len(w) + 1
                    targets.append((cell_at(cx), cx))
            for w, (i, x) in zip(words, targets):
                parts[i].append((l.box[1], x, w, l))
        else:
            i = max(hits, key=lambda t: t[1])[0]
            parts[i].append((l.box[1], l.box[0], l.text, l))
    for i, c in enumerate(cells):
        ps = parts[i]
        if not ps:
            continue
        # top-to-bottom rows of text, left-to-right inside a row
        hs = sorted(p[3].box[3] - p[3].box[1] for p in ps)
        tol = max(4.0, hs[len(hs) // 2] / 2)
        ps.sort(key=lambda p: p[0])
        rows: list[list[tuple[float, float, str, TLine]]] = []
        for p in ps:
            if rows and abs(p[0] - rows[-1][0][0]) <= tol:
                rows[-1].append(p)
            else:
                rows.append([p])
        c.text = " ".join(" ".join(p[2] for p in sorted(r, key=lambda p: p[1])) for r in rows).strip()
        c.handwritten = any(p[3].handwritten for p in ps)
        c.line_ids = sorted({p[3].idx for p in ps})


# ---------------------------------------------------------------------------
# Fit a typed structure (HTML) to the image
# ---------------------------------------------------------------------------

def _pick(n_edges: int, cands: list[float], score) -> list[Optional[int]]:
    """Assign edges 1..n_edges to increasing candidate indices (or None =
    interpolate) maximising the summed score. Small DP, fine for tables."""
    m = len(cands)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def f(k: int, i: int) -> tuple[float, tuple]:
        if k > n_edges:
            return 0.0, ()
        best = f(k + 1, i)
        best = (best[0], (None,) + best[1])
        # leave room: later edges may still need candidates, but None is allowed
        for j in range(i, m):
            s = score(k, j)
            if s <= 0:
                continue
            rest = f(k + 1, j + 1)
            if s + rest[0] > best[0]:
                best = (s + rest[0], (j,) + rest[1])
        return best

    return list(f(1, 0)[1])


def _fill_bounds(lo: float, hi: float, picked: list[Optional[float]],
                 weights: Optional[list[float]] = None) -> list[float]:
    """[lo, e1, …, hi]; missing edges are spread between their known
    neighbours (in proportion to `weights` per slot when given)."""
    n = len(picked) + 1
    w = weights or [1.0] * n
    pts: list[Optional[float]] = [lo, *picked, hi]
    i = 0
    while i < len(pts) - 1:
        j = i + 1
        while pts[j] is None:
            j += 1
        if j > i + 1:
            a, b = pts[i], pts[j]
            tot = sum(w[i:j]) or 1.0
            acc = 0.0
            for k in range(i + 1, j):
                acc += w[k - 1]
                pts[k] = a + (b - a) * acc / tot
        i = j
    return [float(p) for p in pts]  # type: ignore[arg-type]


def fit_html(frame: Frame, lines: list[TLine], html: str) -> Optional[Structure]:
    """Use the rows / columns / spans / text the annotator typed, and place
    the grid on the ruling lines (or text rows) found in the image."""
    hl = html_layout(html)
    if not hl:
        return None
    nr, nc = hl["rows"], hl["cols"]
    cells_h = hl["cells"]

    ink = _ink(frame.img)
    rx0, ry0, rx1, ry1 = frame.region
    W, H = rx1 - rx0, ry1 - ry0

    h_rules = [r for r in find_h_rules(ink, frame.region)
               if ry0 - 6 <= r.pos <= ry1 + 6 and sum(e - s for s, e in r.segs) >= 0.12 * W]
    h_full = [r for r in h_rules if r.covers(rx0, rx1, 0.5)]
    top = min([r.pos for r in h_full], default=ry0)
    bot = max([r.pos for r in h_full], default=ry1)
    ty0 = top if h_full and top - ry0 < 0.25 * H else ry0
    ty1 = bot if h_full and ry1 - bot < 0.25 * H else ry1

    lines_in = [l for l in lines
                if rx0 - 2 <= (l.box[0] + l.box[2]) / 2 <= rx1 + 2
                and ty0 - 2 <= (l.box[1] + l.box[3]) / 2 <= ty1 + 2]
    med_h = sorted(l.box[3] - l.box[1] for l in lines_in)[len(lines_in) // 2] if lines_in else 20.0

    # ---- rows: interior horizontal rules, else gaps between text rows ----
    row_c = _dedupe([r.pos for r in h_rules], ty0, ty1, max(6.0, 0.5 * med_h))
    row_rule = {i: _nearest(h_rules, y) for i, y in enumerate(row_c)}
    if len(row_c) < nr - 1:
        bands_t = _rows_from_lines(lines_in)
        extra = [(a[1] + b[0]) / 2 for a, b in zip(bands_t, bands_t[1:])]
        extra = [y for y in extra if all(abs(y - c) > 0.5 * med_h for c in row_c)]
        row_c = sorted(row_c + extra)
        row_rule = {i: _nearest(h_rules, y) for i, y in enumerate(row_c)}

    def row_score(k: int, j: int) -> float:
        rule = row_rule.get(j)
        # cells of the typed table that END at row edge k need a rule there
        need = [c for c in cells_h if c["row"] + c["rowspan"] == k]
        span = [c for c in cells_h if c["row"] < k < c["row"] + c["rowspan"]]
        if rule is None:
            return 0.3 if not span else 0.05      # a text gap: weak evidence
        cov = sum(e - s for s, e in rule.segs) / max(1.0, W)
        return (1.0 if need else 0.2) * min(1.0, cov + 0.1) - (0.6 if span and cov > 0.8 else 0.0)

    picked_r = _pick(nr - 1, row_c, row_score)
    ys = _fill_bounds(ty0, ty1, [row_c[j] if j is not None else None for j in picked_r])
    bands = list(zip(ys, ys[1:]))

    # ---- columns: vertical rules, judged row by row against the spans ----
    v_rules = [r for r in find_v_rules(ink, frame.region, bands) if rx0 - 6 <= r.pos <= rx1 + 6]
    v_major = [r for r in v_rules if sum(1 for a, b in bands if r.covers(a, b, 0.7)) >= 1]
    left = min([r.pos for r in v_major], default=rx0)
    right = max([r.pos for r in v_major], default=rx1)
    tx0 = left if v_major and left - rx0 < 0.2 * W else rx0
    tx1 = right if v_major and rx1 - right < 0.2 * W else rx1
    col_c = _dedupe([r.pos for r in v_rules], tx0, tx1, max(8.0, 0.8 * med_h))
    col_rule = [_nearest(v_rules, x) for x in col_c]

    def col_score(k: int, j: int) -> float:
        rule = col_rule[j]
        if rule is None:
            return 0.0
        s = 0.0
        for r, (a, b) in enumerate(bands):
            row_cells = [c for c in cells_h if c["row"] <= r < c["row"] + c["rowspan"]]
            edge = any(c["col"] == k for c in row_cells) or any(c["col"] + c["colspan"] == k for c in row_cells)
            inside = any(c["col"] < k < c["col"] + c["colspan"] for c in row_cells)
            has = rule.covers(a, b, 0.6)
            if edge and has:
                s += 1.0
            elif inside and has:
                s -= 1.0
        return s

    picked_c = _pick(nc - 1, col_c, col_score)
    xs = _fill_bounds(tx0, tx1, [col_c[j] if j is not None else None for j in picked_c])

    # ---- cells: typed text, typed spans, typed headers ----
    cells: list[Cell] = []
    for hc in cells_h:
        r, c = hc["row"], hc["col"]
        rs = min(hc["rowspan"], nr - r); cs = min(hc["colspan"], nc - c)
        cells.append(Cell(row=r, col=c, rowspan=rs, colspan=cs,
                          box=(xs[c], ys[r], xs[c + cs], ys[r + rs]),
                          is_header=hc["th"], text=hc["text"], handwritten=hc.get("hw", False)))
    hr = 0
    for r in range(nr):
        rc = [x for x in cells if x.row == r]
        if rc and all(x.is_header for x in rc):
            hr = r + 1
        else:
            break

    has_h, has_v = bool(h_rules), bool(v_rules)
    style = ("full" if has_h and has_v and all(j is not None for j in picked_c) else
             "partial" if has_h and has_v else
             "horizontal" if has_h else "partial" if has_v else "none")
    return Structure(xs=xs, ys=ys, cells=cells, header_rows=hr, ruling_style=style,
                     source="html fitted to image")
