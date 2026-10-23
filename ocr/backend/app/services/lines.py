"""Text lines measured from the page image.

Line boxes are never placed by default (even strips, the block's width):
every box comes from the ink on the scan.

``detect_lines(gray, bbox)`` makes two passes over the text pixels inside a
box: a row projection finds each printed line (top, bottom), then a column
projection inside THAT line's rows gives the line's own left / right edge.
``ink_mask`` decides what counts as text: adaptive threshold (faded print
still counts), minus long horizontal / vertical strokes (underlines, table
borders, page frames) and specks.

``detect_page_lines`` runs that over the text regions of a v2 page and
matches the boxes, in order, to each region's text lines. When the counts
differ it warns and leaves the region's lines alone instead of guessing.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from app.services.geometry import bbox_to_quad

TEXT_TYPES = frozenset({
    "paragraph", "header", "title", "heading", "subheading", "list", "sublist",
    "form_field", "table_heading", "footer", "page_number", "caption",
})

# keys of an old line worth carrying over to the new line at the same index
# "id" too: fields reference lines by id, so a re-detected line keeps it
_CARRY = ("id", "font_id", "role", "handwritten", "list_level", "item_index")

_HEADING_MARK = re.compile(r"^\s*#+\s*")


# ---------------------------------------------------------------------------
# ink -> printed lines
# ---------------------------------------------------------------------------

def runs(mask: np.ndarray, gap: int = 3) -> list[tuple[int, int]]:
    """(start, end) index pairs of True runs; runs closer than ``gap`` merge."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out = []
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - p > gap:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


def ink_mask(gray_crop: np.ndarray, min_area: int = 12) -> np.ndarray:
    """Text pixels = 1. Removes long ruling lines, underlines and specks."""
    if gray_crop.size == 0:
        return np.zeros(gray_crop.shape, np.uint8)
    crop = np.ascontiguousarray(gray_crop, dtype=np.uint8)
    b = cv2.adaptiveThreshold(255 - crop, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 31, -10)
    h, w = b.shape
    # long horizontal / vertical strokes = rules, underlines, table borders -> not text
    hor = cv2.morphologyEx(b, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, w // 3), 1)))
    ver = cv2.morphologyEx(b, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, h // 2))))
    b = b & (1 - hor) & (1 - ver)
    # drop specks (scan noise) smaller than a dot of text
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab].astype(np.uint8)


def _crop_box(gray: np.ndarray, bbox: Iterable[float], inset: int = 0) -> Optional[tuple[int, int, int, int]]:
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x0 + inset), max(0, y0 + inset)
    x1, y1 = min(W, x1 - inset), min(H, y1 - inset)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _clip(box: list[int], to: Iterable[float]) -> list[int]:
    cx0, cy0, cx1, cy1 = [int(round(v)) for v in to]
    return [max(box[0], cx0), max(box[1], cy0), min(box[2], cx1), min(box[3], cy1)]


def detect_lines(gray: np.ndarray, bbox: Iterable[float], min_h: int = 12, row_gap: int = 4,
                 pad: int = 6, inset: int = 4) -> list[list[int]]:
    """Printed-line boxes inside ``bbox``, top to bottom, each with its own x-extent.

    Pass 1: rows with ink -> groups of rows (gaps <= ``row_gap`` bridged) =
    one printed line; groups shorter than ``min_h`` are dropped. Pass 2:
    columns with ink inside that line's rows -> its left / right edge.
    Boxes are padded by ``pad`` and stay inside ``bbox``.
    """
    bbox = [int(round(v)) for v in bbox]
    c = _crop_box(gray, bbox, inset)
    if c is None:
        return []
    x0, y0, x1, y1 = c
    m = ink_mask(gray[y0:y1, x0:x1])
    lines = []
    for a, b in runs(m.sum(1) > 1, row_gap):              # pass 1: rows
        if b - a + 1 < min_h:
            continue
        cols = np.flatnonzero(m[a:b + 1].any(0))           # pass 2: columns of THIS line
        box = [x0 + int(cols.min()) - pad, y0 + a - pad,
               x0 + int(cols.max()) + 1 + pad, y0 + b + 1 + pad]
        lines.append(_clip(box, bbox))
    return lines


def split_row(gray: np.ndarray, row: Iterable[float], k: int, region: Iterable[float],
              pad: int = 6, inset: int = 4) -> Optional[list[list[int]]]:
    """Split ONE printed row into ``k`` side-by-side pieces, left to right, at
    its ``k - 1`` widest blank gaps (the fill-in areas between a label and the
    next printed word). Returns None unless those gaps are clearly wider than
    every other gap in the row (at least one row height, and twice the widest
    gap left over) — otherwise which word belongs to which piece is a guess.

    Dot leaders don't close a gap: ink pieces smaller than a third of the
    row height in both directions are ignored when looking for gaps. Each
    piece's box is measured from its own ink.
    """
    if k < 2:
        return None
    c = _crop_box(gray, row, 0)
    if c is None:
        return None
    x0, y0, x1, y1 = c
    m = ink_mask(gray[y0:y1, x0:x1])
    row_h = y1 - y0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = max(3, row_h // 3)
    big = np.zeros(n, bool)
    big[1:] = (stats[1:, cv2.CC_STAT_WIDTH] > small) | (stats[1:, cv2.CC_STAT_HEIGHT] > small)
    cols = big[lab].any(0)
    ink_runs = runs(cols, 1)
    if len(ink_runs) < k:
        return None
    gaps = sorted(((ink_runs[i + 1][0] - ink_runs[i][1] - 1, i) for i in range(len(ink_runs) - 1)),
                  reverse=True)
    chosen, rest = gaps[:k - 1], gaps[k - 1:]
    widest_rest = rest[0][0] if rest else 0
    if chosen[-1][0] < row_h or chosen[-1][0] < 2 * widest_rest:
        return None
    cuts = sorted(i for _, i in chosen)
    near = max(2, row_h // 3)  # a colon / dot / matra next to a word belongs to it
    pieces, start = [], 0
    for i in cuts + [len(ink_runs) - 1]:
        a, b = ink_runs[start][0], ink_runs[i][1]
        for j in range(1, n):
            if big[j]:
                continue
            cx0 = int(stats[j, cv2.CC_STAT_LEFT])
            cx1 = cx0 + int(stats[j, cv2.CC_STAT_WIDTH]) - 1
            if cx1 >= ink_runs[start][0] - near and cx0 <= ink_runs[i][1] + near:
                a, b = min(a, cx0), max(b, cx1)
        ys = np.flatnonzero(m[:, a:b + 1].any(1))
        box = [x0 + a - pad, y0 + int(ys.min()) - pad, x0 + b + 1 + pad, y0 + int(ys.max()) + 1 + pad]
        pieces.append([int(v) for v in _clip(box, region)])  # plain ints: the box goes into JSON
        start = i + 1
    return pieces


def snap_box_to_ink(gray: np.ndarray, bbox: Iterable[float], pad: int = 6,
                    clip_to: Optional[Iterable[float]] = None) -> list[int]:
    """Tighten a roughly drawn box to the ink inside it (snap after drawing).

    Returns the box unchanged when it holds no text ink. ``clip_to`` (the
    parent region) keeps the result inside it.
    """
    bbox = [int(round(v)) for v in bbox]
    c = _crop_box(gray, bbox)
    if c is None:
        return bbox
    x0, y0, x1, y1 = c
    ys, xs = np.nonzero(ink_mask(gray[y0:y1, x0:x1]))
    if xs.size == 0:
        return bbox
    H, W = gray.shape[:2]
    box = [x0 + int(xs.min()) - pad, y0 + int(ys.min()) - pad,
           x0 + int(xs.max()) + 1 + pad, y0 + int(ys.max()) + 1 + pad]
    box = _clip(box, [0, 0, W, H])
    if clip_to is not None:
        box = _clip(box, clip_to)
    return box if box[2] > box[0] and box[3] > box[1] else bbox


# below this the text counts as straight and the snapped box stays a rectangle
MIN_TILT_DEG = 0.5
MAX_TILT_DEG = 20.0


def ink_angle(xs: np.ndarray, ys: np.ndarray, max_deg: float = MAX_TILT_DEG) -> float:
    """Tilt of the text line(s) made of these ink pixels, in degrees.

    Image coordinates (y down): a line rising to the right gives a negative
    angle. Projection profile: the angle at which the ink, projected across
    the line, piles up into the sharpest rows (largest sum of squared row
    counts). Coarse 0.5 degree search, then 0.05 degree around the best.
    """
    if xs.size < 20:
        return 0.0
    if xs.size > 20000:  # plenty to measure an angle; keeps snap-on-mouse-up quick
        step = int(math.ceil(xs.size / 20000))
        xs, ys = xs[::step], ys[::step]
    x = xs.astype(np.float64) - xs.mean()
    y = ys.astype(np.float64) - ys.mean()

    def score(deg: float) -> float:
        t = math.radians(deg)
        v = -x * math.sin(t) + y * math.cos(t)
        h = np.bincount(np.round(v - v.min()).astype(np.int64))
        return float(np.dot(h, h))

    best = max(np.arange(-max_deg, max_deg + 1e-9, 0.5), key=score)
    best = max(np.arange(best - 0.5, best + 0.5 + 1e-9, 0.05), key=score)
    return float(best)


def snap_box_to_ink_tilted(gray: np.ndarray, bbox: Iterable[float], pad: int = 6,
                           clip_to: Optional[Iterable[float]] = None,
                           ) -> tuple[list[int], Optional[list[list[float]]]]:
    """Like ``snap_box_to_ink``, but follows tilted text.

    Returns ``(bbox, quad)``. When the ink inside the drawn box is a line
    tilted by at least ``MIN_TILT_DEG``, ``quad`` is a tight rotated
    rectangle around it (TL, TR, BR, BL, padded by ``pad`` along its own
    axes) and ``bbox`` its outer bounds. Otherwise ``quad`` is None and
    ``bbox`` is exactly what ``snap_box_to_ink`` returns.
    """
    straight = snap_box_to_ink(gray, bbox, pad=pad, clip_to=clip_to)
    drawn = [int(round(v)) for v in bbox]
    c = _crop_box(gray, drawn)
    if c is None:
        return straight, None
    x0, y0, x1, y1 = c
    ys, xs = np.nonzero(ink_mask(gray[y0:y1, x0:x1]))
    if xs.size < 20:
        return straight, None
    deg = ink_angle(xs, ys)
    if abs(deg) < MIN_TILT_DEG:
        return straight, None

    t = math.radians(deg)
    cos, sin = math.cos(t), math.sin(t)
    px, py = xs.astype(np.float64) + x0, ys.astype(np.float64) + y0
    u = px * cos + py * sin          # along the line
    v = -px * sin + py * cos         # across it
    umin, umax = u.min() - pad, u.max() + 1 + pad
    vmin, vmax = v.min() - pad, v.max() + 1 + pad
    if (umax - umin) < 1.5 * (vmax - vmin):
        return straight, None        # a word or two: too short to trust the angle

    H, W = gray.shape[:2]
    lo = [0.0, 0.0, float(W), float(H)]
    cl = [float(v) for v in clip_to] if clip_to is not None else lo
    lim = [max(lo[0], cl[0]), max(lo[1], cl[1]), min(lo[2], cl[2]), min(lo[3], cl[3])]
    quad: list[list[float]] = []
    for uu, vv in ((umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax)):  # TL, TR, BR, BL
        qx = min(max(uu * cos - vv * sin, lim[0]), lim[2])
        qy = min(max(uu * sin + vv * cos, lim[1]), lim[3])
        quad.append([round(qx, 1), round(qy, 1)])
    qxs, qys = [p[0] for p in quad], [p[1] for p in quad]
    box = [int(math.floor(min(qxs))), int(math.floor(min(qys))),
           int(math.ceil(max(qxs))), int(math.ceil(max(qys)))]
    if box[2] <= box[0] or box[3] <= box[1]:
        return straight, None
    return box, quad


# ---------------------------------------------------------------------------
# region -> text lines
# ---------------------------------------------------------------------------

def _is_table_line(ln: dict[str, Any]) -> bool:
    return ln.get("table_index") is not None or ln.get("cell_id") is not None


def _own_lines(rec: dict[str, Any], rid: int) -> list[dict[str, Any]]:
    mine = [ln for ln in rec.get("lines") or []
            if isinstance(ln, dict) and ln.get("block_id") == rid and not _is_table_line(ln)]
    return sorted(mine, key=lambda ln: ln.get("line_in_block", 0))


def _texts(el: dict[str, Any], old: list[dict[str, Any]]) -> list[str]:
    """The region's text, one entry per line.

    The region ``text`` is split at "\\n" (finalize saves every region's
    text as its lines joined with "\\n"). A one-line text falls back to the
    region's current lines, which is what the editor's text box shows.
    """
    raw = el.get("text") or ""
    if "\n" in raw:
        parts = raw.split("\n")
    elif [ln for ln in old if (ln.get("text") or "").strip()]:
        parts = [ln.get("text") or "" for ln in old]
    else:
        parts = [raw]
    if parts and el.get("type") == "heading":
        parts[0] = _HEADING_MARK.sub("", parts[0])
    return [p.strip() for p in parts if p.strip()]


def _line(text: str, box: list[int], rid: int, idx: int, is_heading: bool,
          carry: Optional[dict[str, Any]] = None, role: Optional[str] = None) -> dict[str, Any]:
    ln: dict[str, Any] = {
        "text": text, "bbox": box, "block_id": rid, "line_in_block": idx,
        "is_heading": is_heading,
        "font_size": round(0.8 * (box[3] - box[1]), 1),
        "font_id": None,
    }
    for k in _CARRY:
        if carry and carry.get(k) is not None:
            ln[k] = carry[k]
    if role:
        ln["role"] = role
    ln["quad"] = bbox_to_quad(box)
    return ln


def _carry_for(old: list[dict[str, Any]], texts: list[str], i: int) -> Optional[dict[str, Any]]:
    """The old line whose id / role / font / handwriting the i-th new line
    keeps: same position when the count is unchanged, else the old line
    with the same text."""
    if len(old) == len(texts):
        return old[i]
    want = texts[i].strip()
    return next((ln for ln in old if (ln.get("text") or "").strip() == want), None)


def _match(gray, el, box, texts, old, start, role=None, what="") -> tuple[Optional[list[dict]], Optional[str]]:
    """One line per printed line inside ``box``, or ``(None, warning)`` when the counts differ."""
    rid = el["id"]
    if not texts:
        return [], None
    found = detect_lines(gray, box)
    if len(found) != len(texts):
        more_rows = len(texts) > len(found)
        # lines drawn side by side on one printed row (label / option words)
        # that already match the text: they are right as they are
        if more_rows and [(ln.get("text") or "").strip() for ln in old] == texts:
            return list(old), None
        # one printed row holding several text rows: split it at its blank gaps
        if more_rows and len(found) == 1:
            pieces = split_row(gray, found[0], len(texts), box)
            if pieces is not None:
                found = pieces
    if len(found) != len(texts):
        hint = ("no printed text found inside the box — check the box" if not found else
                "add line breaks" if len(texts) < len(found) else
                "rows that share one printed line: join them, or draw each with the Line tool")
        return None, (f"block {rid}{what}: {len(texts)} text lines but {len(found)} printed lines"
                      f" — {hint}; lines left unchanged")
    heading = el.get("type") == "heading"
    return [_line(t, b, rid, start + i, heading, carry=_carry_for(old, texts, i), role=role)
            for i, (t, b) in enumerate(zip(texts, found))], None


def detect_page_lines(
    gray: np.ndarray, rec: dict[str, Any], region_ids: Optional[set[int]] = None,
) -> tuple[dict[str, Any], list[str]]:
    """Rebuild ``rec["lines"]`` for the text regions (all, or ``region_ids``).

    Returns ``(rec, warnings)``; ``rec`` is changed in place. A region whose
    text and printed lines don't match keeps its lines and gets a warning.
    """
    warnings: list[str] = []
    for el in rec.get("layout") or []:
        if not isinstance(el, dict) or el.get("type") not in TEXT_TYPES:
            continue
        rid = el.get("id")
        if region_ids is not None and rid not in region_ids:
            continue
        if not el.get("bbox"):
            continue
        old = _own_lines(rec, rid)

        # every region type, form_field included: lines come from the
        # region's own printed text (fields only point at lines, never back)
        texts = _texts(el, old)
        if not texts:
            continue
        new, w = _match(gray, el, el["bbox"], texts, old, 0)
        if w:
            warnings.append(w)
        if new is None:
            continue

        keep = [ln for ln in rec.get("lines") or []
                if not (isinstance(ln, dict) and ln.get("block_id") == rid and not _is_table_line(ln))]
        rec["lines"] = keep + new
    return rec, warnings
