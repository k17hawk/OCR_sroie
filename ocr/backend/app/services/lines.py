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

import re
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from app.services.geometry import bbox_to_quad

TEXT_TYPES = frozenset({
    "paragraph", "header", "title", "heading", "subheading", "list",
    "form_field", "table_heading", "footer", "caption",
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
        return None, (f"block {rid}{what}: {len(texts)} text lines but {len(found)} printed lines"
                      " — add line breaks; lines left unchanged")
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
