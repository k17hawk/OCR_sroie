"""Post-processing heuristics for raw adapter output.

Every function here is pure. Called by ``pipeline/preannotate.py`` after
adapters run and before ``core.finalize`` is applied.
"""
from __future__ import annotations

import re
from typing import Any

from app.schemas.doc_v2 import (
    FormFieldEntry,
    FormFieldLabel,
    Line,
    NormalFieldValue,
)


# ---------------------------------------------------------------------------
# 1. Line -> layout region assignment
# ---------------------------------------------------------------------------

def _area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _intersect_area(a: list[int], b: list[int]) -> int:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def assign_lines_to_regions(
    lines: list[Line], elements: list[Any]
) -> list[int]:
    """Assign each line to the region with highest overlap.

    Returns the indices of orphan lines (no region overlap).
    Mutates ``line.block_id`` in place.
    """
    orphans: list[int] = []
    for i, ln in enumerate(lines):
        best_id, best_area = None, 0
        for el in elements:
            a = _intersect_area(ln.bbox, el.bbox)
            if a > best_area:
                best_area = a
                best_id = el.id
        if best_id is None:
            orphans.append(i)
        else:
            ln.block_id = best_id

    # Re-number line_in_block after assignment.
    counters: dict[int, int] = {}
    for ln in lines:
        n = counters.get(ln.block_id, 0)
        ln.line_in_block = n
        counters[ln.block_id] = n + 1
    return orphans


# ---------------------------------------------------------------------------
# 2. Order lines within each block (rotation-aware, top-to-bottom, left-to-right)
# ---------------------------------------------------------------------------

def order_lines_in_blocks(lines: list[Line], row_tolerance: int = 12) -> list[Line]:
    """Sort by coarse row (quantized), then left-to-right. Preserves block grouping."""
    by_block: dict[int, list[Line]] = {}
    for ln in lines:
        by_block.setdefault(ln.block_id, []).append(ln)

    out: list[Line] = []
    for block_id in sorted(by_block):
        bucket = by_block[block_id]

        def key(ln: Line) -> tuple[int, float]:
            cy = (ln.bbox[1] + ln.bbox[3]) / 2
            cx = (ln.bbox[0] + ln.bbox[2]) / 2
            return (int(cy // row_tolerance), cx)

        bucket.sort(key=key)
        for i, ln in enumerate(bucket):
            ln.line_in_block = i
        out.extend(bucket)
    return out


# ---------------------------------------------------------------------------
# 3. Initial reading order (XY-cut, locked elements stay null)
# ---------------------------------------------------------------------------

READING_ORDER_LOCKED = frozenset({"logo", "signature", "stamp"})


def initial_reading_order(elements: list[Any]) -> None:
    """Assign ``reading_order`` in place. Locked element types stay ``None``."""
    candidates = [el for el in elements if el.type not in READING_ORDER_LOCKED]
    candidates.sort(key=lambda el: (el.bbox[1], el.bbox[0]))
    for i, el in enumerate(candidates):
        el.reading_order = i
    for el in elements:
        if el.type in READING_ORDER_LOCKED:
            el.reading_order = None


# ---------------------------------------------------------------------------
# 4. Form-field label -> value pairing
# ---------------------------------------------------------------------------

_LABEL_END = re.compile(r"[:\u0903]\s*$")
_RE_CITIZENSHIP = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{5}$")
_RE_PHONE = re.compile(r"^9[78]\d{8}$")
_RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_RE_BS_DATE = re.compile(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$")
_RE_MONEY = re.compile(r"^(रु\.?|Rs\.?|NPR)\s?[\d,]+")


def guess_field_type(text: str) -> str:
    """Regex-driven ``field_type`` guess. Falls back to ``"text"``."""
    t = text.strip()
    if _RE_CITIZENSHIP.match(t):
        return "citizenship"
    if _RE_PHONE.match(t):
        return "phone"
    if _RE_EMAIL.match(t):
        return "email"
    if _RE_BS_DATE.match(t):
        return "date"
    if _RE_MONEY.match(t):
        return "money"
    if t.isdigit() and len(t) >= 8:
        return "account_number"
    return "text"


def pair_form_fields(lines: list[Line]) -> list[FormFieldEntry]:
    """Pair each label line with the nearest value line on its right or below."""
    entries: list[FormFieldEntry] = []
    used: set[int] = set()

    for i, ln in enumerate(lines):
        if i in used or not _LABEL_END.search(ln.text):
            continue

        lx0, ly0, lx1, ly1 = ln.bbox
        lcy = (ly0 + ly1) / 2
        lheight = max(1, ly1 - ly0)

        best_j, best_d = None, float("inf")
        for j, other in enumerate(lines):
            if j == i or j in used:
                continue
            ox0, oy0, ox1, oy1 = other.bbox
            ocy = (oy0 + oy1) / 2
            # Same row band, to the right
            if abs(ocy - lcy) <= lheight and ox0 > lx1:
                d = ox0 - lx1
                if d < best_d:
                    best_d, best_j = d, j
            # Or directly below within the label's column
            elif oy0 > ly1 and ox0 >= lx0 - 10 and ox1 <= lx1 + 200:
                d = oy0 - ly1
                if d < best_d:
                    best_d, best_j = d, j

        if best_j is None:
            entries.append(FormFieldEntry(
                label=FormFieldLabel(text=ln.text, bbox=ln.bbox, quad=ln.quad),
                field_type="text",
                value=None,
            ))
            used.add(i)
        else:
            used.add(i)
            used.add(best_j)
            v = lines[best_j]
            entries.append(FormFieldEntry(
                label=FormFieldLabel(text=ln.text, bbox=ln.bbox, quad=ln.quad),
                field_type=guess_field_type(v.text),
                value=NormalFieldValue(
                    text=v.text, bbox=v.bbox, quad=v.quad, handwritten=True
                ),
            ))
    return entries


# ---------------------------------------------------------------------------
# 5. List marker detection
# ---------------------------------------------------------------------------

_LIST_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\d+[.)]\s"), "latin"),
    (re.compile(r"^[०-९]+[.)]\s"), "dev"),
    (re.compile(r"^[क-ह][).]\s"), "dev"),
    (re.compile(r"^[a-zA-Z][).]\s"), "latin"),
    (re.compile(r"^[ivxIVX]+[.)]\s"), "roman"),
    (re.compile(r"^[•●◦▪‣·\-*]\s"), "bullet"),
]


def detect_list_marker(text: str) -> tuple[str, str] | None:
    """Return ``(marker, list_style)`` or ``None``."""
    for pattern, style in _LIST_PATTERNS:
        m = pattern.match(text)
        if m:
            return m.group(0).strip(), style
    return None


# ---------------------------------------------------------------------------
# 6. Heading level from font size
# ---------------------------------------------------------------------------

def propose_heading_level(font_size: float, all_sizes: list[float]) -> int:
    """Rank this size among all sizes on the page. Largest -> level 1."""
    if not all_sizes:
        return 2
    ranked = sorted(set(all_sizes), reverse=True)
    try:
        idx = ranked.index(font_size)
    except ValueError:
        # Find nearest rank
        idx = min(range(len(ranked)), key=lambda i: abs(ranked[i] - font_size))
    return min(6, max(1, idx + 1))