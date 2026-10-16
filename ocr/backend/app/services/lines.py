"""Auto text lines inside regions ("Detect lines").

For each text region, find the printed lines on the page image (horizontal
bands of ink inside the region box) and match them, in order, to the
region's text lines. Each match becomes one ``lines[]`` record whose box is
the printed line, not the whole region.

Works on the raw v2 page dict (the editor's JSON), so it can run on unsaved
edits. Only the processed regions' lines are replaced; every other line
(other regions, table lines) is left exactly as it was.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

import numpy as np

from app.services.geometry import bbox_to_quad

TEXT_TYPES = frozenset({
    "paragraph", "header", "title", "heading", "subheading", "list", "list_item",
    "form_field", "table_heading", "footer", "page_number",
})

# keys of an old line worth carrying over to the new line at the same index
_CARRY = ("font_id", "role", "handwritten", "list_level", "item_index")

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


def ink_bands(
    gray: np.ndarray,
    box: Iterable[float],
    *,
    inset: int = 0,
    thresh: int = 140,
    min_h: int = 12,
    merge: int = 4,
    pad: int = 6,
    clip: bool = True,
) -> list[list[int]]:
    """Boxes of the printed lines inside ``box``, top to bottom.

    A pixel is ink when its gray value is < ``thresh``. A row with more than
    one ink pixel belongs to a band; bands closer than ``merge`` px join,
    bands shorter than ``min_h`` px are dropped (specks, underlines), and
    each band is padded by ``pad`` px. With ``clip`` the result stays inside
    ``box`` (a line never sticks out of its region).
    """
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    ax0, ay0, ax1, ay1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
    if ax1 <= ax0 or ay1 <= ay0:
        return []
    sub = gray[ay0:ay1, ax0:ax1] < thresh
    rows = sub.sum(1) > 1
    bands = []
    for a, b in runs(rows, gap=merge):
        if b - a + 1 < min_h:
            continue
        cols = np.flatnonzero(sub[a:b + 1].any(0))
        bx = [ax0 + int(cols.min()) - pad, ay0 + a - pad, ax0 + int(cols.max()) + 1 + pad, ay0 + b + 1 + pad]
        if clip:
            bx = [max(bx[0], x0), max(bx[1], y0), min(bx[2], x1), min(bx[3], y1)]
        bands.append(bx)
    return bands


def union(boxes: list[list[int]]) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


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

    The region ``text`` is split at "\\n" when it has line breaks. Saved
    regions keep their text joined by spaces (finalize does that), so
    otherwise the per-line texts come from the region's current lines -
    which is what the editor's text box shows, one row per line.
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


def _match(gray, el, box, texts, old, start, role=None, what="") -> tuple[list[dict], Optional[str]]:
    """Lines for one box: one per printed line, or one for the whole box + a warning."""
    rid = el["id"]
    heading = el.get("type") == "heading"
    bands = ink_bands(gray, box)
    if not texts:
        return [], None
    if len(bands) == len(texts):
        return [_line(t, b, rid, start + i, heading,
                      carry=old[i] if len(old) == len(texts) else None, role=role)
                for i, (t, b) in enumerate(zip(texts, bands))], None
    one = _line(" ".join(texts), union(bands) if bands else [int(v) for v in box], rid, start, heading,
                carry=old[0] if old else None, role=role)
    warn = (f"block {rid}{what}: {len(texts)} text lines but {len(bands)} printed lines"
            " — add line breaks")
    return [one], warn


def _form_field_lines(gray, el, old) -> tuple[Optional[list[dict]], list[str]]:
    """Lines for every field's label and value box (role "label" / "value").

    Returns ``(None, warnings)`` when a field's label and value are still one
    copied box: there is no way to tell which printed line is which, and old
    lines don't record their field, so the region's lines are left alone.
    """
    rid = el["id"]
    fields = el.get("fields") or []
    shared = [k for k, f in enumerate(fields)
              if isinstance((f.get("value") or None), dict) and (f.get("value") or {}).get("bbox")
              and list((f.get("label") or {}).get("bbox") or []) == list(f["value"]["bbox"])]
    if shared:
        return None, [f"block {rid} field {k}: label and value share one box"
                      " — draw them separately; lines left unchanged" for k in shared]
    out, warns = [], []
    for k, f in enumerate(fields):
        label, value = f.get("label") or {}, f.get("value") or {}
        parts = [("label", label)]
        if isinstance(value, dict) and value.get("bbox"):  # checkbox values have no box
            parts.append(("value", value))
        for role, part in parts:
            if not part.get("bbox"):
                continue
            texts = [t.strip() for t in (part.get("text") or "").split("\n") if t.strip()]
            mine = [ln for ln in old if ln.get("role") == role]
            got, w = _match(gray, el, part["bbox"], texts, mine, len(out), role=role,
                            what=f" field {k} {role}")
            out += got
            if w:
                warns.append(w)
    return out, warns


def detect_lines(
    gray: np.ndarray, rec: dict[str, Any], region_ids: Optional[set[int]] = None,
) -> tuple[dict[str, Any], list[str]]:
    """Rebuild ``rec["lines"]`` for the text regions (all, or ``region_ids``).

    Returns ``(rec, warnings)``; ``rec`` is changed in place.
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

        if el["type"] == "form_field":
            new, warns = _form_field_lines(gray, el, old)
            warnings += warns
            if new is None:
                continue
        else:
            texts = _texts(el, old)
            if not texts:
                continue
            new, w = _match(gray, el, el["bbox"], texts, old, 0)
            if w:
                warnings.append(w)
                if el["type"] == "list":
                    # one record would lose which item each line belongs to
                    warnings[-1] += " (list lines left unchanged)"
                    continue

        keep = [ln for ln in rec.get("lines") or []
                if not (isinstance(ln, dict) and ln.get("block_id") == rid and not _is_table_line(ln))]
        rec["lines"] = keep + new
    return rec, warnings
