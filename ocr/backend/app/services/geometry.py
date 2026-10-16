"""Auto quads: every box gets a quad that agrees with its bbox.

Rule: a quad is computed from its bbox, except when someone deliberately
drew a tilted quad that still fits the box. Called on every save (before
validation) and every load, so quads never have to be drawn by hand.

Pure functions, no imports from the rest of the app (the schema imports
this module, so it must not import the schema back).
"""
from __future__ import annotations

from typing import Any

# Record keys whose items carry ``bbox`` + ``quad``. ``ruling_lines`` is
# deliberately absent: a RulingLine has a bbox but no quad in the v2 schema
# (and the schema forbids extra keys), so adding one would break validation.
_BOX_LISTS = ("blocks", "lines", "tables", "extra_regions")


def bbox_to_quad(b: list[int]) -> list[list[int]]:
    x0, y0, x1, y1 = b
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # TL, TR, BR, BL


def quad_matches_bbox(q: Any, b: list[int], tol: float = 3) -> bool:
    """True if the quad's outer bounds equal the bbox (keeps real tilted quads)."""
    try:
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        return (
            len(q) == 4
            and all(len(p) == 2 for p in q)
            and abs(min(xs) - b[0]) <= tol and abs(min(ys) - b[1]) <= tol
            and abs(max(xs) - b[2]) <= tol and abs(max(ys) - b[3]) <= tol
        )
    except (TypeError, IndexError, ValueError):
        return False


def fix_quad(obj: Any) -> None:
    """Rebuild ``obj["quad"]`` from ``obj["bbox"]`` when missing or stale (in place)."""
    if not isinstance(obj, dict):
        return
    b = obj.get("bbox")
    if (
        isinstance(b, (list, tuple)) and len(b) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)
    ):
        if not quad_matches_bbox(obj.get("quad"), b):
            obj["quad"] = bbox_to_quad(list(b))


def ensure_quads(rec: dict[str, Any]) -> dict[str, Any]:
    """Give every box in a v2 page record a quad that agrees with its bbox.

    Mutates and returns ``rec``. Objects without a usable bbox are left alone
    (e.g. checkbox options, which only have ``box_quad`` / ``text_quad``).
    """
    for item in rec.get("layout") or []:
        fix_quad(item)
        if isinstance(item, dict):
            for f in item.get("fields") or []:
                if isinstance(f, dict):
                    fix_quad(f.get("label"))
                    fix_quad(f.get("value"))  # checkbox values have no bbox: skipped
    for key in _BOX_LISTS:
        for item in rec.get(key) or []:
            fix_quad(item)
    for t in rec.get("tables") or []:
        if isinstance(t, dict):
            for c in t.get("cells") or []:
                fix_quad(c)
    return rec
