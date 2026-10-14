"""Geometry helpers. Pure functions only — no I/O, no model coupling."""
from __future__ import annotations

import math

from app.schemas.doc_v2 import BBox, Quad


def bbox_from_quad(quad: Quad) -> BBox:
    """Return the tight integer bbox of ``quad``.

    Rule (verified against every element in ``schema/sample.json``):

        x0 = floor(min(px for px, _ in quad))
        y0 = floor(min(py for _, py in quad))
        x1 = ceil (max(px for px, _ in quad))
        y1 = ceil (max(py for _, py in quad))

    The extremes are taken over **all four** points, not just the two
    diagonal corners.
    """
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [
        math.floor(min(xs)),
        math.floor(min(ys)),
        math.ceil(max(xs)),
        math.ceil(max(ys)),
    ]
