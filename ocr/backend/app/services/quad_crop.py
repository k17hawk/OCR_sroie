"""Cut a quad out of the scan as a straight image.

A tilted line's quad is not a rectangle in page pixels. Cropping its outer
bbox hands the recognizer slanted text plus pieces of the lines above and
below; warping the quad onto a flat rectangle gives it one straight line.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
from PIL import Image as PILImage


def _is_axis_rect(q: Sequence[Sequence[float]], tol: float = 0.5) -> bool:
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = q
    return (abs(tly - try_) <= tol and abs(bly - bry) <= tol
            and abs(tlx - blx) <= tol and abs(trx - brx) <= tol)


def crop_quad(img: PILImage.Image, quad: Sequence[Sequence[float]]) -> PILImage.Image:
    """The quad (TL, TR, BR, BL in page pixels) as an upright image.

    An axis-aligned quad is a plain crop (same pixels as before); any other
    quad is perspective-warped so its top edge becomes horizontal.
    """
    if len(quad) != 4:
        raise ValueError("a quad has 4 points")
    q = [(float(p[0]), float(p[1])) for p in quad]
    xs, ys = [p[0] for p in q], [p[1] for p in q]
    W, H = img.size
    if _is_axis_rect(q):
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(W, int(math.ceil(max(xs)))), min(H, int(math.ceil(max(ys))))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("empty quad")
        return img.crop((x0, y0, x1, y1))

    tl, tr, br, bl = q
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    out_w = int(round(max(dist(tl, tr), dist(bl, br))))
    out_h = int(round(max(dist(tl, bl), dist(tr, br))))
    if out_w < 1 or out_h < 1:
        raise ValueError("empty quad")
    src = np.array(q, dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    rgb = np.asarray(img.convert("RGB"))
    warped = cv2.warpPerspective(rgb, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return PILImage.fromarray(warped)
