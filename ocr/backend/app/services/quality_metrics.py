"""Auto-computed quality metrics for real images.

Aimed at being *useful*, not research-grade. Every value is deterministic.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


MAX_METRIC_EDGE = 1500  # px — quality metrics don't need full resolution
MAX_SKEW_EDGE = 500

def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _laplacian_variance(g: np.ndarray) -> float:
    lap = (
        -4.0 * g
        + np.roll(g, 1, 0) + np.roll(g, -1, 0)
        + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    )
    return float(lap[1:-1, 1:-1].var())


def _noise_sigma(g: np.ndarray) -> float:
    """Rough sigma from the residual of a 3x3 median filter."""
    med = np.asarray(
        Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
        dtype=np.float32,
    )
    resid = g - med
    return float(resid.std())


def _skew_deg(g: np.ndarray) -> float:
    """Coarse skew from horizontal projection-profile variance."""
    best_angle, best_score = 0.0, -1.0
    h, w = g.shape
    cy, cx = h / 2.0, w / 2.0
    ys, xs = np.mgrid[0:h, 0:w]

    for angle_deg in np.arange(-5.0, 5.01, 0.5):
        a = np.deg2rad(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        # Rotated coordinates (inverse mapping, no PIL)
        rx = ca * (xs - cx) - sa * (ys - cy) + cx
        ry = sa * (xs - cx) + ca * (ys - cy) + cy
        ri = np.clip(ry.round().astype(int), 0, h - 1)
        # Sum only the pixels that landed inside the source image
        valid = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)
        rows = np.bincount(ri[valid], weights=g[valid], minlength=h)
        score = float(rows.var())
        if score > best_score:
            best_score, best_angle = score, float(angle_deg)
    return round(best_angle, 1)

def _is_grayscale(img: Image.Image) -> bool:
    if img.mode in ("L", "LA", "1"):
        return True
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    return bool(
        np.abs(rgb[..., 0] - rgb[..., 1]).max() <= 2
        and np.abs(rgb[..., 1] - rgb[..., 2]).max() <= 2
    )


def _is_binarized(g: np.ndarray) -> bool:
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    nonzero = (hist > (g.size * 0.005)).sum()
    mid = hist[60:200].sum() / max(1, g.size)
    return bool(nonzero <= 4 and mid < 0.02)


def _jpeg_quality(orig_bytes: bytes | None) -> int | None:
    if not orig_bytes:
        return None
    try:
        im = Image.open(BytesIO(orig_bytes))
        if im.format != "JPEG":
            return None
        q = getattr(im, "quantization", None)
        if not q:
            return None
        # Mean of the luminance table's first 10 entries -> rough quality estimate.
        table = list(q.get(0, []))[:10]
        if not table:
            return None
        avg = sum(table) / len(table)
        return int(max(1, min(100, round(100 - avg / 2))))
    except Exception:
        return None


def _downscale(img: Image.Image, max_edge: int) -> Image.Image:
    if max(img.size) <= max_edge:
        return img
    scale = max_edge / max(img.size)
    return img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.LANCZOS,
    )


def compute_quality(
    img: Image.Image, orig_bytes: bytes | None = None
) -> dict[str, Any]:
    small = _downscale(img, MAX_METRIC_EDGE)
    g = _gray(small)
    lap_var = _laplacian_variance(g)
    blur_sigma = round(float(np.clip(40.0 / (1.0 + lap_var), 0.0, 3.0)), 2)
    return {
        "blur_sigma": blur_sigma,
        "downscale": None,
        "noise_sigma": round(_noise_sigma(g), 2),
        "is_grayscale": _is_grayscale(small),
        "is_binarized": _is_binarized(g),
        "skew_deg": _skew_deg(_gray(_downscale(small, MAX_SKEW_EDGE))),
        "perspective": False,
        "ops": [],
        "jpeg_quality": _jpeg_quality(orig_bytes),
        "profile": "unknown",
        "is_clean": blur_sigma < 1.0,
        "has_stamp": False,
        "has_signature": False,
        "has_handwriting": False,
        "has_fold": False,
        "has_bleedthrough": False,
    }