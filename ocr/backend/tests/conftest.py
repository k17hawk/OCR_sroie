"""Shared fixtures: a minimal v2 page, a synthetic scan, a temp store."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

W, H = 1000, 1400


def make_page(**overrides):
    """A small valid v2 page record with no regions; tests add what they need."""
    rec = {
        "v": 2, "image": "img/p1.png", "width": W, "height": H,
        "document_id": "DOC-0001", "page_number": 1, "page_count": 1,
        "language": "ne", "digit_style": "devanagari", "doc_type": "letter",
        "text": "", "layout": [], "blocks": [], "lines": [], "tables": [],
        "ruling_lines": [], "extra_regions": [],
        "quality": {
            "is_grayscale": False, "is_binarized": False, "perspective": False,
            "is_clean": True, "has_stamp": False, "has_signature": False,
            "has_handwriting": False, "has_fold": False, "has_bleedthrough": False,
            "raw_lengths": {"paragraph_chars": 0, "list_items": 0, "form_fields": 0, "table_cells": 0},
        },
        "annotation": {"source": "real", "tool": "devalabel", "status": "in_progress"},
    }
    rec.update(overrides)
    return rec


def blank() -> np.ndarray:
    return np.full((H, W), 255, np.uint8)


def ink(gray: np.ndarray, box) -> None:
    """Paint a solid block: a rule, a border, a speck or a bar."""
    x0, y0, x1, y1 = box
    gray[y0:y1, x0:x1] = 0


def write(gray: np.ndarray, x: int, y: int, text: str, size: int = 28) -> list[int]:
    """Render real text at (x, y) and return its ink box [x0, y0, x1, y1].

    Line detection must see glyph-like ink: a long solid bar is (correctly)
    treated as a rule or an underline and ignored.
    """
    from PIL import ImageDraw, ImageFont

    before = gray.copy()
    img = Image.fromarray(gray)
    ImageDraw.Draw(img).text((x, y), text, fill=0, font=ImageFont.load_default(size=size))
    gray[:] = np.asarray(img)
    ys, xs = np.nonzero((gray < 128) & (before >= 128))   # only this text's ink
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


@pytest.fixture()
def page():
    return make_page


@pytest.fixture()
def pagestore(tmp_path, monkeypatch):
    """Point settings + storage at a temp dir; returns ``put(rec, gray)``,
    which saves a page record and its scan."""
    from app.config import get_settings
    from app.storage import get_storage

    monkeypatch.setenv("DEVALABEL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("DEVALABEL_STORAGE_BACKEND", "local")
    from app.api.documents import _decode_gray

    get_settings.cache_clear()
    get_storage.cache_clear()
    _decode_gray.cache_clear()
    from app.storage import doc_store

    def put(rec, gray=None):
        doc_store._doc_path(rec["document_id"]).write_text(json.dumps(rec), encoding="utf-8")
        if gray is not None:
            buf = io.BytesIO()
            Image.fromarray(gray).save(buf, format="PNG")
            p = tmp_path / rec["image"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(buf.getvalue())

    yield put
    get_settings.cache_clear()
    get_storage.cache_clear()
    _decode_gray.cache_clear()
