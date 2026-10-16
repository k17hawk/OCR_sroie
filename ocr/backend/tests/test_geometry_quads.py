"""Auto quads: app/services/geometry.py, the schema hook, save and load."""
from __future__ import annotations

import copy
import json

import pytest

from app.schemas.doc_v2 import Doc
from app.services.geometry import bbox_to_quad, ensure_quads, quad_matches_bbox


# ---------------------------------------------------------------------------
# a minimal valid v2 page (quads left out on purpose where noted)
# ---------------------------------------------------------------------------

def _page(**overrides):
    rec = {
        "v": 2,
        "image": "img/p1.png",
        "width": 1000,
        "height": 1400,
        "document_id": "DOC-0001",
        "page_number": 1,
        "page_count": 1,
        "language": "ne",
        "digit_style": "devanagari",
        "doc_type": "letter",
        "text": "",
        "layout": [
            {"id": 0, "type": "paragraph", "bbox": [100, 100, 400, 200], "text": "क ख"},
            {
                "id": 1, "type": "form_field", "bbox": [100, 300, 600, 340], "text": "नाम",
                "fields": [{
                    "label": {"text": "नाम", "bbox": [100, 300, 200, 340]},
                    "field_type": "text",
                    "value": {"text": "राम", "bbox": [210, 300, 600, 340], "handwritten": False},
                }],
            },
        ],
        "blocks": [{"type": "paragraph", "bbox": [100, 100, 400, 200], "block_id": 0}],
        "lines": [{
            "text": "क ख", "bbox": [100, 100, 400, 140], "block_id": 0,
            "line_in_block": 0, "is_heading": False, "font_size": 32.0,
        }],
        "tables": [{
            "bbox": [100, 500, 900, 700], "num_rows": 1, "num_cols": 1, "header_rows": 0,
            "ruling_style": "full", "html": "<table><tr><td>x</td></tr></table>",
            "reading_order": [0],
            "cells": [{
                "id": 0, "row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "x",
                "is_header": False, "handwritten": False, "empty": False,
                "bbox": [100, 500, 900, 700],
            }],
        }],
        "ruling_lines": [{
            "table_index": 0, "orientation": "h", "width": 2,
            "p0": [100.0, 500.0], "p1": [900.0, 500.0], "bbox": [100, 499, 900, 501],
        }],
        "extra_regions": [{"type": "fingerprint", "hand": "right", "bbox": [700, 900, 800, 1000]}],
        "quality": {
            "is_grayscale": False, "is_binarized": False, "perspective": False,
            "is_clean": True, "has_stamp": False, "has_signature": False,
            "has_handwriting": False, "has_fold": False, "has_bleedthrough": False,
            "raw_lengths": {"paragraph_chars": 3, "list_items": 0, "form_fields": 1, "table_cells": 1},
        },
        "annotation": {"source": "real", "tool": "devalabel", "status": "in_progress"},
    }
    rec.update(overrides)
    return rec


TILTED = [[102, 100], [400, 104], [398, 200], [100, 196]]  # fits [100,100,400,200]


# ---------------------------------------------------------------------------
# ensure_quads (the three required cases)
# ---------------------------------------------------------------------------

def test_missing_quad_is_filled():
    rec = ensure_quads(_page())
    assert rec["layout"][0]["quad"] == bbox_to_quad([100, 100, 400, 200])
    # every place a box lives
    assert rec["layout"][1]["quad"] == bbox_to_quad([100, 300, 600, 340])
    assert rec["layout"][1]["fields"][0]["label"]["quad"] == bbox_to_quad([100, 300, 200, 340])
    assert rec["layout"][1]["fields"][0]["value"]["quad"] == bbox_to_quad([210, 300, 600, 340])
    assert rec["blocks"][0]["quad"] == bbox_to_quad([100, 100, 400, 200])
    assert rec["lines"][0]["quad"] == bbox_to_quad([100, 100, 400, 140])
    assert rec["tables"][0]["quad"] == bbox_to_quad([100, 500, 900, 700])
    assert rec["tables"][0]["cells"][0]["quad"] == bbox_to_quad([100, 500, 900, 700])
    assert rec["extra_regions"][0]["quad"] == bbox_to_quad([700, 900, 800, 1000])


def test_stale_quad_is_rebuilt_after_bbox_change():
    rec = ensure_quads(_page())
    # the box is moved 50 px right; its quad still describes the old place
    rec["layout"][0]["bbox"] = [150, 100, 450, 200]
    ensure_quads(rec)
    assert rec["layout"][0]["quad"] == bbox_to_quad([150, 100, 450, 200])


def test_tilted_quad_that_matches_bbox_is_kept():
    rec = _page()
    rec["layout"][0]["quad"] = copy.deepcopy(TILTED)
    ensure_quads(rec)
    assert rec["layout"][0]["quad"] == TILTED


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_ruling_lines_get_no_quad():
    # RulingLine has no quad field and the schema forbids extra keys.
    rec = ensure_quads(_page())
    assert "quad" not in rec["ruling_lines"][0]


def test_checkbox_value_without_bbox_is_left_alone():
    opt = {"text": "हो", "checked": True,
           "box_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
           "text_quad": [[12, 0], [40, 0], [40, 10], [12, 10]]}
    rec = _page()
    rec["layout"][1]["fields"][0]["value"] = {"text": "हो", "options": [opt]}
    ensure_quads(rec)
    assert "quad" not in rec["layout"][1]["fields"][0]["value"]


@pytest.mark.parametrize("bad", [None, [], [[1, 2]] * 3, "nope", [[1], [2], [3], [4]], [[1, "a"]] * 4])
def test_garbage_quad_is_replaced(bad):
    assert not quad_matches_bbox(bad, [0, 0, 10, 10])


def test_within_tolerance_is_kept():
    q = [[1, 1], [11, 1], [11, 11], [1, 11]]
    assert quad_matches_bbox(q, [0, 0, 10, 10])          # 1 px off
    assert not quad_matches_bbox(q, [0, 0, 10, 10], tol=0)


# ---------------------------------------------------------------------------
# schema: quad optional on input, always present on output
# ---------------------------------------------------------------------------

def test_doc_validates_without_quads_and_dumps_them():
    doc = Doc.model_validate(_page())
    out = doc.model_dump(mode="json")
    assert out["layout"][0]["quad"] == [[100.0, 100.0], [400.0, 100.0], [400.0, 200.0], [100.0, 200.0]]
    assert "quad" in out["lines"][0] and "quad" in out["tables"][0]["cells"][0]
    assert "quad" not in out["ruling_lines"][0]


# ---------------------------------------------------------------------------
# load: old files with missing / stale quads still load
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path, monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("DEVALABEL_STORAGE_ROOT", str(tmp_path))
    get_settings.cache_clear()
    from app.storage import doc_store
    yield doc_store
    get_settings.cache_clear()


def test_old_file_without_quads_loads(store):
    rec = _page()
    rec["lines"][0]["bbox"] = [100, 100, 400, 150]
    rec["lines"][0]["quad"] = bbox_to_quad([100, 100, 400, 140])  # stale
    path = store._doc_path("DOC-0001")
    path.write_text(json.dumps(rec), encoding="utf-8")
    row = store.load("DOC-0001")
    assert row is not None
    assert row.doc.layout[0].quad == [[100, 100], [400, 100], [400, 200], [100, 200]]
    assert row.doc.lines[0].quad == [[100, 100], [400, 100], [400, 150], [100, 150]]


# ---------------------------------------------------------------------------
# save: a moved box keeps its new place
# ---------------------------------------------------------------------------

def test_save_payload_rebuilds_stale_quad(store):
    from app.core.finalize import finalize
    from app.services.annotations import _clean_payload

    store._doc_path("DOC-0001").write_text(json.dumps(_page()), encoding="utf-8")
    row = store.load("DOC-0001")

    payload = row.doc.model_dump(mode="json")
    payload["layout"][0]["bbox"] = [150, 100, 450, 200]  # moved; quad not updated

    doc = Doc.model_validate(_clean_payload(payload, row))
    # finalize derives layout bboxes FROM quads; without the rebuild it would
    # snap the box back to its old position.
    assert finalize(doc).layout[0].bbox == [150, 100, 450, 200]
