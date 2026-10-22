"""A "page_number" region is its own type: offered by default, and kept as
page_number through save and reload (no longer turned into a footer)."""
from __future__ import annotations

from app.core.finalize import finalize
from app.schemas.doc_v2 import REGION_TYPES
from app.services.projects import DEFAULT_LAYOUT_TYPES
from app.storage.doc_store import doc_from_record

from conftest import make_page


def q(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def test_page_number_is_a_region_type_offered_by_default():
    assert "page_number" in REGION_TYPES
    assert "page_number" in DEFAULT_LAYOUT_TYPES


def test_page_number_survives_save_and_reload():
    rec = make_page(layout=[{"id": 1, "type": "page_number", "bbox": [480, 1600, 520, 1630],
                             "reading_order": 0, "text": "३", "quad": q([480, 1600, 520, 1630])}])
    a = finalize(doc_from_record(rec)[0]).model_dump(mode="json")
    b = finalize(doc_from_record(a)[0]).model_dump(mode="json")
    assert a == b
    assert a["layout"][0]["type"] == "page_number"
    assert a["layout"][0]["text"] == "३"
