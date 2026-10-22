"""A "sublist" region has the same properties as a list and survives save and
reload; bullet markers are kept as printed (•), not rewritten to "-"."""
from __future__ import annotations

from app.core.finalize import finalize
from app.schemas.doc_v2 import REGION_TYPES, ListLayout, SublistLayout
from app.services.projects import DEFAULT_LAYOUT_TYPES
from app.storage.doc_store import doc_from_record

from conftest import make_page


def q(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def _round_trip(rec):
    a = finalize(doc_from_record(rec)[0]).model_dump(mode="json")
    b = finalize(doc_from_record(a)[0]).model_dump(mode="json")
    assert a == b
    return a


def test_sublist_is_a_region_type_offered_by_default():
    assert "sublist" in REGION_TYPES
    assert "sublist" in DEFAULT_LAYOUT_TYPES


def test_sublist_has_the_same_properties_as_list():
    assert list(SublistLayout.model_fields) == list(ListLayout.model_fields)


def test_sublist_keeps_bullets_through_save_and_reload():
    rec = make_page(layout=[{"id": 1, "type": "sublist", "bbox": [100, 100, 900, 300], "reading_order": 0,
                             "text": "", "list_style": "bullet", "quad": q([100, 100, 900, 300]),
                             "items": [{"marker": "•", "level": 0, "text": "पहिलो"},
                                       {"marker": "•", "level": 0, "text": "दोस्रो"}]}])
    out = _round_trip(rec)
    el = out["layout"][0]
    assert el["type"] == "sublist"
    assert el["list_style"] == "bullet"
    assert el["text"] == "• पहिलो\n• दोस्रो"
    assert [i["marker"] for i in el["items"]] == ["•", "•"]
