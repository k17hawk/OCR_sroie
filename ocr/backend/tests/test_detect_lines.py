"""Step 2: one lines[] record per printed line inside each text region."""
from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.schemas.doc_v2 import Doc
from app.services.lines import detect_lines, ink_bands
from conftest import blank, ink, make_page

PARA = [100, 100, 600, 260]
# three printed lines of different lengths inside PARA
PRINTED = [(120, 115, 580, 140), (120, 165, 500, 190), (120, 215, 420, 240)]


def _scan():
    g = blank()
    for b in PRINTED:
        ink(g, b)
    ink(g, (700, 115, 900, 140))  # ink of another region, outside PARA
    return g


def _region(text, rid=0, typ="paragraph", bbox=PARA):
    return {"id": rid, "type": typ, "bbox": list(bbox), "text": text}


def _whole_block_line(text, rid=0, bbox=PARA):
    return {"text": text, "bbox": list(bbox), "block_id": rid, "line_in_block": 0,
            "is_heading": False, "font_size": 20.0, "font_id": None}


def _inside(inner, outer):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def test_three_line_paragraph_gives_three_lines_inside_the_block():
    rec = make_page(layout=[_region("एक\nदुई\nतीन")])
    rec, warnings = detect_lines(_scan(), rec)
    assert warnings == []
    lines = rec["lines"]
    assert [ln["text"] for ln in lines] == ["एक", "दुई", "तीन"]
    assert [ln["line_in_block"] for ln in lines] == [0, 1, 2]
    for ln, (x0, y0, x1, y1) in zip(lines, PRINTED):
        assert _inside(ln["bbox"], PARA)
        # the printed line, padded by 6 px (clipped to the region)
        assert ln["bbox"] == [x0 - 6, y0 - 6, x1 + 6, y1 + 6]
        assert ln["font_size"] == round(0.8 * (ln["bbox"][3] - ln["bbox"][1]), 1)
        assert ln["quad"][0] == ln["bbox"][:2] and ln["quad"][2] == ln["bbox"][2:]
        assert ln["block_id"] == 0 and ln["font_id"] is None and ln["is_heading"] is False
    Doc.model_validate(rec)  # still a valid v2 page


def test_texts_come_from_existing_lines_when_region_text_is_joined():
    # saved regions keep their text joined by spaces; the lines hold the rows
    lines = [dict(_whole_block_line(t), line_in_block=i) for i, t in enumerate(["एक", "दुई", "तीन"])]
    rec = make_page(layout=[_region("एक दुई तीन")], lines=lines)
    rec, warnings = detect_lines(_scan(), rec)
    assert warnings == []
    assert [ln["text"] for ln in rec["lines"]] == ["एक", "दुई", "तीन"]
    assert all(ln["bbox"] != PARA for ln in rec["lines"])


def test_count_mismatch_keeps_one_line_and_warns():
    rec = make_page(layout=[_region("एक दुई तीन")], lines=[_whole_block_line("एक दुई तीन")])
    rec, warnings = detect_lines(_scan(), rec)
    assert warnings == ["block 0: 1 text lines but 3 printed lines — add line breaks"]
    assert len(rec["lines"]) == 1
    assert rec["lines"][0]["text"] == "एक दुई तीन"
    assert _inside(rec["lines"][0]["bbox"], PARA)


def test_other_regions_and_table_lines_are_not_touched():
    other = _whole_block_line("अर्को", rid=1, bbox=[700, 100, 950, 150])
    table_line = dict(_whole_block_line("x", rid=0), table_index=0, cell_id=0)
    rec = make_page(layout=[_region("एक\nदुई\nतीन"), _region("अर्को", rid=1, bbox=[700, 100, 950, 150])],
                    lines=[copy.deepcopy(other), copy.deepcopy(table_line)])
    rec, _ = detect_lines(_scan(), rec, region_ids={0})
    assert other in rec["lines"] and table_line in rec["lines"]
    assert len(rec["lines"]) == 5


def test_heading_marker_is_ignored():
    rec = make_page(layout=[{**_region("## शीर्षक", typ="heading", bbox=[100, 100, 600, 150]), "level": 2}])
    g = blank()
    ink(g, (120, 115, 400, 140))
    rec, warnings = detect_lines(g, rec)
    assert warnings == []
    assert rec["lines"][0]["text"] == "शीर्षक" and rec["lines"][0]["is_heading"] is True


def test_form_field_lines_get_label_and_value_roles():
    ff = {"id": 3, "type": "form_field", "bbox": [100, 300, 900, 360], "text": "नाम: राम",
          "fields": [{"label": {"text": "नाम", "bbox": [100, 300, 300, 360]}, "field_type": "text",
                      "value": {"text": "राम", "bbox": [320, 300, 900, 360], "handwritten": False}}]}
    g = blank()
    ink(g, (120, 315, 250, 340))
    ink(g, (340, 315, 600, 340))
    rec, warnings = detect_lines(g, make_page(layout=[ff]))
    assert warnings == []
    assert [(ln["role"], ln["text"]) for ln in rec["lines"]] == [("label", "नाम"), ("value", "राम")]
    assert _inside(rec["lines"][0]["bbox"], [100, 300, 300, 360])


def test_form_field_with_copied_boxes_is_left_alone():
    box = [100, 300, 900, 360]
    old = dict(_whole_block_line("नाम राम", rid=3, bbox=box), role="label")
    ff = {"id": 3, "type": "form_field", "bbox": box, "text": "नाम: राम",
          "fields": [{"label": {"text": "नाम", "bbox": box}, "field_type": "text",
                      "value": {"text": "राम", "bbox": box, "handwritten": False}}]}
    rec, warnings = detect_lines(blank(), make_page(layout=[ff], lines=[copy.deepcopy(old)]))
    assert rec["lines"] == [old]
    assert "share one box" in warnings[0]


def test_specks_and_underlines_are_not_lines():
    g = blank()
    ink(g, (120, 120, 500, 125))  # 5 px tall: an underline
    ink(g, (300, 200, 303, 203))  # a speck
    assert ink_bands(g, PARA) == []


def test_endpoint_uses_the_editor_doc_and_returns_warnings(pagestore):
    from app.main import app

    saved = make_page(layout=[_region("पुरानो")])
    pagestore(saved, _scan())
    editor = make_page(layout=[_region("एक\nदुई\nतीन")])  # unsaved edit
    r = TestClient(app).post("/documents/DOC-0001/detect-lines", json={"doc": editor, "element_id": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["warnings"] == []
    assert [ln["text"] for ln in body["record"]["lines"]] == ["एक", "दुई", "तीन"]


def test_endpoint_unknown_region_is_404(pagestore):
    from app.main import app

    pagestore(make_page(), _scan())
    r = TestClient(app).post("/documents/DOC-0001/detect-lines", json={"element_id": 99})
    assert r.status_code == 404
