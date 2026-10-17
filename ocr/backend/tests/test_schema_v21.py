"""Schema 2.1 shape tests (A1, A2, A3, A4, A8). Content rules are tested with
the validate package, not here."""
from __future__ import annotations

import copy
import json
import shutil

import pytest
from pydantic import ValidationError

from app.schemas import doc_v21 as s
from scripts import gen_schema


def _box(x0=0, y0=0, x1=10, y1=10):
    return [x0, y0, x1, y1]


def _quality():
    return {
        "is_grayscale": False, "is_binarized": False, "perspective": False,
        "is_clean": True, "has_stamp": False, "has_signature": False,
        "has_handwriting": False, "has_fold": False, "has_bleedthrough": False,
        "raw_lengths": {"paragraph_chars": 0, "list_items": 0, "form_fields": 0, "table_cells": 0},
    }


def _line(i, text, block_id, **kw):
    return {"id": i, "text": text, "bbox": _box(), "block_id": block_id, "line_in_block": 0,
            "is_heading": False, "font_size": 12.0, **kw}


def minimal_doc() -> dict:
    """A page with: a checkbox table (A2 tick cells), a two-line header cell
    (A4) and a form field with a blank value (A3)."""
    cells = [
        {"id": 0, "row": 0, "col": 0, "rowspan": 1, "colspan": 2, "text": "क्र.\nसं.",
         "is_header": True, "handwritten": False, "empty": False, "bbox": _box(0, 0, 20, 10)},
        {"id": 1, "row": 1, "col": 0, "rowspan": 1, "colspan": 1, "text": "कृषि",
         "is_header": False, "handwritten": False, "empty": False, "bbox": _box(0, 10, 10, 20)},
        {"id": 2, "row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "",
         "is_header": False, "handwritten": False, "empty": True, "bbox": _box(10, 10, 20, 20)},
    ]
    return {
        "v": "2.1", "image": "gov_005_page3.png", "width": 100, "height": 100,
        "document_id": "doc_0000007", "page_number": 3, "page_count": 4,
        "language": "ne", "digit_style": "none", "doc_type": "government_form", "text": "",
        "layout": [
            {"id": 0, "type": "table", "bbox": _box(0, 0, 20, 20), "reading_order": 0,
             "text": "", "table_index": 0,
             "fields": [{"label": {"region_id": 0}, "field_type": "checkbox",
                         "options": [{"line_id": 1, "checked": None, "tick_cell_id": 2}]}]},
            {"id": 1, "type": "form_field", "bbox": _box(0, 30, 50, 40), "reading_order": 1,
             "text": "मिति:",
             "fields": [{"label": {"line_id": 2}, "field_type": "text",
                         "value": {"text": "", "bbox": _box(20, 30, 50, 40),
                                   "handwritten": False, "empty": True}}]},
        ],
        "blocks": [],
        "lines": [
            _line(0, "क्र.", 0, table_index=0, cell_id=0),
            _line(1, "कृषि", 0, role="option", table_index=0, cell_id=1),
            _line(2, "मिति:", 1, role="label"),
        ],
        "tables": [{
            "bbox": _box(0, 0, 20, 20), "num_rows": 2, "num_cols": 2, "header_rows": 1,
            "ruling_style": "full", "html": "", "reading_order": [0, 1, 2], "cells": cells,
        }],
        "ruling_lines": [], "extra_regions": [],
        "quality": _quality(),
        "annotation": {"source": "real", "tool": "devalabel", "status": "in_progress"},
    }


def _dump(doc: s.Doc) -> str:
    return json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2)


# --- round trip -------------------------------------------------------------

def test_round_trip_is_byte_identical():
    once = _dump(s.Doc.model_validate(minimal_doc()))
    twice = _dump(s.Doc.model_validate(json.loads(once)))
    assert once == twice


def test_multiline_cell_text_keeps_newline():
    doc = s.Doc.model_validate(minimal_doc())
    assert doc.tables[0].cells[0].text == "क्र.\nसं."


def test_region_key_order_matches_v2():
    keys = list(s.Doc.model_validate(minimal_doc()).model_dump()["layout"][1])
    assert keys == ["id", "type", "bbox", "reading_order", "text", "fields", "quad"]
    keys = list(s.TitleLayout(id=0, bbox=_box(), text="x").model_dump())
    assert keys == ["id", "type", "bbox", "reading_order", "text", "quad"]


# --- A1 region types --------------------------------------------------------

@pytest.mark.parametrize("removed", s.REMOVED_REGION_TYPES)
def test_removed_region_types_rejected(removed):
    d = minimal_doc()
    d["layout"].append({"id": 2, "type": removed, "bbox": _box(), "text": "x"})
    with pytest.raises(ValidationError):
        s.Doc.model_validate(d)


@pytest.mark.parametrize("rtype", s.FIELD_REGION_TYPES)
def test_fields_allowed_on_form_table_list(rtype):
    model = {"form_field": s.FormFieldLayout, "table": s.TableLayout, "list": s.ListLayout}[rtype]
    assert "fields" in model.model_fields


def test_fields_not_allowed_on_paragraph():
    with pytest.raises(ValidationError):
        s.ParagraphLayout(id=0, bbox=_box(), text="x", fields=[])


# --- A2 checkbox design -----------------------------------------------------

def test_label_is_optional():
    assert s.FormField(field_type="checkbox").label is None


@pytest.mark.parametrize("ref", [{}, {"line_id": 1, "cell_id": 2},
                                 {"line_id": 1, "region_id": 0}])
def test_label_needs_exactly_one_reference(ref):
    with pytest.raises(ValidationError):
        s.LabelRef(**ref)


@pytest.mark.parametrize("opt", [{"line_id": 1},
                                 {"line_id": 1, "box": _box(), "tick_cell_id": 3}])
def test_option_needs_box_xor_tick_cell(opt):
    with pytest.raises(ValidationError):
        s.FieldOption(**opt)


def test_option_checked_defaults_to_null_and_omits_unused_mark():
    o = s.FieldOption(line_id=1, tick_cell_id=4)
    assert o.model_dump() == {"line_id": 1, "checked": None, "tick_cell_id": 4}


@pytest.mark.parametrize("bad", ["field_index", "checked", "box", "tick_cell_id"])
def test_lines_cannot_carry_field_keys(bad):
    d = minimal_doc()
    d["lines"][0][bad] = 0
    with pytest.raises(ValidationError):
        s.Doc.model_validate(d)


def test_lines_need_an_id():
    d = minimal_doc()
    del d["lines"][0]["id"]
    with pytest.raises(ValidationError):
        s.Doc.model_validate(d)


def test_no_choice_grid_and_no_checkbox_extra_region():
    d = minimal_doc()
    d["layout"][0]["choice_grid"] = []
    with pytest.raises(ValidationError):
        s.Doc.model_validate(d)
    d = minimal_doc()
    d["extra_regions"] = [{"type": "checkbox", "checked": True, "option": "x", "bbox": _box()}]
    with pytest.raises(ValidationError):
        s.Doc.model_validate(d)


# --- A3 blanks --------------------------------------------------------------

def test_blank_value_shape():
    v = s.Doc.model_validate(minimal_doc()).layout[1].fields[0].value
    assert v.model_dump() == {"text": "", "bbox": [20, 30, 50, 40],
                              "handwritten": False, "empty": True}
    with pytest.raises(ValidationError):
        s.FieldValue(text="", bbox=_box(), handwritten=False)  # empty is required


def test_preprinted_value_points_at_its_line():
    v = s.FieldValue(line_id=3, text="लुम्बिनी", bbox=_box(), handwritten=False, empty=False)
    assert list(v.model_dump()) == ["line_id", "text", "bbox", "handwritten", "empty"]
    assert "line_id" not in s.FieldValue(text="", bbox=_box(), handwritten=False,
                                         empty=True).model_dump()


# --- A8 page conventions ----------------------------------------------------

@pytest.mark.parametrize("ok", ["dev", "latin", "mixed", "none"])
def test_digit_style_values(ok):
    d = minimal_doc()
    d["digit_style"] = ok
    s.Doc.model_validate(d)


def test_old_digit_style_spelling_rejected():
    d = minimal_doc()
    d["digit_style"] = "devanagari"
    with pytest.raises(ValidationError):
        s.Doc.model_validate(d)


def test_version_is_2_1():
    d = minimal_doc()
    d["v"] = 2
    with pytest.raises(ValidationError):
        s.Doc.model_validate(d)


# --- generated files stay in sync ------------------------------------------

@pytest.mark.skipif(not (gen_schema.FRONTEND / "node_modules" / ".bin" / "json2ts").exists(),
                    reason="pnpm install not run in ocr/frontend")
def test_generated_schema_files_up_to_date():
    assert gen_schema.main(["--check"]) == 0, "run: python -m scripts.gen_schema"


def test_minimal_doc_not_mutated():
    d = minimal_doc()
    before = copy.deepcopy(d)
    s.Doc.model_validate(d)
    assert d == before
