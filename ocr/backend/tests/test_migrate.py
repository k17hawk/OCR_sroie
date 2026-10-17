"""Loading old page records: region-type mapping, line ids, old checkbox
shapes and choice_grid -> field.options[] (core/migrate.py)."""
from __future__ import annotations

import json

from app.core.finalize import finalize
from app.core.migrate import migrate
from app.schemas.doc_v2 import REGION_TYPES, Doc
from app.storage.doc_store import doc_from_record

from conftest import make_page


def q(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def region(id_, type_, bbox, ro, text="", **extra):
    return {"id": id_, "type": type_, "bbox": bbox, "reading_order": ro, "text": text, "quad": q(bbox), **extra}


def line(text, bbox, block, lib, **extra):
    return {"text": text, "bbox": bbox, "block_id": block, "line_in_block": lib,
            "is_heading": False, "font_size": 16.0, "font_id": None, **extra, "quad": q(bbox)}


def messages(warns):
    return " | ".join(w["message"] for w in warns)


# ---------------------------------------------------------------------------
# region types
# ---------------------------------------------------------------------------

def test_enum_is_the_sixteen_region_types():
    assert REGION_TYPES == {
        "title", "heading", "subheading", "paragraph", "list", "form_field", "table",
        "table_heading", "header", "footer", "figure", "logo", "stamp", "signature",
        "handwriting", "caption",
    }


def test_text_line_and_list_item_are_mapped_and_reported():
    rec = make_page(layout=[
        region(1, "text_line", [10, 10, 300, 40], 0, "वडा नं. १"),
        region(2, "list_item", [10, 50, 300, 80], 1, "क) सामान्य"),
        region(3, "list_item", [10, 90, 300, 120], 2, "छ ☐ छैन ☐"),
        region(4, "list_item", [10, 130, 300, 160], 3, "हो", fields=[]),
        region(5, "signature_caption", [10, 170, 300, 200], 4, "हस्ताक्षर"),
        region(6, "handwritten_number", [10, 210, 300, 240], 5, "१२"),
    ], lines=[line("हो", [12, 132, 40, 158], 4, 0, role="option")])
    doc, warns = doc_from_record(rec)
    types = {el.id: el.type for el in doc.layout}
    assert types == {1: "paragraph", 2: "paragraph", 3: "form_field", 4: "form_field",
                     5: "caption", 6: "handwriting"}
    assert doc.layout[5].subtype == "number"
    text = messages(warns)
    for frag in ("#1 text_line → paragraph", "#2 list_item → paragraph",
                 "#3 list_item → form_field", "#4 list_item → form_field"):
        assert frag in text


# ---------------------------------------------------------------------------
# line ids
# ---------------------------------------------------------------------------

def test_lines_get_ids_in_reading_order():
    rec = make_page(
        layout=[region(1, "paragraph", [0, 200, 500, 300], 1), region(2, "paragraph", [0, 0, 500, 100], 0)],
        lines=[
            line("b2", [0, 240, 100, 260], 1, 1), line("b1", [0, 210, 100, 230], 1, 0),
            line("a1", [0, 10, 100, 30], 2, 0),
        ],
    )
    out, _ = migrate(rec)
    ids = {ln["text"]: ln["id"] for ln in out["lines"]}
    assert ids == {"a1": 0, "b1": 1, "b2": 2}
    assert out["next_line_id"] == 3


def test_existing_ids_are_kept_and_new_ones_follow():
    rec = make_page(layout=[region(1, "paragraph", [0, 0, 500, 100], 0)],
                    lines=[line("x", [0, 0, 50, 20], 1, 0, id=7), line("y", [0, 30, 50, 50], 1, 1)],
                    next_line_id=12)
    out, _ = migrate(rec)
    assert [ln["id"] for ln in out["lines"]] == [7, 12]
    assert out["next_line_id"] == 13


# ---------------------------------------------------------------------------
# old checkbox shapes
# ---------------------------------------------------------------------------

def _old_checkbox_page():
    return make_page(layout=[region(1, "form_field", [0, 0, 800, 60], 0, fields=[{
        "label": {"text": "(ऊ) अपाङ्गता भएका सदस्य: (कुनै एकमा √ लगाउनुहोस्)",
                  "bbox": [0, 0, 500, 40], "quad": q([0, 0, 500, 40])},
        "field_type": "checkbox",
        "value": {"text": "[ ] छ [x] छैन", "options": [
            {"text": "छ", "checked": False, "box_quad": q([540, 5, 560, 25]), "text_quad": q([510, 5, 535, 30])},
            {"text": "छैन", "checked": True, "box_quad": q([640, 5, 660, 25]), "text_quad": q([600, 5, 635, 30])},
        ]},
    }])], lines=[
        line("(ऊ) अपाङ्गता भएका सदस्य: (कुनै एकमा √ लगाउनुहोस्)", [0, 0, 500, 40], 1, 0, role="label"),
        line("छ", [510, 5, 535, 30], 1, 1),  # already there: reused
    ])


def test_old_checkbox_value_becomes_options():
    doc, warns = doc_from_record(_old_checkbox_page())
    f = doc.layout[0].fields[0]
    assert f.value is None and f.field_type == "checkbox" and f.single_choice is True
    by_id = {ln.id: ln for ln in doc.lines}
    assert [by_id[o.line_id].text for o in f.options] == ["छ", "छैन"]
    assert [o.checked for o in f.options] == [False, True]
    assert f.options[0].box == [540, 5, 560, 25]
    assert all(by_id[o.line_id].role == "option" for o in f.options)
    assert len(doc.lines) == 3  # "छ" reused, "छैन" added
    assert f.label.line_id == doc.lines[0].id


def test_line_level_option_keys_move_into_field_options():
    rec = make_page(layout=[region(1, "form_field", [0, 0, 800, 60], 0, fields=[
        {"label": {"text": "लिङ्ग:", "bbox": [0, 0, 100, 40]}, "field_type": "text", "value": None},
    ])], lines=[
        line("लिङ्ग:", [0, 0, 100, 40], 1, 0, role="label"),
        line("क) पुरुष", [120, 0, 200, 40], 1, 1, role="option", field_index=0, checked=None, box=[205, 10, 225, 30]),
        line("ख) महिला", [240, 0, 320, 40], 1, 2, role="option", field_index=0, checked=True, box=[325, 10, 345, 30]),
    ])
    doc, _ = doc_from_record(rec)
    f = doc.layout[0].fields[0]
    assert f.field_type == "checkbox" and f.single_choice is False
    assert [(o.line_id, o.checked, o.box) for o in f.options] == [
        (1, None, [205, 10, 225, 30]), (2, True, [325, 10, 345, 30])]
    dumped = doc.model_dump(mode="json")
    assert not any(k in ln for ln in dumped["lines"] for k in ("field_index", "checked", "box", "tick_cell_id"))


# ---------------------------------------------------------------------------
# choice_grid (gov_005 p1 (अ): 2 x 8, option names over empty tick cells)
# ---------------------------------------------------------------------------

NAMES = ["क) आदिवासी जनजाती", "ख) दलित", "ग) मधेशी", "घ) मुसलमान",
         "ङ) थारु", "च) पिछडा वर्ग", "छ) खस आर्य", "ज) अन्य"]


def _grid_page(choice_grid, on="table"):
    cells = []
    for r in range(2):
        for c in range(8):
            b = [100 + c * 100, 300 + r * 50, 200 + c * 100, 350 + r * 50]
            cells.append({"id": r * 8 + c, "row": r, "col": c, "rowspan": 1, "colspan": 1,
                          "text": NAMES[c] if r == 0 else "", "is_header": r == 0,
                          "handwritten": False, "empty": r == 1, "bbox": b, "quad": q(b)})
    table = {"bbox": [100, 300, 900, 400], "quad": q([100, 300, 900, 400]), "num_rows": 2, "num_cols": 8,
             "header_rows": 1, "ruling_style": "full", "theme": None, "col_types": None, "html": "",
             "reading_order": list(range(16)), "cells": cells}
    t_region = region(2, "table", [100, 300, 900, 400], 1, table_index=0)
    if on == "table":
        t_region["choice_grid"] = choice_grid
    else:
        table["choice_grid"] = choice_grid
    lines = [line(NAMES[c], [105 + c * 100, 305, 195 + c * 100, 345], 2, c, table_index=0, cell_id=c)
             for c in range(8)]
    return make_page(layout=[region(1, "table_heading", [100, 250, 900, 295], 0,
                                    "(अ) वर्ग / समूह: (कुनै एकमा √ लगाउनुहोस्)"), t_region],
                     lines=lines, tables=[table])


def _check_grid(doc):
    t = next(el for el in doc.layout if el.type == "table")
    assert len(t.fields) == 1
    f = t.fields[0]
    assert f.field_type == "checkbox" and f.value is None and f.single_choice is True
    assert f.label.region_id == 1
    assert [o.tick_cell_id for o in f.options] == list(range(8, 16))
    assert [o.checked for o in f.options] == [None] * 8
    by_id = {ln.id: ln for ln in doc.lines}
    assert [by_id[o.line_id].text for o in f.options] == NAMES
    assert all(by_id[o.line_id].role == "option" for o in f.options)
    assert "choice_grid" not in json.dumps(doc.model_dump(mode="json"))


def test_choice_grid_rows_on_table_region():
    doc, warns = doc_from_record(_grid_page({"option_row": 0, "tick_row": 1, "label_region_id": 1}))
    _check_grid(doc)
    assert "choice_grid of table 0 converted" in messages(warns)


def test_choice_grid_explicit_pairs_on_tables_entry():
    cg = {"region_id": 1, "options": [{"cell_id": c, "tick_cell_id": 8 + c} for c in range(8)]}
    doc, _ = doc_from_record(_grid_page(cg, on="tables"))
    _check_grid(doc)


def test_unreadable_choice_grid_is_reported_not_silently_dropped():
    doc, warns = doc_from_record(_grid_page("something else"))
    assert any(w["level"] == "error" and "choice_grid could not be read" in w["message"] for w in warns)


# ---------------------------------------------------------------------------
# idempotent, round trip, save path
# ---------------------------------------------------------------------------

def test_migrate_is_idempotent():
    for rec in (_old_checkbox_page(), _grid_page({"option_row": 0, "tick_row": 1, "label_region_id": 1})):
        once, _ = migrate(rec)
        twice, warns = migrate(once)
        assert twice == once and warns == []


def test_round_trip_is_stable():
    doc, _ = doc_from_record(_grid_page({"option_row": 0, "tick_row": 1, "label_region_id": 1}))
    a = finalize(doc).model_dump(mode="json")
    b = finalize(doc_from_record(a)[0]).model_dump(mode="json")
    assert a == b


def test_form_text_lists_option_words_without_glyphs():
    """Region text = its lines joined with "\\n" (never built from fields)."""
    doc = finalize(doc_from_record(_old_checkbox_page())[0])
    assert doc.layout[0].text == "(ऊ) अपाङ्गता भएका सदस्य: (कुनै एकमा √ लगाउनुहोस्)\nछ\nछैन"


def test_checkbox_list_without_question_is_valid():
    rec = make_page(layout=[region(1, "list", [0, 0, 400, 100], 0, list_style="dev", items=[
        {"marker": "क)", "level": 0, "text": "नयाँ"}, {"marker": "ख)", "level": 0, "text": "नविकरण"}],
        fields=[{"label": None, "field_type": "checkbox", "single_choice": False, "value": None,
                 "options": [{"line_id": 40, "box": [300, 5, 320, 25], "checked": None},
                             {"line_id": 41, "box": [300, 55, 320, 75], "checked": None}]}])],
        lines=[line("क) नयाँ", [0, 0, 200, 40], 1, 0, id=40, role="option"),
               line("ख) नविकरण", [0, 50, 200, 90], 1, 1, id=41, role="option")])
    doc, warns = doc_from_record(rec)
    f = doc.layout[0].fields[0]
    assert f.label is None and len(f.options) == 2
    assert f.options[0].box_quad == q([300, 5, 320, 25])
    dumped = doc.model_dump(mode="json")["layout"][0]["fields"][0]
    assert list(dumped) == ["label", "field_type", "single_choice", "value", "options"]
    assert Doc.model_validate(doc.model_dump(mode="json")) == doc


def test_save_path_keeps_options_and_line_ids():
    """_clean_payload trims every object to its model's keys: the new field
    keys must survive it, and an editor sending old shapes gets migrated."""
    from types import SimpleNamespace

    from app.services.annotations import _clean_payload

    server_doc, _ = doc_from_record(_grid_page({"option_row": 0, "tick_row": 1, "label_region_id": 1}))
    row = SimpleNamespace(document_id=server_doc.document_id, image_key=server_doc.image, doc=server_doc)
    cleaned = _clean_payload(server_doc.model_dump(mode="json"), row)
    _check_grid(Doc.model_validate(cleaned))

    old = _old_checkbox_page()  # an old editor tab sending the old shape
    old_doc = doc_from_record(old)[0]
    row = SimpleNamespace(document_id=old_doc.document_id, image_key=old_doc.image, doc=old_doc)
    f = Doc.model_validate(_clean_payload(old, row)).layout[0].fields[0]
    assert f.value is None and len(f.options) == 2
