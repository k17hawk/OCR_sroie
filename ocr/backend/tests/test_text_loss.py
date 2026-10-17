"""Bugfix: form_field text was rebuilt from fields[] on save, deleting printed
text (gov_005 page 2, item ४ saved as only "४. स्थायी ठेगाना:").

Rules (forms prompt 1b, 5, 6, 7, 9):
  * region text = its lines joined with "\\n", for every region type;
    never derived from fields (fields only reference lines);
  * saving compares printed characters before / after (ignoring leaders,
    tabs, ☐/✓ and spaces) and is blocked when anything else is lost.
"""
from __future__ import annotations

import copy

import pytest

from app.core.finalize import finalize
from app.core.textloss import find_text_loss, printed_chars
from app.schemas.doc_v2 import Doc
from conftest import make_page
from test_per_file_folders import _jpg, _project, _upload, client  # noqa: F401  (fixture)

ITEM_4 = ("४. स्थायी ठेगाना: प्रदेश: लुम्बिनी, जिल्ला: रुकुम (पूर्व), गा.पा.: भूमे, "
          "वडा नं.: ........, टोल: ......")
LABEL_4 = "४. स्थायी ठेगाना:"
BOX = [100, 300, 900, 360]


def _line(text, block_id, i, bbox=BOX, **kw):
    return {"text": text, "bbox": bbox, "block_id": block_id, "line_in_block": i,
            "is_heading": False, "font_size": 20.0, "font_id": None, **kw}


def _item4(lines=None, text=ITEM_4):
    """Item ४ as a form_field whose only field is the printed label."""
    ff = {"id": 0, "type": "form_field", "bbox": BOX, "reading_order": 0, "text": text,
          "fields": [{"label": {"text": LABEL_4, "bbox": [100, 300, 330, 360]},
                      "field_type": "text", "value": None}]}
    return make_page(layout=[ff], lines=lines if lines is not None else [])


def _final(rec):
    return finalize(Doc.model_validate(rec)).model_dump(mode="json")


# --- region text is never rebuilt from fields ---------------------------------

def test_item_4_keeps_full_text_when_region_has_no_lines():
    out = _final(_item4())
    assert out["layout"][0]["text"] == ITEM_4
    assert out["text"] == ITEM_4


def test_item_4_text_is_its_lines_joined_with_newline():
    parts = [LABEL_4, "प्रदेश: लुम्बिनी, जिल्ला: रुकुम (पूर्व), गा.पा.: भूमे,",
             "वडा नं.: ........, टोल: ......"]
    lines = [_line(t, 0, i, role="label" if i == 0 else None) for i, t in enumerate(parts)]
    out = _final(_item4(lines=[{k: v for k, v in ln.items() if v is not None} for ln in lines]))
    assert out["layout"][0]["text"] == "\n".join(parts)
    assert out["layout"][0]["fields"][0]["label"]["text"] == LABEL_4  # fields untouched


@pytest.mark.parametrize("rtype", ["paragraph", "heading", "form_field", "list", "header"])
def test_every_region_type_uses_its_lines(rtype):
    el = {"id": 0, "type": rtype, "bbox": BOX, "reading_order": 0, "text": "पुरानो"}
    if rtype == "heading":
        el["level"] = 2
    if rtype == "form_field":
        el["fields"] = []
    if rtype == "list":
        el.update(list_style="numbered", items=[{"marker": "१.", "level": 0, "text": "क"}])
    rec = make_page(layout=[el], lines=[_line("एक", 0, 0), _line("दुई", 0, 1)])
    assert _final(rec)["layout"][0]["text"] == "एक\nदुई"


def test_table_text_uses_cell_lines_in_cell_reading_order():
    cells = [{"id": i, "row": 0, "col": i, "rowspan": 1, "colspan": 1, "text": t,
              "is_header": False, "handwritten": False, "empty": False,
              "bbox": [100 + 100 * i, 300, 200 + 100 * i, 360]}
             for i, t in enumerate(["क्र.\nसं.", "नाम"])]
    table = {"bbox": [100, 300, 300, 360], "num_rows": 1, "num_cols": 2, "header_rows": 0,
             "ruling_style": "full", "html": "", "reading_order": [0, 1], "cells": cells}
    lines = [_line("नाम", 0, 0, table_index=0, cell_id=1),
             _line("क्र.", 0, 1, table_index=0, cell_id=0),
             _line("सं.", 0, 2, table_index=0, cell_id=0)]
    rec = make_page(layout=[{"id": 0, "type": "table", "bbox": [100, 300, 300, 360],
                             "reading_order": 0, "text": "", "table_index": 0}],
                    tables=[table], lines=lines)
    out = _final(rec)
    assert out["layout"][0]["text"] == "क्र.\nसं.\nनाम"
    assert out["tables"][0]["html"] == "<table><tr><td>क्र.<br>सं.</td><td>नाम</td></tr></table>"


def test_finalize_is_still_idempotent_with_line_text():
    rec = _item4(lines=[_line(LABEL_4, 0, 0, role="label")], text=LABEL_4)
    once = _final(rec)
    assert _final(once) == once


# --- the printed-character check ---------------------------------------------

def test_printed_chars_ignore_fill_in_marks():
    assert printed_chars("वडा नं.: ........, टोल: …… \t ☐ ✓ ✔ ____ \\t") == printed_chars("वडानं.:,टोल:")


def test_fixing_a_typed_backslash_n_is_not_a_loss():
    """gov_005 p2 item ४ had a typed "\\n" (backslash + n) in its text."""
    typed = "४. स्थायी ठेगाना:\\nप्रदेश: लुम्बिनी"
    assert printed_chars(typed) == printed_chars("४. स्थायी ठेगाना:\nप्रदेश: लुम्बिनी")


def test_printed_chars_keep_real_dots():
    assert printed_chars("क्र. सं.")["."] == 2


def test_text_loss_reports_what_disappears():
    before = _item4()
    after = copy.deepcopy(before)
    after["layout"][0]["text"] = LABEL_4  # what the old finalize produced
    [issue] = find_text_loss(before, after)
    assert issue["region_id"] == 0
    for word in ("प्रदेश", "लुम्बिनी", "रुकुम", "भूमे", "टोल"):
        assert all(ch in issue["lost"] for ch in word)
    assert issue["after"] == LABEL_4


def test_moving_text_into_lines_is_not_a_loss():
    before = _item4()
    after = _item4(lines=[_line(ITEM_4, 0, 0)], text="")
    assert find_text_loss(before, after) == []


def test_only_leaders_removed_is_not_a_loss():
    before = _item4()
    after = _item4(text=ITEM_4.replace("........", "").replace("......", ""))
    assert find_text_loss(before, after) == []


def test_old_v2_markup_is_not_counted_as_lost():
    """v2 pages stored derived markup: '## ' headings, table html, '- ' bullets."""
    heading = {"id": 0, "type": "heading", "bbox": BOX, "reading_order": 0,
               "text": "## शीर्षक", "level": 2}
    before = make_page(layout=[heading])
    after = make_page(layout=[dict(heading, text="शीर्षक")], lines=[_line("शीर्षक", 0, 0)])
    assert find_text_loss(before, after) == []


# --- end to end: the save endpoint -------------------------------------------

def _uploaded(client):  # noqa: F811
    pid = _project(client)
    [doc_id] = _upload(client, pid, "gov_005.jpg", _jpg())["document_ids"]
    got = client.get(f"/documents/{doc_id}").json()
    return doc_id, got["doc"], got["revision"]


def _with_item4(doc, lines, text=ITEM_4):
    doc["layout"] = [{"id": 0, "type": "form_field", "bbox": [100, 300, 800, 360],
                      "reading_order": 0, "text": text,
                      "fields": [{"label": {"text": LABEL_4, "bbox": [100, 300, 330, 360]},
                                  "field_type": "text", "value": None}]}]
    doc["lines"] = [{k: v for k, v in ln.items() if v is not None} for ln in lines]
    return doc


def test_save_keeps_item_4_full_text(client):  # noqa: F811
    doc_id, doc, rev = _uploaded(client)
    r = client.put(f"/documents/{doc_id}/annotation",
                   json={"doc": _with_item4(doc, []), "revision": rev})
    assert r.status_code == 200, r.text
    saved = client.get(f"/documents/{doc_id}").json()["doc"]
    assert saved["layout"][0]["text"] == ITEM_4


def test_save_is_blocked_when_printed_text_would_be_lost(client):  # noqa: F811
    """Lines cover only the label: saving would cut item ४ down to its label."""
    doc_id, doc, rev = _uploaded(client)
    doc = _with_item4(doc, [_line(LABEL_4, 0, 0, bbox=[100, 300, 330, 360], role="label")])
    r = client.put(f"/documents/{doc_id}/annotation", json={"doc": doc, "revision": rev})
    assert r.status_code == 422, r.text
    [issue] = r.json()["detail"]
    assert issue["type"] == "text_loss" and issue["region_id"] == 0
    assert all(ch in issue["got"] for ch in "लुम्बिनी")
    # nothing was written
    assert client.get(f"/documents/{doc_id}").json()["revision"] == rev


# --- dry-run script on page files --------------------------------------------

def test_check_script_restores_text_from_lines_and_blocks_loss(tmp_path, capsys):
    import json as _json

    from scripts import check_text_loss

    # saved by the old code: text cut to the label, but its lines hold it all
    cut = _item4(lines=[_line(LABEL_4, 0, 0, role="label"), _line(ITEM_4[len(LABEL_4) + 1:], 0, 1)],
                 text=LABEL_4)
    full_but_label_line_only = _item4(lines=[_line(LABEL_4, 0, 0, role="label")])
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(_json.dumps(cut, ensure_ascii=False), encoding="utf-8")
    b.write_text(_json.dumps(full_but_label_line_only, ensure_ascii=False), encoding="utf-8")

    assert check_text_loss.main([str(a)]) == 0
    assert "लुम्बिनी" in capsys.readouterr().out          # the save brings it back
    assert check_text_loss.main([str(b)]) == 1
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "प्रदेश: लुम्बिनी" in out
    assert _json.loads(b.read_text(encoding="utf-8")) == full_but_label_line_only  # nothing written
