"""Line boxes measured from the page image (detect_lines / ink_mask / snap_box_to_ink)."""
from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from app.schemas.doc_v2 import Doc
from app.services.lines import detect_lines, detect_page_lines, ink_mask, snap_box_to_ink
from conftest import blank, ink, make_page, write

PAD = 6
PARA = [100, 100, 700, 330]


def _paragraph():
    """Centered line, indented first line, full line, short last line."""
    g = blank()
    inks = [
        write(g, 330, 120, "Centered title"),
        write(g, 160, 180, "Indented first line of text"),
        write(g, 120, 230, "and the next full line here."),
        write(g, 120, 280, "Short."),
    ]
    return g, inks


def _region(text, rid=0, typ="paragraph", bbox=PARA):
    return {"id": rid, "type": typ, "bbox": list(bbox), "text": text}


def _whole_block_line(text, rid=0, bbox=PARA, i=0):
    return {"text": text, "bbox": list(bbox), "block_id": rid, "line_in_block": i,
            "is_heading": False, "font_size": 20.0, "font_id": None}


def _inside(inner, outer):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


# ---------------------------------------------------------------------------
# the required test: each line's x-extent is that line's own ink
# ---------------------------------------------------------------------------

def test_each_line_box_follows_its_own_ink():
    g, inks = _paragraph()
    boxes = detect_lines(g, PARA)
    assert len(boxes) == 4
    for box, (ix0, iy0, ix1, iy1) in zip(boxes, inks):
        # box = the line's ink + the pad, within 3 px; never the block's edges
        assert abs((box[0] + PAD) - ix0) <= 3, (box, ix0)
        assert abs((box[2] - PAD) - ix1) <= 3, (box, ix1)
        assert box[0] <= ix0 and box[2] >= ix1 and box[1] <= iy0 and box[3] >= iy1
        assert _inside(box, PARA)
    centered, indented, full, short = boxes
    assert centered[0] > PARA[0] + 200          # starts near the middle, not at the block edge
    assert indented[0] > full[0] + 30           # the indent is kept
    assert short[2] < full[2] - 200             # the short last line ends early


def test_tight_without_pad():
    g, inks = _paragraph()
    for box, ink_box in zip(detect_lines(g, PARA, pad=0), inks):
        assert all(abs(a - b) <= 3 for a, b in zip(box, ink_box)), (box, ink_box)


# ---------------------------------------------------------------------------
# ink_mask: what is NOT text
# ---------------------------------------------------------------------------

def test_underline_does_not_stretch_the_line():
    g = blank()
    title = write(g, 300, 120, "Title")
    ink(g, (150, 155, 650, 158))                # underline far wider than the title
    (box,) = detect_lines(g, PARA)
    assert abs((box[0] + PAD) - title[0]) <= 3 and abs((box[2] - PAD) - title[2]) <= 3


def test_page_frame_and_table_border_are_ignored():
    g = blank()
    text = write(g, 300, 200, "Inside a frame")
    ink(g, (102, 102, 698, 105)); ink(g, (102, 325, 698, 328))   # frame top / bottom
    ink(g, (102, 102, 105, 328)); ink(g, (695, 102, 698, 328))   # frame left / right
    (box,) = detect_lines(g, PARA)
    assert abs((box[0] + PAD) - text[0]) <= 3 and abs((box[2] - PAD) - text[2]) <= 3


def test_specks_do_not_pull_the_box_sideways():
    g = blank()
    text = write(g, 300, 200, "Text")
    ink(g, (130, 210, 132, 212))                # dust dot on the same rows
    ink(g, (660, 205, 662, 207))
    (box,) = detect_lines(g, PARA)
    assert abs((box[0] + PAD) - text[0]) <= 3 and abs((box[2] - PAD) - text[2]) <= 3


def test_faded_print_is_still_found():
    g = blank()
    write(g, 200, 200, "Faded gray print")
    g[g < 128] = 170                            # lighter than the old fixed threshold (140)
    assert len(detect_lines(g, PARA)) == 1
    assert ink_mask(g[100:330, 100:700]).any()


# ---------------------------------------------------------------------------
# region matching
# ---------------------------------------------------------------------------

def test_region_text_split_at_newlines_gives_one_record_per_line():
    g, inks = _paragraph()
    texts = ["Centered title", "Indented first line of text", "and the next full line here.", "Short."]
    rec, warnings = detect_page_lines(g, make_page(layout=[_region("\n".join(texts))]))
    assert warnings == []
    assert [ln["text"] for ln in rec["lines"]] == texts
    assert [ln["line_in_block"] for ln in rec["lines"]] == [0, 1, 2, 3]
    for ln, ink_box in zip(rec["lines"], inks):
        assert abs((ln["bbox"][0] + PAD) - ink_box[0]) <= 3
        assert ln["font_size"] == round(0.8 * (ln["bbox"][3] - ln["bbox"][1]), 1)
        assert ln["quad"][0] == ln["bbox"][:2] and ln["quad"][2] == ln["bbox"][2:]
    Doc.model_validate(rec)


def test_placeholder_strips_are_replaced_by_measured_boxes():
    # what the editor makes when you type 4 lines: even strips over the block
    g, _ = _paragraph()
    texts = ["a", "b", "c", "d"]
    strips = [_whole_block_line(t, bbox=[100, 100 + 57 * i, 700, 157 + 57 * i], i=i) for i, t in enumerate(texts)]
    rec, warnings = detect_page_lines(g, make_page(layout=[_region("a b c d")], lines=strips))
    assert warnings == []
    assert [ln["text"] for ln in rec["lines"]] == texts
    assert all(ln["bbox"][0] != 100 or ln["bbox"][2] != 700 for ln in rec["lines"])


def test_count_mismatch_warns_and_leaves_lines_alone():
    g, _ = _paragraph()
    old = [_whole_block_line("all of it")]
    rec, warnings = detect_page_lines(g, make_page(layout=[_region("all of it")], lines=copy.deepcopy(old)))
    assert rec["lines"] == old
    assert warnings == ["block 0: 1 text lines but 4 printed lines — add line breaks; lines left unchanged"]


def test_other_regions_and_table_lines_are_not_touched():
    g, _ = _paragraph()
    other = _whole_block_line("other", rid=1, bbox=[720, 100, 950, 150])
    table_line = dict(_whole_block_line("x"), table_index=0, cell_id=0)
    rec = make_page(layout=[_region("a\nb\nc\nd"), _region("other", rid=1, bbox=[720, 100, 950, 150])],
                    lines=[copy.deepcopy(other), copy.deepcopy(table_line)])
    rec, _ = detect_page_lines(g, rec, region_ids={0})
    assert other in rec["lines"] and table_line in rec["lines"]
    assert len(rec["lines"]) == 6


def test_heading_marker_is_ignored():
    g = blank()
    write(g, 200, 110, "Heading")
    rec, warnings = detect_page_lines(
        g, make_page(layout=[{**_region("## Heading", typ="heading", bbox=[100, 100, 600, 160]), "level": 2}]))
    assert warnings == []
    assert rec["lines"][0]["text"] == "Heading" and rec["lines"][0]["is_heading"] is True


def test_form_field_lines_come_from_region_text_not_fields():
    """Bugfix: every printed line of a form_field gets a line record, even the
    parts that are no field. The old label line keeps its id and role, so
    label.line_id still points at it."""
    g = blank()
    write(g, 110, 310, "4. Permanent address:")
    write(g, 110, 370, "Province: Lumbini, District: Rukum")
    ff = {"id": 3, "type": "form_field", "bbox": [100, 300, 900, 420],
          "text": "4. Permanent address:\nProvince: Lumbini, District: Rukum",
          "fields": [{"label": {"text": "4. Permanent address:", "line_id": 7}, "field_type": "text",
                      "value": None}]}
    old = dict(_whole_block_line("4. Permanent address:", rid=3, bbox=[100, 300, 500, 350]),
               id=7, role="label")
    rec, warnings = detect_page_lines(g, make_page(layout=[ff], lines=[old]))
    assert warnings == []
    assert [ln["text"] for ln in rec["lines"]] == ["4. Permanent address:",
                                                   "Province: Lumbini, District: Rukum"]
    first, second = rec["lines"]
    assert (first["id"], first["role"]) == (7, "label")
    assert "role" not in second and "id" not in second  # a new line: the save gives it an id


def test_form_field_line_count_mismatch_leaves_lines_alone():
    box = [100, 300, 900, 360]
    old = dict(_whole_block_line("Name Ram", rid=3, bbox=box), id=4, role="label")
    ff = {"id": 3, "type": "form_field", "bbox": box, "text": "Name:\nRam",
          "fields": [{"label": {"text": "Name", "bbox": box}, "field_type": "text",
                      "value": {"text": "Ram", "bbox": box, "handwritten": False}}]}
    rec, warnings = detect_page_lines(blank(), make_page(layout=[ff], lines=[copy.deepcopy(old)]))
    assert rec["lines"] == [old]
    assert "lines left unchanged" in warnings[0]


# ---------------------------------------------------------------------------
# several text rows on ONE printed row (gov_005 page 2 layouts)
# ---------------------------------------------------------------------------

ROW = [50, 300, 990, 370]   # the test page is 1000 px wide


def test_side_by_side_lines_that_match_the_text_are_kept():
    """Block 1: three label lines drawn side by side on one printed row."""
    g = blank()
    boxes = [write(g, 60, 315, "Pregnant:", 22), write(g, 330, 315, "months or after birth:", 22),
             write(g, 850, 315, "months", 22)]
    old = [dict(_whole_block_line(t, rid=1, bbox=[b[0] - 6, b[1] - 6, b[2] + 6, b[3] + 6], i=i),
                id=126 + i, role="label")
           for i, (t, b) in enumerate(zip(["Pregnant:", "months or after birth:", "months"], boxes))]
    ff = _region("Pregnant:\nmonths or after birth:\nmonths", rid=1, typ="form_field", bbox=ROW)
    rec, warnings = detect_page_lines(g, make_page(layout=[ff], lines=copy.deepcopy(old)))
    assert warnings == []
    assert rec["lines"] == old


def test_one_printed_row_is_split_at_its_fill_in_gaps():
    """Block 5: label, a dot leader, then the unit word - no lines yet."""
    g = blank()
    label = write(g, 60, 315, "7. Days employed", 22)
    ink(g, (label[2] + 2, 330, label[2] + 7, 335))         # a small mark next to the word (like a matra dot)
    label = [label[0], label[1], label[2] + 7, label[3]]
    write(g, 320, 315, "......................", 22)      # the leader: not a word
    unit = write(g, 880, 315, "days", 22)
    ff = _region("7. Days employed\ndays", rid=5, typ="form_field", bbox=ROW)
    rec, warnings = detect_page_lines(g, make_page(layout=[ff]))
    assert warnings == []
    first, second = rec["lines"]
    assert (first["text"], second["text"]) == ("7. Days employed", "days")
    json.dumps(rec)  # plain Python numbers only (numpy ints broke the API response)
    assert all(type(v) is int for ln in rec["lines"] for v in ln["bbox"])
    # each box is measured from its own ink, not from the region
    assert abs((first["bbox"][0] + PAD) - label[0]) <= 3 and abs((first["bbox"][2] - PAD) - label[2]) <= 3
    assert abs((second["bbox"][0] + PAD) - unit[0]) <= 3 and abs((second["bbox"][2] - PAD) - unit[2]) <= 3


def test_ambiguous_row_is_not_split():
    """Ordinary word gaps only: which word goes in which row is a guess."""
    g = blank()
    write(g, 60, 315, "Province Lumbini District Rukum", 22)
    ff = _region("Province Lumbini\nDistrict Rukum", rid=2, typ="form_field", bbox=ROW)
    rec, warnings = detect_page_lines(g, make_page(layout=[ff]))
    assert rec["lines"] == []
    assert warnings == ["block 2: 2 text lines but 1 printed lines — rows that share one printed line:"
                        " join them, or draw each with the Line tool; lines left unchanged"]


def test_rows_with_no_line_get_lines_and_the_old_line_keeps_its_id():
    """gov_005 header: 4 printed rows, text has all 4, only the first has a line."""
    g = blank()
    rows = ["Ministry of Labour", "Department of Safety", "Minbhawan, Kathmandu", "Governance form"]
    for i, t in enumerate(rows):
        write(g, 200, 120 + 50 * i, t)
    hd = _region("\n".join(rows), rid=0, typ="header", bbox=[100, 100, 900, 330])
    old = dict(_whole_block_line(rows[0], rid=0, bbox=[190, 112, 600, 160]), id=42)
    rec, warnings = detect_page_lines(g, make_page(layout=[hd], lines=[old]))
    assert warnings == []
    assert [ln["text"] for ln in rec["lines"]] == rows
    assert rec["lines"][0]["id"] == 42 and all("id" not in ln for ln in rec["lines"][1:])


# ---------------------------------------------------------------------------
# snap_box_to_ink
# ---------------------------------------------------------------------------

def test_rough_box_snaps_to_the_ink():
    g = blank()
    text = write(g, 300, 200, "Snap me")
    box = snap_box_to_ink(g, [250, 170, 600, 260])
    assert all(abs(a - b) <= 3 for a, b in zip(box, [text[0] - PAD, text[1] - PAD, text[2] + PAD, text[3] + PAD]))


def test_empty_box_is_unchanged_and_clip_to_parent():
    g = blank()
    assert snap_box_to_ink(g, [10, 10, 90, 90]) == [10, 10, 90, 90]
    write(g, 300, 200, "Edge")
    box = snap_box_to_ink(g, [250, 170, 600, 260], clip_to=[302, 0, 1000, 1000])
    assert box[0] == 302


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


def test_detect_endpoint_uses_the_editor_doc(pagestore, client):
    g, _ = _paragraph()
    pagestore(make_page(layout=[_region("old")]), g)
    editor = make_page(layout=[_region("a\nb\nc\nd")])  # unsaved edit
    r = client.post("/documents/DOC-0001/detect-lines", json={"doc": editor, "element_id": 0})
    assert r.status_code == 200, r.text
    assert r.json()["warnings"] == []
    assert [ln["text"] for ln in r.json()["record"]["lines"]] == ["a", "b", "c", "d"]


def test_detect_endpoint_unknown_region_is_404(pagestore, client):
    pagestore(make_page(), blank())
    assert client.post("/documents/DOC-0001/detect-lines", json={"element_id": 99}).status_code == 404


def test_snap_box_endpoint(pagestore, client):
    g = blank()
    text = write(g, 300, 200, "Snap me")
    pagestore(make_page(), g)
    r = client.post("/documents/DOC-0001/snap-box", json={"bbox": [250, 170, 600, 260]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True
    assert abs(body["bbox"][0] - (text[0] - PAD)) <= 3
