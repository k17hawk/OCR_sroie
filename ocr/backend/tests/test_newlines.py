"""A new line between printed lines survives save and reload (spec section 5,
second prompt section 8): region text joins its lines with "\\n", never a space,
and a list item printed on two lines keeps both rows."""
from __future__ import annotations

from app.core.finalize import finalize
from app.storage.doc_store import doc_from_record

from conftest import make_page

Q13 = "१३. परिवारमा काम गर्ने उमेर समूह (१८ देखि ५९ वर्ष) को र काम गर्न सक्ने शारीरिक अवस्था भएको परिवारको सदस्य"
INSTR = "(कुनै एकमा √ लगाउनुहोस्)"


def q(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def ln(text, bbox, block, lib):
    return {"text": text, "bbox": bbox, "block_id": block, "line_in_block": lib, "is_heading": False,
            "font_size": 18.4, "font_id": None, "quad": q(bbox)}


def _round_trip(rec):
    a = finalize(doc_from_record(rec)[0]).model_dump(mode="json")
    b = finalize(doc_from_record(a)[0]).model_dump(mode="json")
    assert a == b
    return a


def test_paragraph_lines_are_joined_by_newline_and_survive_reload():
    rec = make_page(layout=[{"id": 1, "type": "paragraph", "bbox": [110, 1065, 980, 1155], "reading_order": 0,
                             "text": "", "quad": q([110, 1065, 980, 1155])}],
                    lines=[ln(Q13, [116, 1069, 975, 1092], 1, 0), ln(INSTR, [186, 1128, 381, 1151], 1, 1)])
    out = _round_trip(rec)
    assert out["layout"][0]["text"] == Q13 + "\n" + INSTR
    assert out["text"] == Q13 + "\n" + INSTR


def test_list_item_printed_on_two_lines_keeps_both_rows():
    rec = make_page(layout=[{"id": 1, "type": "list", "bbox": [100, 100, 900, 300], "reading_order": 0, "text": "",
                             "list_style": "dev", "quad": q([100, 100, 900, 300]),
                             "items": [{"marker": "१३.", "level": 0, "text": Q13[4:] + "\n" + INSTR},
                                       {"marker": "क)", "level": 1, "text": "उप-बुँदा"}]}])
    out = _round_trip(rec)
    assert out["layout"][0]["text"] == f"१३. {Q13[4:]}\n  {INSTR}\n  क) उप-बुँदा"
    assert out["layout"][0]["items"][0]["text"] == Q13[4:] + "\n" + INSTR
