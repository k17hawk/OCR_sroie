"""The pure ``finalize(doc) -> doc`` function.

Every derived field in the schema is produced by exactly one code path in
this module. The exporter, the validator, and the frontend JSON preview all
call this — nothing else is allowed to derive.

Contract
--------
* ``finalize`` never mutates its argument. It returns a deep copy.
* ``finalize`` is idempotent: ``finalize(finalize(d)) == finalize(d)``.
* Every golden file in ``schema/`` and ``tests/fixtures/`` must satisfy
  ``finalize(Doc.model_validate(g)) == g`` at the JSON value level.

Fixes (annotator data was being lost on save)
---------------------------------------------
1. A list with no ``items`` (drawn with the List tool, text typed in the
   panel) had its text replaced by ``serialize_list([]) == ""``. Items are
   now parsed from the text first, so the list keeps what was typed.
2. Text boxes whose text is not backed by ``lines`` (OCR / import / JSON
   edits) were blanked by ``"\\n".join(<no lines>)``. Text is now rebuilt
   from lines only when the box has lines.
3. Signature and stamp boxes fell into the text branch, and setting
   ``.text`` on a model without that field raises — any page with a
   signature or stamp could not be saved. They are now skipped like logos.
4. Step 4 rebuilt checkboxes from form fields only, deleting standalone
   checkboxes. They are now kept (form-option checkboxes are still derived).
5. ``doc.tables[el.table_index]`` crashed (or picked the last table) for a
   table box without cell data (index -1 / out of range). Now skipped.

All fixes leave already-finalized documents unchanged, so the golden-file
contract still holds.
"""
from __future__ import annotations

import re

from app.core.geometry import bbox_from_quad
from app.core.serialize import (
    serialize_checkbox_value,
    serialize_form_field,
    serialize_list,
    serialize_table_html,
)
from app.schemas.doc_v2 import (
    Block,
    CheckboxExtraRegion,
    CheckboxFieldValue,
    Doc,
    FigureLayout,
    FormFieldLayout,
    HeadingLayout,
    Line,
    ListItem,
    ListLayout,
    LogoLayout,
    ParagraphLayout,
    RawLengths,
    SignatureLayout,
    StampLayout,
    TableLayout,
)


# ---------------------------------------------------------------------------
# Unicode helpers (kept here so finalize.py is self-contained)
# ---------------------------------------------------------------------------

_DEVANAGARI_START = 0x0900
_DEVANAGARI_END = 0x097F
_DEV_DIGIT_START = 0x0966
_DEV_DIGIT_END = 0x096F
_ASCII_DIGIT_START = 0x0030
_ASCII_DIGIT_END = 0x0039
_ASCII_UPPER_START = 0x0041
_ASCII_UPPER_END = 0x005A
_ASCII_LOWER_START = 0x0061
_ASCII_LOWER_END = 0x007A


def _has_devanagari(text: str) -> bool:
    return any(_DEVANAGARI_START <= ord(c) <= _DEVANAGARI_END for c in text)


def _has_latin_letter(text: str) -> bool:
    for c in text:
        o = ord(c)
        if _ASCII_UPPER_START <= o <= _ASCII_UPPER_END:
            return True
        if _ASCII_LOWER_START <= o <= _ASCII_LOWER_END:
            return True
    return False


def _has_dev_digits(text: str) -> bool:
    return any(_DEV_DIGIT_START <= ord(c) <= _DEV_DIGIT_END for c in text)


def _has_latin_digits(text: str) -> bool:
    return any(_ASCII_DIGIT_START <= ord(c) <= _ASCII_DIGIT_END for c in text)


def _derive_language(text: str, current: str) -> str:
    """Devanagari cannot be split into Nepali vs Hindi from Unicode alone.

    If the derived language is Devanagari-only and the input already says
    ``"ne"`` or ``"hi"``, we preserve the input. Otherwise we fall back to
    ``"ne"`` for Devanagari, ``"en"`` for Latin, ``"mixed"`` when both.
    """
    has_dev = _has_devanagari(text)
    has_lat = _has_latin_letter(text)
    if has_dev and has_lat:
        return "mixed"
    if has_dev:
        return current if current in ("ne", "hi") else "ne"
    if has_lat:
        return "en"
    return current


def _derive_digit_style(text: str, current: str) -> str:
    has_dev = _has_dev_digits(text)
    has_lat = _has_latin_digits(text)
    if has_dev and has_lat:
        return "mixed"
    if has_dev:
        return "devanagari"
    if has_lat:
        return "latin"
    return current


# ---------------------------------------------------------------------------
# List text -> items (fix 1)
# ---------------------------------------------------------------------------

_LIST_MARKER_RE = re.compile(
    r"^(\s*)((?:[०-९]+|[0-9]+)[.)]|\(?[क-ह]\)|\([०-९0-9]+\)|[a-zA-Z][.)]"
    r"|(?:[ivxlcdm]+|[IVXLCDM]+)[.)]|[-•*●▪◦–])\s+(.*)$"
)


def _items_from_text(text: str) -> list[ListItem]:
    """"१. a\\n२. b" -> [ListItem('१.', 0, 'a'), ListItem('२.', 0, 'b')].

    Two leading spaces (or a tab) per nesting level; a line without a
    marker continues the previous item.
    """
    items: list[ListItem] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        if not raw.strip():
            continue
        m = _LIST_MARKER_RE.match(raw)
        if m:
            marker = "•" if m.group(2) in ("-", "*", "●") else m.group(2)
            level = len(m.group(1).replace("\t", "  ")) // 2
            items.append(ListItem(marker=marker, level=level, text=m.group(3).strip()))
        elif items:
            items[-1].text = f"{items[-1].text} {raw.strip()}"
        else:
            items.append(ListItem(marker="", level=0, text=raw.strip()))
    return items


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

def finalize(doc: Doc) -> Doc:
    """Recompute every derived field on a deep copy of ``doc`` and return it."""
    doc = doc.model_copy(deep=True)

    # Index lines by block, ordered by ``line_in_block``.
    lines_by_block: dict[int, list[Line]] = {}
    for line in doc.lines:
        lines_by_block.setdefault(line.block_id, []).append(line)
    for bucket in lines_by_block.values():
        bucket.sort(key=lambda ln: ln.line_in_block)

    # ------------------------------------------------------------------
    # 1) Recompute bbox + text on every layout element.
    # ------------------------------------------------------------------
    for el in doc.layout:
        # bbox from quad — always.
        el.bbox = bbox_from_quad(el.quad)

        if isinstance(el, (LogoLayout, SignatureLayout, StampLayout)):
            # No ``text`` field on these (fix 3). Nothing to derive.
            continue

        if isinstance(el, HeadingLayout):
            block_lines = lines_by_block.get(el.id, [])
            if block_lines:
                el.text = "#" * el.level + " " + block_lines[0].text

        elif isinstance(el, FigureLayout):
            el.text = "<figure/>"

        elif isinstance(el, FormFieldLayout):
            for field in el.fields:
                value = field.value
                if isinstance(value, CheckboxFieldValue):
                    value.text = serialize_checkbox_value(value.options)
            if el.fields:
                el.text = serialize_form_field(el.fields)

        elif isinstance(el, ListLayout):
            # fix 1: an item-less list keeps its typed text as items
            if not el.items and el.text.strip():
                el.items = _items_from_text(el.text)
            if el.items:
                el.text = serialize_list(el.items)

        elif isinstance(el, TableLayout):
            # fix 5: only a table box that points at real cell data
            if 0 <= el.table_index < len(doc.tables):
                table = doc.tables[el.table_index]
                table.html = serialize_table_html(table.cells)
                el.text = table.html

        else:
            # header, paragraph, title, text_line, list_item, form_label, …
            # fix 2: rebuild from lines only when the box has lines
            block_lines = lines_by_block.get(el.id, [])
            if block_lines:
                el.text = "\n".join(ln.text for ln in block_lines)

    # ------------------------------------------------------------------
    # 2) Recompute blocks from layout (same order as layout).
    # ------------------------------------------------------------------
    doc.blocks = [
        Block(
            type=el.type,
            bbox=el.bbox,
            block_id=el.id,
            font_id=None,
            quad=el.quad,
        )
        for el in doc.layout
    ]

    # ------------------------------------------------------------------
    # 3) Recompute every table html (in case a table has no layout element).
    # ------------------------------------------------------------------
    for table in doc.tables:
        table.html = serialize_table_html(table.cells)

    # ------------------------------------------------------------------
    # 4) Recompute extra_regions: derived checkboxes first, then the
    #    standalone checkboxes the annotator drew (fix 4), then the rest.
    # ------------------------------------------------------------------
    derived_checkboxes: list[CheckboxExtraRegion] = []
    for el in doc.layout:
        if not isinstance(el, FormFieldLayout):
            continue
        for field in el.fields:
            value = field.value
            if not isinstance(value, CheckboxFieldValue):
                continue
            for opt in value.options:
                derived_checkboxes.append(
                    CheckboxExtraRegion(
                        type="checkbox",
                        checked=opt.checked,
                        option=opt.text,
                        bbox=bbox_from_quad(opt.box_quad),
                        quad=opt.box_quad,
                    )
                )
    derived_boxes = {tuple(cb.bbox) for cb in derived_checkboxes}
    standalone = [
        e for e in doc.extra_regions
        if e.type == "checkbox" and tuple(bbox_from_quad(e.quad)) not in derived_boxes
    ]
    for e in standalone:
        e.bbox = bbox_from_quad(e.quad)
    non_checkbox = [e for e in doc.extra_regions if e.type != "checkbox"]
    doc.extra_regions = derived_checkboxes + standalone + non_checkbox

    # ------------------------------------------------------------------
    # 5) Recompute the page text from the reading order.
    # ------------------------------------------------------------------
    ordered = sorted(
        (el for el in doc.layout if el.reading_order is not None),
        key=lambda el: el.reading_order,  # type: ignore[arg-type, return-value]
    )
    doc.text = "\n\n".join(
        el.text for el in ordered if getattr(el, "text", None) is not None
    )

    # ------------------------------------------------------------------
    # 6) Recompute quality counts, bboxes, and raw_lengths.
    # ------------------------------------------------------------------
    doc.quality.line_count = len(doc.lines)
    doc.quality.table_count = len(doc.tables)
    doc.quality.figure_count = sum(1 for el in doc.layout if isinstance(el, FigureLayout))

    doc.quality.stamp_bboxes = [
        el.bbox for el in doc.layout if isinstance(el, StampLayout)
    ]
    doc.quality.signature_bboxes = [
        el.bbox for el in doc.layout if isinstance(el, SignatureLayout)
    ]

    paragraph_chars = sum(
        len(el.text)
        for el in doc.layout
        if isinstance(el, ParagraphLayout)
    )
    list_items = sum(
        len(el.items) for el in doc.layout if isinstance(el, ListLayout)
    )
    form_fields = sum(
        len(el.fields) for el in doc.layout if isinstance(el, FormFieldLayout)
    )
    table_cells = sum(len(t.cells) for t in doc.tables)

    doc.quality.raw_lengths = RawLengths(
        paragraph_chars=paragraph_chars,
        list_items=list_items,
        form_fields=form_fields,
        table_cells=table_cells,
    )

    # ------------------------------------------------------------------
    # 7) Derive language and digit_style from the full page text.
    # ------------------------------------------------------------------
    doc.language = _derive_language(doc.text, doc.language)  # type: ignore[assignment]
    doc.digit_style = _derive_digit_style(doc.text, doc.digit_style)  # type: ignore[assignment]

    return doc