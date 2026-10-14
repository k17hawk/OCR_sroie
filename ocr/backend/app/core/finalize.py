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
"""
from __future__ import annotations

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
    #
    # Rule: lines win when they exist; human-entered text survives when
    # they don't. This lets an annotator type/paste text into an element
    # that has no backing OCR lines (e.g. a region drawn over a bitmap
    # label, or a heading whose line was deleted) without finalize
    # silently blanking it.
    # ------------------------------------------------------------------
    for el in doc.layout:
        # bbox from quad — always.
        el.bbox = bbox_from_quad(el.quad)

        if isinstance(el, HeadingLayout):
            block_lines = lines_by_block.get(el.id, [])
            if block_lines:
                el.text = "#" * el.level + " " + block_lines[0].text
            # else: no lines — keep el.text as typed by the human

        elif isinstance(el, FigureLayout):
            el.text = "<figure/>"

        elif isinstance(el, FormFieldLayout):
            for field in el.fields:
                value = field.value
                if isinstance(value, CheckboxFieldValue):
                    value.text = serialize_checkbox_value(value.options)
            el.text = serialize_form_field(el.fields)

        elif isinstance(el, ListLayout):
            el.text = serialize_list(el.items)

        elif isinstance(el, TableLayout):
            table = doc.tables[el.table_index]
            table.html = serialize_table_html(table.cells)
            el.text = table.html

        elif isinstance(el, LogoLayout):
            # No ``text`` field on logos. Nothing to do.
            continue

        else:
            # header, paragraph, signature_caption, footer
            block_lines = lines_by_block.get(el.id, [])
            if block_lines:
                el.text = "\n".join(ln.text for ln in block_lines)
            # else: no lines — keep el.text as typed by the human

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
    # 4) Recompute extra_regions: derived checkboxes first, then the rest.
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
    non_checkbox = [e for e in doc.extra_regions if e.type != "checkbox"]
    doc.extra_regions = derived_checkboxes + non_checkbox

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