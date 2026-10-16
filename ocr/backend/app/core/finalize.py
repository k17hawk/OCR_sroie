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
    serialize_form_field,
    serialize_list,
    serialize_table_html,
)
from app.schemas.doc_v2 import (
    Block,
    CheckboxExtraRegion,
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
        # keep the page's spelling ("dev" in the reference data)
        return current if current in ("dev", "devanagari") else "devanagari"
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
    lines_by_id: dict[int, Line] = {ln.id: ln for ln in doc.lines if ln.id is not None}

    def _line_text(line_id: int) -> str | None:
        ln = lines_by_id.get(line_id)
        return ln.text.strip() if ln is not None else None

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
            el.text = serialize_form_field(el.fields, _line_text)

        elif isinstance(el, ListLayout):
            pass  # after the loop: needs its children's final text

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
                # wrapped lines of one region read as one text, joined by a
                # space (as in the reference pages), not by a newline
                el.text = " ".join(ln.text.strip() for ln in block_lines if ln.text.strip())
            # else: no lines — keep el.text as typed by the human

    # ------------------------------------------------------------------
    # 1b) Lists: items may own other boxes (a note, a figure, …) through
    #     ``children``. Keep parent_id / parent_item on those boxes in step
    #     with ``children`` (children win), then build the list text with
    #     each child's text indented under its item.
    # ------------------------------------------------------------------
    by_id = {el.id: el for el in doc.layout}
    owned: dict[int, tuple[int, int]] = {}
    for el in doc.layout:
        if not isinstance(el, ListLayout):
            continue
        for i, item in enumerate(el.items):
            if item.children is None:
                continue
            kept = [c for c in item.children if c in by_id and c != el.id and c not in owned]
            for c in kept:
                owned[c] = (el.id, i)
            item.children = kept or None
    for el in doc.layout:
        if el.id in owned:
            el.parent_id, el.parent_item = owned[el.id]
            el.reading_order = None  # read as part of the list item
        elif getattr(el, "parent_id", None) is not None:
            el.parent_id = el.parent_item = None

    def _child_text(cid: int) -> str | None:
        child = by_id.get(cid)
        return getattr(child, "text", None) if child is not None else None

    for el in doc.layout:
        if isinstance(el, ListLayout):
            el.text = serialize_list(el.items, _child_text)

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
    # (one per option with a printed square; checked null is written false)
    derived_checkboxes: list[CheckboxExtraRegion] = []
    for el in doc.layout:
        for field in getattr(el, "fields", None) or []:
            for opt in field.options or []:
                if opt.box is None or opt.box_quad is None:
                    continue
                derived_checkboxes.append(
                    CheckboxExtraRegion(
                        type="checkbox",
                        checked=bool(opt.checked),
                        option=_line_text(opt.line_id) or "",
                        bbox=opt.box,
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