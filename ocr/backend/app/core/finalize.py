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

def _lines_by_block(doc: Doc) -> dict[int, list[Line]]:
    """Lines of each region, in reading order: table lines follow the
    table's cell reading order, then ``line_in_block``; other regions by
    ``line_in_block``. Lines with empty text are left out."""
    cell_rank: dict[tuple[int, int], int] = {}
    for ti, t in enumerate(doc.tables):
        order = t.reading_order or [c.id for c in t.cells]
        for rank, cid in enumerate(order):
            cell_rank[(ti, cid)] = rank
    out: dict[int, list[Line]] = {}
    for ln in doc.lines:
        if ln.text.strip():
            out.setdefault(ln.block_id, []).append(ln)

    def key(ln: Line) -> tuple[int, int]:
        if ln.table_index is not None and ln.cell_id is not None:
            return cell_rank.get((ln.table_index, ln.cell_id), 10**6), ln.line_in_block
        return 0, ln.line_in_block

    for bucket in out.values():
        bucket.sort(key=key)
    return out


def region_text_from_lines(lines: list[Line]) -> str | None:
    """A region's text: its lines joined with "\\n" (None when it has none)."""
    texts = [ln.text.strip() for ln in lines if ln.text.strip()]
    return "\n".join(texts) if texts else None


def drop_orphan_tables(doc: Doc) -> dict[int, int]:
    """Remove tables no table region points at any more, in place.

    Deleting a table region (or changing its type) used to leave its entry in
    ``doc.tables`` - with its cells, ruling lines and cell lines - so the saved
    JSON kept tables that are not in the layout tree, and the editor drew them.
    Kept tables are re-indexed in order; ``table_index`` on table regions,
    lines and ruling lines follows. Lines of a dropped table go too, unless
    their region still exists (then they just lose their cell link).
    Returns ``{old_index: new_index}`` for the kept tables.
    """
    n = len(doc.tables)
    used = sorted({el.table_index for el in doc.layout
                   if isinstance(el, TableLayout) and 0 <= el.table_index < n})
    remap = {old: new for new, old in enumerate(used)}
    if len(used) == n:
        return remap
    doc.tables = [doc.tables[i] for i in used]
    for el in doc.layout:
        if isinstance(el, TableLayout) and el.table_index in remap:
            el.table_index = remap[el.table_index]
    region_ids = {el.id for el in doc.layout}
    kept_lines = []
    for ln in doc.lines:
        if ln.table_index is not None and ln.table_index not in remap:
            if ln.block_id not in region_ids:
                continue                      # belonged to the deleted table only
            ln.table_index = None
            ln.cell_id = None
        elif ln.table_index is not None:
            ln.table_index = remap[ln.table_index]
        kept_lines.append(ln)
    doc.lines = kept_lines
    kept_rules = []
    for r in doc.ruling_lines:
        if r.table_index in remap:
            r.table_index = remap[r.table_index]
            kept_rules.append(r)
    doc.ruling_lines = kept_rules
    return remap


def finalize(doc: Doc) -> Doc:
    """Recompute every derived field on a deep copy of ``doc`` and return it."""
    doc = doc.model_copy(deep=True)

    # 0) Tables whose region was deleted are not part of the page.
    drop_orphan_tables(doc)

    # Index lines by block, in reading order inside the block.
    lines_by_block = _lines_by_block(doc)
    lines_by_id: dict[int, Line] = {ln.id: ln for ln in doc.lines if ln.id is not None}

    def _line_text(line_id: int) -> str | None:
        ln = lines_by_id.get(line_id)
        return ln.text.strip() if ln is not None else None

    # ------------------------------------------------------------------
    # 1) Recompute bbox + text on every layout element.
    #
    # Rule (forms prompt 1b/5): a region's text is its lines joined with
    # "\n", for EVERY region type. Region text and line text are never
    # derived from fields — fields only point at lines. A region with no
    # lines keeps the text the annotator typed (tables / lists without
    # lines still fall back to their cells / items).
    # ------------------------------------------------------------------
    for el in doc.layout:
        # bbox from quad — always.
        el.bbox = bbox_from_quad(el.quad)

        if isinstance(el, TableLayout):
            table = doc.tables[el.table_index]
            table.html = serialize_table_html(table.cells)

        if isinstance(el, LogoLayout) or not hasattr(el, "text"):
            continue

        own = region_text_from_lines(lines_by_block.get(el.id, []))
        if own is not None:
            el.text = own
        elif isinstance(el, TableLayout):
            el.text = doc.tables[el.table_index].html
        elif isinstance(el, HeadingLayout) and not el.text.lstrip().startswith("#"):
            el.text = "#" * el.level + " " + el.text
        elif isinstance(el, FigureLayout) and not el.text:
            el.text = "<figure/>"
        # else: no lines — keep el.text as typed by the human (lists: below)

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
        if isinstance(el, ListLayout) and el.id not in lines_by_block:
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