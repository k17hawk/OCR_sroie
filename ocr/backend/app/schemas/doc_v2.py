"""Pydantic v2 models for the DevaLabel document schema (v2).

Single source of truth for both synthetic and real document annotations.
Key order, null conventions, and optional-key omission exactly match
``schema/sample.json``.

See ``docs/DECISIONS.md`` for the rules applied when the prompt text and
the sample disagreed.
"""
from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from app.services.geometry import fix_quad


# ---------------------------------------------------------------------------
# Geometry aliases
# ---------------------------------------------------------------------------

Point = Annotated[list[float], Field(min_length=2, max_length=2)]
"""[x, y] as floats."""

BBox = Annotated[list[int], Field(min_length=4, max_length=4)]
"""[x0, y0, x1, y1] in integer pixels."""

Quad = Annotated[list[Point], Field(min_length=4, max_length=4)]
"""Four [x, y] floats, ordered top-left, top-right, bottom-right, bottom-left."""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _Schema(BaseModel):
    """Base model: forbids extras, preserves field order, omits configured
    ``None`` values from the serialized output.

    Subclasses set ``_OMIT_IF_NONE`` to a set of field names whose ``None``
    value should be dropped. All other ``None`` values are kept as JSON ``null``.
    """

    model_config = ConfigDict(extra="forbid")

    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _fill_quad(cls, data: Any) -> Any:
        """``quad`` is optional on input: when it is missing or no longer fits
        ``bbox`` it is rebuilt from the bbox (see services/geometry.py). The
        serialized output always carries a quad, so the v2 JSON shape is
        unchanged."""
        if (
            isinstance(data, dict)
            and "quad" in cls.model_fields and "bbox" in cls.model_fields
        ):
            data = dict(data)
            fix_quad(data)
        return data

    @model_serializer(mode="wrap")
    def _serialize_omit_none(self, handler: Any) -> Any:
        data = handler(self)
        omit = self._OMIT_IF_NONE
        if omit:
            for key in omit:
                if key in data and data[key] is None:
                    data.pop(key, None)
        return data


# ---------------------------------------------------------------------------
# Form-field internals
# ---------------------------------------------------------------------------

FIELD_TYPES: tuple[str, ...] = (
    "text", "name", "number", "date_bs", "date_ad", "checkbox", "photo", "signature_ref",
)


class FormFieldLabel(_Schema):
    """The printed question of a field.

    Where it is, at most one of:
      * ``line_id``   — a label line of this region (text/bbox copied from it)
      * ``cell_id``   — a cell of this region's table (may also have a bbox
                        inside that cell, and a line_id)
      * ``region_id`` — another region (e.g. a table_heading); no bbox
    or only ``text`` + ``bbox`` (a label drawn without a line).
    """
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset(
        {"bbox", "quad", "cell_id", "region_id", "line_id"}
    )

    text: str = ""
    bbox: Optional[BBox] = None
    quad: Optional[Quad] = None
    cell_id: Optional[int] = None
    region_id: Optional[int] = None
    line_id: Optional[int] = None


class FieldOption(_Schema):
    """One choice of a checkbox field. Its word is the line ``line_id``
    (role "option"); the option itself never copies the text.

    Tick area: a printed square (``box``) or a separate tick cell
    (``tick_cell_id``), or neither when the form shows none."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"box", "box_quad", "tick_cell_id"})

    line_id: int
    checked: Optional[bool] = None
    box: Optional[BBox] = None
    box_quad: Optional[Quad] = None
    tick_cell_id: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _fill_box_quad(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if data.get("box") is None:
                data.pop("box_quad", None)
            else:
                tmp = {"bbox": data["box"], "quad": data.get("box_quad")}
                fix_quad(tmp)
                data["box_quad"] = tmp.get("quad")
        return data


class NormalFieldValue(_Schema):
    """Where the answer goes (a leader / blank / value cell) and what is
    written there. ``empty`` is derived: ``text == ""``.

    A pre-printed value points at its line (role "value"):
    ``{line_id, text, bbox, handwritten: false, empty: false}`` (forms prompt 1b)."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"cell_id", "line_id"})

    line_id: Optional[int] = None
    text: str = ""
    bbox: BBox
    quad: Quad
    handwritten: bool = False
    empty: bool = True
    cell_id: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _derive_empty(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data["empty"] = (data.get("text") or "") == ""
        return data


FieldValue = NormalFieldValue


class FormFieldEntry(_Schema):
    """A question of a form_field, table or list region.

    Checkbox fields keep ``value`` null; the answer is ``options[].checked``.
    ``label`` may be null only for a checkbox field (a choice group with no
    printed question)."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"single_choice", "options"})

    label: Optional[FormFieldLabel] = None
    field_type: str = "text"
    single_choice: Optional[bool] = None
    value: Optional[FieldValue] = None
    options: Optional[list[FieldOption]] = None


class ListItem(_Schema):
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"children"})

    marker: str
    level: int
    text: str
    # ids of layout elements that belong to this item (a note, a figure, …);
    # each of them has parent_id = the list's id and parent_item = this index
    children: Optional[list[int]] = None


# ---------------------------------------------------------------------------
# Text-only layout base + 16 subclasses
#
# These must be defined BEFORE the LayoutElement union, because the union
# is a runtime expression and would raise NameError otherwise.
# ---------------------------------------------------------------------------

class _LayoutBase(_Schema):
    """Common base of every layout element: parent_id / parent_item are left
    out of the JSON when not set."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset(
        {"parent_id", "parent_item", "subtype", "handwritten", "fields"}
    )


class _TextLayout(_LayoutBase):
    """Base for text-only layout types. Not used directly in the union."""
    id: int
    type: str
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class TitleLayout(_TextLayout):
    type: Literal["title"] = "title"


class SubheadingLayout(_TextLayout):
    type: Literal["subheading"] = "subheading"


class TableHeadingLayout(_TextLayout):
    type: Literal["table_heading"] = "table_heading"


class HandwritingLayout(_LayoutBase):
    id: int
    type: Literal["handwriting"] = "handwriting"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    subtype: Optional[str] = None
    handwritten: Optional[bool] = None
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


# ---------------------------------------------------------------------------
# Structural / block / form / mark layout classes
# ---------------------------------------------------------------------------

class HeaderLayout(_LayoutBase):
    id: int
    type: Literal["header"] = "header"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class HeadingLayout(_LayoutBase):
    id: int
    type: Literal["heading"] = "heading"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    level: int
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class FigureLayout(_LayoutBase):
    id: int
    type: Literal["figure"] = "figure"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    subtype: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class FormFieldLayout(_LayoutBase):
    id: int
    type: Literal["form_field"] = "form_field"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    fields: list[FormFieldEntry]
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class TableLayout(_LayoutBase):
    id: int
    type: Literal["table"] = "table"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    table_index: int
    fields: Optional[list[FormFieldEntry]] = None
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class ListLayout(_LayoutBase):
    id: int
    type: Literal["list"] = "list"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    list_style: str
    items: list[ListItem]
    fields: Optional[list[FormFieldEntry]] = None
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class SublistLayout(ListLayout):
    """A list nested under another list's item: same properties as a list."""
    type: Literal["sublist"] = "sublist"


# region types that carry list items (list_style / items / text)
LIST_TYPES: frozenset[str] = frozenset({"list", "sublist"})


class ParagraphLayout(_LayoutBase):
    id: int
    type: Literal["paragraph"] = "paragraph"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class CaptionLayout(_TextLayout):
    type: Literal["caption"] = "caption"


class FooterLayout(_LayoutBase):
    id: int
    type: Literal["footer"] = "footer"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class PageNumberLayout(_TextLayout):
    """The printed page number (१, 2, "पृष्ठ ३", …)."""
    type: Literal["page_number"] = "page_number"


class LogoLayout(_LayoutBase):
    id: int
    type: Literal["logo"] = "logo"
    bbox: BBox
    reading_order: Optional[int] = None
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class SignatureLayout(_LayoutBase):
    id: int
    type: Literal["signature"] = "signature"
    bbox: BBox
    reading_order: Optional[int] = None
    signer: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class StampLayout(_LayoutBase):
    id: int
    type: Literal["stamp"] = "stamp"
    bbox: BBox
    reading_order: Optional[int] = None
    stamp_text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


# ---------------------------------------------------------------------------
# Discriminated union — now that all classes exist
# ---------------------------------------------------------------------------

LayoutElement = Annotated[
    Union[
        # structural
        HeaderLayout,
        TitleLayout,
        HeadingLayout,
        SubheadingLayout,
        ParagraphLayout,
        FooterLayout,
        PageNumberLayout,
        CaptionLayout,
        # blocks
        FigureLayout,
        TableLayout,
        TableHeadingLayout,
        ListLayout,
        SublistLayout,
        # forms
        FormFieldLayout,
        # marks
        LogoLayout,
        SignatureLayout,
        StampLayout,
        # handwriting
        HandwritingLayout,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class Block(_Schema):
    type: str
    bbox: BBox
    block_id: int
    font_id: Optional[int] = None
    quad: Quad


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

class Line(_Schema):
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset(
        {"id", "role", "handwritten", "table_index", "cell_id", "list_level", "item_index"}
    )

    # stable per page; never reused after a deletion (see Doc.next_line_id).
    # Fields reference lines by this id (label.line_id, options[].line_id).
    id: Optional[int] = None
    text: str
    bbox: BBox
    block_id: int
    line_in_block: int
    is_heading: bool
    font_size: float
    font_id: Optional[int] = None
    role: Optional[Literal["label", "value", "option"]] = None
    # key order as in the reference pages: a handwritten table line is
    # ..., "table_index", "cell_id", "handwritten", "quad"
    table_index: Optional[int] = None
    cell_id: Optional[int] = None
    handwritten: Optional[bool] = None
    list_level: Optional[int] = None
    item_index: Optional[int] = None
    quad: Quad


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class TableCell(_Schema):
    id: int
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str
    is_header: bool
    handwritten: bool
    empty: bool
    bbox: BBox
    quad: Quad


class Table(_Schema):
    bbox: BBox
    quad: Quad
    num_rows: int
    num_cols: int
    header_rows: int
    ruling_style: Literal["full", "horizontal", "none", "partial"]
    theme: Optional[str] = None
    col_types: Optional[str] = None
    html: str
    reading_order: list[int]
    cells: list[TableCell]


class RulingLine(_Schema):
    table_index: int
    orientation: Literal["h", "v"]
    width: int
    p0: Point
    p1: Point
    bbox: BBox


# ---------------------------------------------------------------------------
# Extra regions (discriminated on ``type``)
# ---------------------------------------------------------------------------

class CheckboxExtraRegion(_Schema):
    type: Literal["checkbox"] = "checkbox"
    checked: bool
    option: str
    bbox: BBox
    quad: Quad


class FingerprintExtraRegion(_Schema):
    type: Literal["fingerprint"] = "fingerprint"
    hand: Literal["left", "right"]
    bbox: BBox
    quad: Quad


ExtraRegion = Annotated[
    Union[CheckboxExtraRegion, FingerprintExtraRegion],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

class RawLengths(_Schema):
    paragraph_chars: int
    list_items: int
    form_fields: int
    table_cells: int


class Quality(_Schema):
    blur_sigma: Optional[float] = None
    downscale: Optional[float] = None
    noise_sigma: Optional[float] = None
    is_grayscale: bool
    is_binarized: bool
    skew_deg: Optional[float] = None
    perspective: bool
    ops: list[str] = []
    jpeg_quality: Optional[int] = None
    profile: str = "unknown"
    is_clean: bool
    has_stamp: bool
    has_signature: bool
    has_handwriting: bool
    has_fold: bool
    has_bleedthrough: bool
    line_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    stamp_bboxes: list[BBox] = []
    signature_bboxes: list[BBox] = []
    raw_lengths: RawLengths


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

class Annotation(_Schema):
    source: Literal["synthetic", "real"]
    tool: str
    status: Literal[
        "generated",
        "pre_annotated",
        "in_progress",
        "submitted",
        "approved",
        "rejected",
    ]


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

class Doc(_Schema):
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"next_line_id"})

    v: int = 2
    image: str
    width: int
    height: int
    document_id: str
    page_number: int
    page_count: int
    language: Literal["ne", "en", "mixed", "hi"]
    digit_style: Literal["devanagari", "dev", "latin", "mixed", "none"]
    doc_type: str
    text: str
    fonts: list[Any] = []
    font_ids: list[Any] = []
    layout: list[LayoutElement]
    blocks: list[Block]
    lines: list[Line]
    tables: list[Table]
    ruling_lines: list[RulingLine]
    extra_regions: list[ExtraRegion]
    quality: Quality
    annotation: Annotation
    # the id the next new line gets (> every id ever used on this page)
    next_line_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Rebuild — needed because forward refs inside discriminated unions
# are resolved lazily by Pydantic v2.
# ---------------------------------------------------------------------------

Doc.model_rebuild()

# The region types a page may use (section 1a), read from the union.
REGION_TYPES: frozenset[str] = frozenset(
    m.model_fields["type"].default
    for m in get_args(get_args(LayoutElement)[0])
)


# ---------------------------------------------------------------------------
# Re-exports for convenience (unchanged from prior file)
# ---------------------------------------------------------------------------

__all__ = [
    "Point", "BBox", "Quad",
    "FIELD_TYPES", "FormFieldLabel", "FieldOption", "NormalFieldValue",
    "FieldValue", "FormFieldEntry", "ListItem",
    "TitleLayout", "SubheadingLayout", "TableHeadingLayout", "CaptionLayout",
    "HandwritingLayout",
    "HeaderLayout", "HeadingLayout", "FigureLayout", "FormFieldLayout",
    "TableLayout", "ListLayout", "ParagraphLayout",
    "FooterLayout", "LogoLayout", "SignatureLayout", "StampLayout",
    "LayoutElement", "Block", "Line", "TableCell", "Table", "RulingLine",
    "CheckboxExtraRegion", "FingerprintExtraRegion", "ExtraRegion",
    "RawLengths", "Quality", "Annotation", "Doc", "REGION_TYPES", "LIST_TYPES", "SublistLayout",
]