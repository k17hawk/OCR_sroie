"""DevaLabel page schema 2.1 — the single source of truth.

Everything else is generated from this module:
    schema/doc_v21.schema.json            (python -m scripts.gen_schema)
    ocr/frontend/src/schema.gen.ts         (same command, via json-schema-to-typescript)
    ANNOTATION_GUIDE.md                    (later step)

What changed from v2 (see the DEVALABEL UPDATE, part A):
    A1  region type = appearance, fields = meaning: form_field, table and list
        regions all carry ``fields[]``; ``text_line`` / ``list_item`` are no
        longer region types (they are mapped on load by the migration).
    A2  a field is {label?, field_type, value?, options[]}; label is a reference
        ({line_id} | {cell_id} | {region_id}) or null; an option is
        {line_id, checked, box | tick_cell_id}. Lines carry a stable ``id`` and a
        ``role`` but NO field_index / checked / box / tick_cell_id.
    A3  a blank fill-in is a FieldValue {text:"", bbox, handwritten:false, empty:true}.
    A8  digit_style is "dev" | "latin" | "mixed" | "none".
        Checkbox extra_regions are gone: checkboxes live in field.options.

This module checks SHAPE only (types, required keys, exactly-one-of). Rules
about content (no leader dots, grid tiling, references that resolve, …)
live in the shared validate package so there is one copy of them.
"""
from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from app.services.geometry import fix_quad

SCHEMA_VERSION = "2.1"

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

Point = Annotated[list[float], Field(min_length=2, max_length=2)]
BBox = Annotated[list[int], Field(min_length=4, max_length=4,
                                  description="[x0, y0, x1, y1] integer pixels")]
Quad = Annotated[list[Point], Field(min_length=4, max_length=4,
                                    description="TL, TR, BR, BL")]


class _Schema(BaseModel):
    """Forbids unknown keys, keeps declaration order, drops ``None`` for the
    keys listed in ``_OMIT_IF_NONE`` (all other ``None`` stay JSON null)."""

    model_config = ConfigDict(extra="forbid")
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _fill_quad(cls, data: Any) -> Any:
        if isinstance(data, dict) and "quad" in cls.model_fields and "bbox" in cls.model_fields:
            data = dict(data)
            fix_quad(data)
        return data

    @model_serializer(mode="wrap")
    def _omit_none(self, handler: Any) -> Any:
        data = handler(self)
        for key in self._OMIT_IF_NONE:
            if key in data and data[key] is None:
                data.pop(key)
        return data


def _exactly_one(obj: Any, keys: tuple[str, ...], what: str) -> None:
    set_keys = [k for k in keys if getattr(obj, k) is not None]
    if len(set_keys) != 1:
        raise ValueError(f"{what} needs exactly one of {' | '.join(keys)}, got "
                         f"{set_keys or 'none'}")


# ---------------------------------------------------------------------------
# Fields (A2 / A3)
# ---------------------------------------------------------------------------

class LabelRef(_Schema):
    """Where a field's question is printed: one line, one table cell (of the
    region's own table) or a whole region."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"line_id", "cell_id", "region_id"})

    line_id: Optional[int] = None
    cell_id: Optional[int] = None
    region_id: Optional[int] = None

    @model_validator(mode="after")
    def _one(self) -> "LabelRef":
        _exactly_one(self, ("line_id", "cell_id", "region_id"), "label")
        return self


class FieldOption(_Schema):
    """One printed choice. ``line_id`` is the option's text line; the mark is
    either a drawn square (``box``) or a table cell to tick (``tick_cell_id``).
    ``checked`` is null until known (tick cells: derived on save)."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"box", "tick_cell_id"})

    line_id: int
    checked: Optional[bool] = None
    box: Optional[BBox] = None
    tick_cell_id: Optional[int] = None

    @model_validator(mode="after")
    def _one(self) -> "FieldOption":
        _exactly_one(self, ("box", "tick_cell_id"), "option")
        return self


class FieldValue(_Schema):
    """The answer area.
    blank:       {text: "", bbox, handwritten: false, empty: true}
    pre-printed: {line_id, text, bbox, handwritten: false, empty: false} — the
                 value line has role "value" (forms prompt 1b)."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"line_id"})

    line_id: Optional[int] = None
    text: str
    bbox: BBox
    handwritten: bool
    empty: bool


class FormField(_Schema):
    label: Optional[LabelRef] = None
    field_type: str
    value: Optional[FieldValue] = None
    options: list[FieldOption] = []


# ---------------------------------------------------------------------------
# Layout regions
# ---------------------------------------------------------------------------

class _Region(_Schema):
    """Key order of every region: id, type, bbox, reading_order, <own keys>,
    parent_id, parent_item, quad (same as v2)."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset(
        {"parent_id", "parent_item", "subtype", "handwritten"})


# Plain text regions
class TitleLayout(_Region):
    id: int
    type: Literal["title"] = "title"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class SubheadingLayout(_Region):
    id: int
    type: Literal["subheading"] = "subheading"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class HeaderLayout(_Region):
    id: int
    type: Literal["header"] = "header"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class ParagraphLayout(_Region):
    id: int
    type: Literal["paragraph"] = "paragraph"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class FooterLayout(_Region):
    id: int
    type: Literal["footer"] = "footer"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class PageNumberLayout(_Region):
    id: int
    type: Literal["page_number"] = "page_number"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class SignatureCaptionLayout(_Region):
    id: int
    type: Literal["signature_caption"] = "signature_caption"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class TableHeadingLayout(_Region):
    id: int
    type: Literal["table_heading"] = "table_heading"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class TableFooterLayout(_Region):
    id: int
    type: Literal["table_footer"] = "table_footer"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class OptionsLayout(_Region):
    id: int
    type: Literal["options"] = "options"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class FormLabelLayout(_Region):
    id: int
    type: Literal["form_label"] = "form_label"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class FormValueLayout(_Region):
    id: int
    type: Literal["form_value"] = "form_value"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class LabelLayout(_Region):
    id: int
    type: Literal["label"] = "label"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class HandwrittenAnnotationLayout(_Region):
    id: int
    type: Literal["handwritten_annotation"] = "handwritten_annotation"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class HandwrittenCorrectionLayout(_Region):
    id: int
    type: Literal["handwritten_correction"] = "handwritten_correction"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class HandwrittenNumberLayout(_Region):
    id: int
    type: Literal["handwritten_number"] = "handwritten_number"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class MixedTextLayout(_Region):
    id: int
    type: Literal["mixed_text"] = "mixed_text"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class HeadingLayout(_Region):
    id: int
    type: Literal["heading"] = "heading"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    level: int
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class HandwritingLayout(_Region):
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


class FigureLayout(_Region):
    id: int
    type: Literal["figure"] = "figure"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    subtype: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class FormFieldLayout(_Region):
    id: int
    type: Literal["form_field"] = "form_field"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    fields: list[FormField] = []
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class TableLayout(_Region):
    id: int
    type: Literal["table"] = "table"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    table_index: int
    fields: list[FormField] = []
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class ListItem(_Schema):
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset({"children"})

    marker: str
    level: int
    text: str
    children: Optional[list[int]] = None


class ListLayout(_Region):
    id: int
    type: Literal["list"] = "list"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    list_style: str
    items: list[ListItem]
    fields: list[FormField] = []
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class LogoLayout(_Region):
    id: int
    type: Literal["logo"] = "logo"
    bbox: BBox
    reading_order: Optional[int] = None
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class SignatureLayout(_Region):
    id: int
    type: Literal["signature"] = "signature"
    bbox: BBox
    reading_order: Optional[int] = None
    signer: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class StampLayout(_Region):
    id: int
    type: Literal["stamp"] = "stamp"
    bbox: BBox
    reading_order: Optional[int] = None
    stamp_text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


LayoutElement = Annotated[
    Union[
        HeaderLayout, TitleLayout, HeadingLayout, SubheadingLayout, ParagraphLayout,
        FooterLayout, PageNumberLayout, SignatureCaptionLayout,
        FigureLayout, TableLayout, TableHeadingLayout, TableFooterLayout, ListLayout,
        FormFieldLayout, FormLabelLayout, FormValueLayout, LabelLayout, OptionsLayout,
        LogoLayout, SignatureLayout, StampLayout,
        HandwritingLayout, HandwrittenAnnotationLayout, HandwrittenCorrectionLayout,
        HandwrittenNumberLayout, MixedTextLayout,
    ],
    Field(discriminator="type"),
]

REGION_TYPES: tuple[str, ...] = tuple(
    m.model_fields["type"].default for m in LayoutElement.__origin__.__args__  # type: ignore[attr-defined]
)
# Region types that may carry fields[] (A1)
FIELD_REGION_TYPES = ("form_field", "table", "list")
# Removed in 2.1; the migration maps them (A1)
REMOVED_REGION_TYPES = ("text_line", "list_item")


# ---------------------------------------------------------------------------
# Blocks / lines
# ---------------------------------------------------------------------------

class Block(_Schema):
    type: str
    bbox: BBox
    block_id: int
    font_id: Optional[int] = None
    quad: Quad


class Line(_Schema):
    """One printed line. ``id`` is stable across saves and is what fields and
    options point at. ``block_id`` is the region id; ``table_index`` +
    ``cell_id`` / ``item_index`` place the line inside a table cell / list item."""
    _OMIT_IF_NONE: ClassVar[frozenset[str]] = frozenset(
        {"role", "handwritten", "table_index", "cell_id", "list_level", "item_index"})

    id: int
    text: str
    bbox: BBox
    block_id: int
    line_in_block: int
    is_heading: bool
    font_size: float
    font_id: Optional[int] = None
    role: Optional[Literal["label", "value", "option"]] = None
    table_index: Optional[int] = None
    cell_id: Optional[int] = None
    handwritten: Optional[bool] = None
    list_level: Optional[int] = None
    item_index: Optional[int] = None
    quad: Quad


# ---------------------------------------------------------------------------
# Tables (A4)
# ---------------------------------------------------------------------------

class TableCell(_Schema):
    """``text`` uses "\\n" between printed lines; ``html`` renders it as <br>."""
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
    """``header_rows`` = number of leading rows whose cells are all <th>."""
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


class FingerprintExtraRegion(_Schema):
    type: Literal["fingerprint"] = "fingerprint"
    hand: Literal["left", "right"]
    bbox: BBox
    quad: Quad


ExtraRegion = Annotated[Union[FingerprintExtraRegion], Field(discriminator="type")]


# ---------------------------------------------------------------------------
# Page
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


class Annotation(_Schema):
    source: Literal["synthetic", "real"]
    tool: str
    status: Literal["generated", "pre_annotated", "in_progress", "submitted",
                    "approved", "rejected"]


DigitStyle = Literal["dev", "latin", "mixed", "none"]
Language = Literal["ne", "en", "mixed", "hi"]


class Doc(_Schema):
    v: Literal["2.1"] = "2.1"
    image: str
    width: int
    height: int
    document_id: str
    page_number: int
    page_count: int
    language: Language
    digit_style: DigitStyle
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


Doc.model_rebuild()
