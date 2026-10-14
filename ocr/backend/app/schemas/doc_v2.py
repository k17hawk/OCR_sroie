"""Pydantic v2 models for the DevaLabel document schema (v2).

Single source of truth for both synthetic and real document annotations.
Key order, null conventions, and optional-key omission exactly match
``schema/sample.json``.

See ``docs/DECISIONS.md`` for the rules applied when the prompt text and
the sample disagreed.
"""
from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer


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

class FormFieldLabel(_Schema):
    text: str
    bbox: BBox
    quad: Quad


class CheckboxOption(_Schema):
    text: str
    checked: bool
    box_quad: Quad
    text_quad: Quad


class NormalFieldValue(_Schema):
    text: str
    bbox: BBox
    quad: Quad
    handwritten: bool


class CheckboxFieldValue(_Schema):
    text: str
    options: list[CheckboxOption]


FieldValue = Union[NormalFieldValue, CheckboxFieldValue]


class FormFieldEntry(_Schema):
    label: FormFieldLabel
    field_type: str
    value: Optional[FieldValue] = None


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
        {"parent_id", "parent_item", "subtype", "handwritten"}
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


class TextLineLayout(_TextLayout):
    type: Literal["text_line"] = "text_line"


class ListItemLayout(_TextLayout):
    type: Literal["list_item"] = "list_item"


class OptionsLayout(_TextLayout):
    type: Literal["options"] = "options"


class FormLabelLayout(_TextLayout):
    type: Literal["form_label"] = "form_label"


class FormValueLayout(_TextLayout):
    type: Literal["form_value"] = "form_value"


class LabelLayout(_TextLayout):
    type: Literal["label"] = "label"


class PageNumberLayout(_TextLayout):
    type: Literal["page_number"] = "page_number"


class TableHeadingLayout(_TextLayout):
    type: Literal["table_heading"] = "table_heading"


class TableFooterLayout(_TextLayout):
    type: Literal["table_footer"] = "table_footer"


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


class HandwrittenAnnotationLayout(_TextLayout):
    type: Literal["handwritten_annotation"] = "handwritten_annotation"


class HandwrittenCorrectionLayout(_TextLayout):
    type: Literal["handwritten_correction"] = "handwritten_correction"


class HandwrittenNumberLayout(_TextLayout):
    type: Literal["handwritten_number"] = "handwritten_number"


class MixedTextLayout(_TextLayout):
    type: Literal["mixed_text"] = "mixed_text"


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
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class ParagraphLayout(_LayoutBase):
    id: int
    type: Literal["paragraph"] = "paragraph"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class SignatureCaptionLayout(_LayoutBase):
    id: int
    type: Literal["signature_caption"] = "signature_caption"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


class FooterLayout(_LayoutBase):
    id: int
    type: Literal["footer"] = "footer"
    bbox: BBox
    reading_order: Optional[int] = None
    text: str
    parent_id: Optional[int] = None
    parent_item: Optional[int] = None
    quad: Quad


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
        TextLineLayout,
        FooterLayout,
        PageNumberLayout,
        SignatureCaptionLayout,
        # blocks
        FigureLayout,
        TableLayout,
        TableHeadingLayout,
        TableFooterLayout,
        ListLayout,
        ListItemLayout,
        # forms
        FormFieldLayout,
        FormLabelLayout,
        FormValueLayout,
        LabelLayout,
        OptionsLayout,
        # marks
        LogoLayout,
        SignatureLayout,
        StampLayout,
        # handwriting
        HandwritingLayout,
        HandwrittenAnnotationLayout,
        HandwrittenCorrectionLayout,
        HandwrittenNumberLayout,
        # misc
        MixedTextLayout,
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
        {"role", "handwritten", "table_index", "cell_id", "list_level", "item_index"}
    )

    text: str
    bbox: BBox
    block_id: int
    line_in_block: int
    is_heading: bool
    font_size: float
    font_id: Optional[int] = None
    role: Optional[Literal["label", "value", "option"]] = None
    handwritten: Optional[bool] = None
    table_index: Optional[int] = None
    cell_id: Optional[int] = None
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
    v: int = 2
    image: str
    width: int
    height: int
    document_id: str
    page_number: int
    page_count: int
    language: Literal["ne", "en", "mixed", "hi"]
    digit_style: Literal["devanagari", "dev", "latin", "mixed"]
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


# ---------------------------------------------------------------------------
# Rebuild — needed because forward refs inside discriminated unions
# are resolved lazily by Pydantic v2.
# ---------------------------------------------------------------------------

Doc.model_rebuild()


# ---------------------------------------------------------------------------
# Re-exports for convenience (unchanged from prior file)
# ---------------------------------------------------------------------------

__all__ = [
    "Point", "BBox", "Quad",
    "FormFieldLabel", "CheckboxOption", "NormalFieldValue",
    "CheckboxFieldValue", "FieldValue", "FormFieldEntry", "ListItem",
    "TitleLayout", "SubheadingLayout", "TextLineLayout", "ListItemLayout",
    "OptionsLayout", "FormLabelLayout", "FormValueLayout", "LabelLayout",
    "PageNumberLayout", "TableHeadingLayout", "TableFooterLayout",
    "HandwritingLayout", "HandwrittenAnnotationLayout",
    "HandwrittenCorrectionLayout", "HandwrittenNumberLayout",
    "MixedTextLayout",
    "HeaderLayout", "HeadingLayout", "FigureLayout", "FormFieldLayout",
    "TableLayout", "ListLayout", "ParagraphLayout", "SignatureCaptionLayout",
    "FooterLayout", "LogoLayout", "SignatureLayout", "StampLayout",
    "LayoutElement", "Block", "Line", "TableCell", "Table", "RulingLine",
    "CheckboxExtraRegion", "FingerprintExtraRegion", "ExtraRegion",
    "RawLengths", "Quality", "Annotation", "Doc",
]