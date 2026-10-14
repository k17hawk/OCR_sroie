"""Create a valid, finalize-idempotent empty document."""
from __future__ import annotations

from app import __version__
from app.schemas.doc_v2 import (
    Annotation,
    Doc,
    Quality,
    RawLengths,
)
from app.services.numbers import format_document_id, format_image_path


def make_skeleton(
    *,
    n: int,
    image_ext: str,
    width: int,
    height: int,
    page_number: int,
    page_count: int,
    language: str,
    quality: dict,
) -> Doc:
    return Doc(
        v=2,
        image=format_image_path(n, image_ext),
        width=width,
        height=height,
        document_id=format_document_id(n),
        page_number=page_number,
        page_count=page_count,
        language=language,          # type: ignore[arg-type]
        digit_style="mixed",
        doc_type="other",           # annotator must change before submit
        text="",
        fonts=[],
        font_ids=[],
        layout=[],
        blocks=[],
        lines=[],
        tables=[],
        ruling_lines=[],
        extra_regions=[],
        quality=Quality(
            **quality,
            line_count=0,
            table_count=0,
            figure_count=0,
            stamp_bboxes=[],
            signature_bboxes=[],
            raw_lengths=RawLengths(
                paragraph_chars=0, list_items=0, form_fields=0, table_cells=0
            ),
        ),
        annotation=Annotation(
            source="real", tool=f"devalabel@{__version__}", status="in_progress"
        ),
    )
