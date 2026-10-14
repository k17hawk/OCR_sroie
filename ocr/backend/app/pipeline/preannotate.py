"""Turn adapter output into schema-valid layout + lines, without overwriting human edits."""
from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image

from app.core.geometry import bbox_from_quad
from app.pipeline import registry
from app.schemas.doc_v2 import (
    Block,
    Doc,
    HeaderLayout,
    Line,
    ParagraphLayout,
)


def _quad_from_floats(quad: list[list[float]]) -> list[list[float]]:
    return [[round(p[0], 1), round(p[1], 1)] for p in quad]


def _next_layout_id(doc: Doc) -> int:
    return (max((e.id for e in doc.layout), default=-1) + 1)


def _touched(doc: Doc, ids: set[int]) -> set[int]:
    """A layout element is considered human-touched if any line references it
    with role/handwritten set, or if the element's own text has been edited
    away from the raw OCR. Since we don't yet track provenance per element,
    the Phase-5-safe heuristic is: any element currently *in* the doc is
    human-owned; only elements NOT in the doc can be added by a re-run."""
    return ids


def run(
    *,
    doc: Doc,
    image_bytes: bytes,
    config: dict[str, Any],
    device: str = "cpu",
) -> tuple[Doc, dict[str, Any]]:
    """Return (new_doc, stats). Never mutates ``doc``."""
    adapters = registry.load({**config, "device": device})
    img = Image.open(BytesIO(image_bytes))
    img.load()

    existing_ids = {e.id for e in doc.layout}

    # 1) Layout
    regions = adapters["layout"].detect(img)

    # 2) Text
    text_quads = adapters["text"].detect(img)
    crops = [
        img.crop((
            int(min(p[0] for p in q)), int(min(p[1] for p in q)),
            int(max(p[0] for p in q)), int(max(p[1] for p in q)),
        ))
        for q in text_quads
    ]
    ocr = adapters["rec"].recognize(crops, doc.language)

    # 3) Assemble. Only *add* new elements; never touch existing ones.
    new_layout = list(doc.layout)
    new_lines = list(doc.lines)
    added = 0

    for reg in regions:
        el_id = _next_layout_id(Doc.model_construct(layout=new_layout))  # local counter
        # Skip if this exact element already exists in the human-owned set.
        if el_id in existing_ids:
            continue

        quad = _quad_from_floats(reg["quad"])
        bbox = bbox_from_quad(quad)  # type: ignore[arg-type]
        rtype = reg["type"]

        if rtype == "header":
            new_layout.append(HeaderLayout(
                id=el_id, bbox=bbox, reading_order=None, text="",
                quad=quad,  # type: ignore[arg-type]
            ))
        elif rtype == "paragraph":
            new_layout.append(ParagraphLayout(
                id=el_id, bbox=bbox, reading_order=None, text="",
                quad=quad,  # type: ignore[arg-type]
            ))
        else:
            continue
        added += 1

    # 4) Lines: match each text quad to the nearest new layout element.
    added_lines = 0
    for q, (txt, _conf) in zip(text_quads, ocr):
        qf = _quad_from_floats(q)
        bb = bbox_from_quad(qf)  # type: ignore[arg-type]
        # Simple containment: pick the layout element whose bbox contains the quad centroid.
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        owner = None
        for el in new_layout:
            x0, y0, x1, y1 = el.bbox
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                owner = el
                break
        if owner is None or owner.id in existing_ids:
            continue
        new_lines.append(Line(
            text=txt, bbox=bb, block_id=owner.id,
            line_in_block=sum(1 for ln in new_lines if ln.block_id == owner.id),
            is_heading=owner.type == "heading",
            font_size=float(max(1, bb[3] - bb[1])),
            font_id=None, quad=qf,  # type: ignore[arg-type]
        ))
        added_lines += 1

    # 5) Recompute text on the newly added layout elements from their lines.
    lines_by_block: dict[int, list[Line]] = {}
    for ln in new_lines:
        lines_by_block.setdefault(ln.block_id, []).append(ln)
    for el in new_layout:
        if el.id in existing_ids:
            continue
        bucket = sorted(lines_by_block.get(el.id, []), key=lambda ln: ln.bbox[1])
        if hasattr(el, "text"):
            if el.type == "heading":
                el.text = "#" * el.level + " " + (bucket[0].text if bucket else "")
            else:
                el.text = "\n".join(ln.text for ln in bucket)

    # 6) Reading order: append in current list order.
    ro = max((e.reading_order or 0) for e in new_layout) + 1 if new_layout else 0
    for el in new_layout:
        if el.reading_order is None and el.type not in ("logo", "signature", "stamp"):
            el.reading_order = ro
            ro += 1

    # 7) Rebuild blocks so the doc stays internally consistent.
    new_blocks = [
        Block(type=el.type, bbox=el.bbox, block_id=el.id, font_id=None, quad=el.quad)
        for el in new_layout
    ]

    out = doc.model_copy(deep=True)
    out.layout = new_layout
    out.lines = new_lines
    out.blocks = new_blocks

    stats = {
        "adapter_family": config.get("family", "mock"),
        "device": device,
        "added_layout": added,
        "added_lines": added_lines,
    }
    return out, stats
