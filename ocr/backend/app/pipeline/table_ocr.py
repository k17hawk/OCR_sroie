"""Table structure recognition helpers.

Runs the ``table`` adapter on a sub-region and produces a fully populated
``Table`` object with cell text drawn from the document's existing lines.
"""
from __future__ import annotations

from typing import Any

from PIL.Image import Image

from app.core.geometry import bbox_from_quad
from app.schemas.doc_v2 import Doc, Table, TableCell


# ---------------------------------------------------------------------------
# Bbox overlap helpers
# ---------------------------------------------------------------------------

def _area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _overlap(a: list[int], b: list[int]) -> int:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def _quad_from_bbox(b: list[int]) -> list[list[float]]:
    return [
        [float(b[0]), float(b[1])],
        [float(b[2]), float(b[1])],
        [float(b[2]), float(b[3])],
        [float(b[0]), float(b[3])],
    ]


# ---------------------------------------------------------------------------
# Adapter output normalization
# ---------------------------------------------------------------------------

def _normalize_cells(raw: dict[str, Any], quad: list[list[float]]) -> tuple[
    int, int, int, list[dict[str, Any]]
]:
    """Return (rows, cols, header_rows, cells) regardless of adapter shape."""

    # Case 1: already a grid — ``cells`` have explicit row/col.
    cells_raw = raw.get("cells") or []
    if cells_raw and "row" in cells_raw[0] and "col" in cells_raw[0]:
        rows = int(raw.get("rows") or max(c["row"] for c in cells_raw) + 1)
        cols = int(raw.get("cols") or max(c["col"] for c in cells_raw) + 1)
        header_rows = int(raw.get("header_rows", 1))
        return rows, cols, header_rows, cells_raw

    # Case 2: TATR-style — rows and columns are separate detections.
    rows_det = [c for c in cells_raw if c.get("kind") == "table row"]
    cols_det = [c for c in cells_raw if c.get("kind") == "table column"]
    if rows_det and cols_det:
        rows = len(rows_det)
        cols = len(cols_det)
        header_rows = 1
        cells: list[dict[str, Any]] = []
        for r, rb in enumerate(sorted(rows_det, key=lambda c: c["bbox"][1])):
            for c, cb in enumerate(sorted(cols_det, key=lambda c: c["bbox"][0])):
                x0 = max(rb["bbox"][0], cb["bbox"][0])
                y0 = max(rb["bbox"][1], cb["bbox"][1])
                x1 = min(rb["bbox"][2], cb["bbox"][2])
                y1 = min(rb["bbox"][3], cb["bbox"][3])
                if x1 > x0 and y1 > y0:
                    cells.append({
                        "row": r, "col": c,
                        "rowspan": 1, "colspan": 1,
                        "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    })
        if cells:
            return rows, cols, header_rows, cells

    # Case 3: fallback — uniform grid.
    rows, cols, header_rows = 2, 2, 1
    x0 = int(min(p[0] for p in quad))
    y0 = int(min(p[1] for p in quad))
    x1 = int(max(p[0] for p in quad))
    y1 = int(max(p[1] for p in quad))
    cells = []
    for r in range(rows):
        for c in range(cols):
            cx0 = x0 + (x1 - x0) * c // cols
            cx1 = x0 + (x1 - x0) * (c + 1) // cols
            cy0 = y0 + (y1 - y0) * r // rows
            cy1 = y0 + (y1 - y0) * (r + 1) // rows
            cells.append({
                "row": r, "col": c, "rowspan": 1, "colspan": 1,
                "bbox": [cx0, cy0, cx1, cy1],
            })
    return rows, cols, header_rows, cells


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recognize_table(
    *,
    doc: Doc,
    img: Image,
    quad: list[list[float]],
    adapter: Any,
) -> Table:
    """Run table structure recognition and populate cell text from existing lines."""
    raw = adapter.recognize(img, quad)
    rows, cols, header_rows, cells_raw = _normalize_cells(raw, quad)

    cells: list[TableCell] = []
    for i, cr in enumerate(cells_raw):
        bbox = [int(v) for v in cr["bbox"]]
        cell_area = max(1, _area(bbox))

        # Lines whose bboxes are mostly contained in this cell.
        contained = [
            ln for ln in doc.lines
            if _overlap(ln.bbox, bbox) >= 0.4 * _area(ln.bbox)
        ]
        contained.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        text = " ".join(ln.text for ln in contained).strip()

        is_header = bool(cr.get("is_header", cr["row"] < header_rows))
        handwritten = any(getattr(ln, "handwritten", False) for ln in contained)

        cells.append(TableCell(
            id=i,
            row=int(cr["row"]),
            col=int(cr["col"]),
            rowspan=int(cr.get("rowspan", 1)),
            colspan=int(cr.get("colspan", 1)),
            text=text,
            is_header=is_header,
            handwritten=handwritten,
            empty=(text == ""),
            bbox=bbox,
            quad=_quad_from_bbox(bbox),
        ))

    return Table(
        bbox=bbox_from_quad(quad),
        quad=quad,
        num_rows=rows,
        num_cols=cols,
        header_rows=header_rows,
        ruling_style="full",
        theme=None,
        col_types=None,
        html="",
        reading_order=[c.id for c in cells],
        cells=cells,
    )