"""Table structure recognition.

Builds a fully populated ``Table`` for a region of the page: grid from the
image / table model / text lines (see ``table_structure``), cell text from
the document's existing lines. The old version fell back to a 2x2 grid
whenever the model gave nothing usable, which is why cells went missing.
"""
from __future__ import annotations

from typing import Any

from PIL.Image import Image

from app.core.geometry import bbox_from_quad
from app.schemas.doc_v2 import Doc, Table, TableCell


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _shift_raw(raw: dict[str, Any], frame: Any) -> dict[str, Any]:
    """Adapter boxes are in crop coords already (we pass it the crop)."""
    out = dict(raw or {})
    # adapters that only return a grid of cells: turn it into row/col boxes
    cells = out.get("cells") or []
    if cells and not out.get("row_boxes") and "row" in cells[0] and "bbox" in cells[0]:
        rows: dict[int, list[float]] = {}
        cols: dict[int, list[float]] = {}
        for c in cells:
            b = c["bbox"]
            r = rows.setdefault(int(c["row"]), [b[0], b[1], b[2], b[3]])
            r[1] = min(r[1], b[1]); r[3] = max(r[3], b[3])
            k = cols.setdefault(int(c["col"]), [b[0], b[1], b[2], b[3]])
            k[0] = min(k[0], b[0]); k[2] = max(k[2], b[2])
        if len(rows) >= 2:
            out["row_boxes"] = [rows[k] for k in sorted(rows)]
        if len(cols) >= 2:
            out["col_boxes"] = [cols[k] for k in sorted(cols)]
    return out


def recognize_table(
    *,
    doc: Doc,
    img: Image,
    quad: list[list[float]],
    adapter: Any,
    rec: Any = None,
    lang_hint: str = "ne",
) -> Table:
    """Find every row / column / cell of the table inside ``quad``.

    The grid comes from the image (ruling lines) first, then the table model,
    then the page's text lines - per axis, so a table with only horizontal
    rules still gets its columns from the text. Cell text comes from the
    page's existing lines; cells with ink but no line are read with ``rec``
    (when it is a real recognizer).
    """
    from app.pipeline import table_structure as ts

    frame = ts.make_frame(img, quad)

    raw: dict[str, Any] = {}
    try:
        rx0, ry0, rx1, ry1 = frame.region
        sub = frame.img.crop((rx0, ry0, rx1, ry1))
        sub_quad = [[0.0, 0.0], [float(rx1 - rx0), 0.0],
                    [float(rx1 - rx0), float(ry1 - ry0)], [0.0, float(ry1 - ry0)]]
        got = adapter.recognize(sub, sub_quad) or {}
        # back to crop coords
        def mv(b): return [b[0] + rx0, b[1] + ry0, b[2] + rx0, b[3] + ry0]
        for key in ("cells", "row_boxes", "col_boxes"):
            if got.get(key):
                got[key] = [({**c, "bbox": mv(c["bbox"])} if isinstance(c, dict) else mv(c))
                            for c in got[key]]
        raw = _shift_raw(got, frame)
    except RuntimeError:
        raise
    except Exception:  # noqa: BLE001 - a broken model must not stop the image path
        raw = {}

    # page lines -> deskewed crop coords
    tlines = []
    for i, ln in enumerate(doc.lines):
        b = frame.box_to_crop(ln.bbox)
        tlines.append(ts.TLine(text=ln.text, box=b, handwritten=bool(getattr(ln, "handwritten", False)), idx=i))

    st = ts.detect(frame, tlines, raw)

    # cells with ink but no text: read them, if there is a real recognizer
    if rec is not None and not str(getattr(rec, "name", "")).startswith("mock"):
        ink = ts._ink(frame.img)
        todo = []
        for c in st.cells:
            if c.text:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in c.box]
            inner = ink[y0 + 3:y1 - 2, x0 + 3:x1 - 2]
            if inner.size and inner.mean() > 0.01:
                todo.append(c)
        if todo:
            crops = [frame.img.crop(tuple(int(round(v)) for v in c.box)) for c in todo]
            try:
                res = rec.recognize(crops, lang_hint=lang_hint)
                for c, (t, _conf) in zip(todo, res):
                    c.text = (t or "").strip()
            except Exception:  # noqa: BLE001
                pass

    cells: list[TableCell] = []
    for i, c in enumerate(sorted(st.cells, key=lambda c: (c.row, c.col))):
        q = frame.quad_to_page(c.box)
        cells.append(TableCell(
            id=i, row=c.row, col=c.col, rowspan=c.rowspan, colspan=c.colspan,
            text=c.text, is_header=c.is_header, handwritten=c.handwritten,
            empty=(c.text.strip() == ""), bbox=bbox_from_quad(q), quad=q,
        ))

    tq = frame.quad_to_page((st.xs[0], st.ys[0], st.xs[-1], st.ys[-1]))
    return Table(
        bbox=bbox_from_quad(tq),
        quad=tq,
        num_rows=len(st.ys) - 1,
        num_cols=len(st.xs) - 1,
        header_rows=st.header_rows,
        ruling_style=st.ruling_style,
        theme=None,
        col_types=None,
        html=_html(cells),
        reading_order=[c.id for c in cells],
        cells=cells,
    )


def _html(cells: list[TableCell]) -> str:
    import html as _h
    rows: dict[int, list[TableCell]] = {}
    for c in cells:
        rows.setdefault(c.row, []).append(c)
    out = ["<table>"]
    for r in sorted(rows):
        out.append("<tr>")
        for c in sorted(rows[r], key=lambda c: c.col):
            tag = "th" if c.is_header else "td"
            attrs = (f' colspan="{c.colspan}"' if c.colspan > 1 else "") + (f' rowspan="{c.rowspan}"' if c.rowspan > 1 else "")
            out.append(f"<{tag}{attrs}>{_h.escape(c.text, quote=False)}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)
