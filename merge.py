"""
Stage 5 - Merge + reading order + output serialisation (rule-based, no training).

Lines are dicts: {"quad": (4,2) array, "bbox": [x1,y1,x2,y2], "text": str, "conf": float}
All coordinates must be in the same frame (page pixels).

Confidence per table cell = min(recognition, structure, assignment):
  recognition : weakest line confidence inside the cell (recognizer conf_min)
  structure   : table decoder probability for that cell token
  assignment  : fraction of the line's area inside the cell (low = line straddles cells)

Lines that straddle two or more cells (the detector merged neighbouring cells' text) are re-read per cell through
the `reread` callback, so each cell gets only its own text.

serialize_page() writes the same format as your ground-truth `text` field ("## " headings, one list item per
line, tables as HTML, blocks separated by a blank line), so you can score full pages end to end.
"""
import csv
import io
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from table_slanet import tokens_to_cells, tokens_to_html

# reread(cell_box_xyxy_in_page_pixels) -> (text, conf)
RereadFn = Callable[[List[float]], Tuple[str, float]]


def quad_to_bbox(quad) -> List[float]:
    q = np.asarray(quad, np.float32)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def _inter_frac(line_box, cell_box) -> float:
    ix1, iy1 = max(line_box[0], cell_box[0]), max(line_box[1], cell_box[1])
    ix2, iy2 = min(line_box[2], cell_box[2]), min(line_box[3], cell_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max((line_box[2] - line_box[0]) * (line_box[3] - line_box[1]), 1e-6)
    return inter / area


def _join_lines(lines: List[Dict], sep: str = " ") -> str:
    """Group lines into visual rows (by vertical overlap), rows top-to-bottom, lines in a row left-to-right."""
    if not lines:
        return ""
    rows: List[List[Dict]] = []
    for l in sorted(lines, key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2):
        cy = (l["bbox"][1] + l["bbox"][3]) / 2
        if rows:
            r = rows[-1]
            top = min(x["bbox"][1] for x in r)
            bot = max(x["bbox"][3] for x in r)
            if top <= cy <= bot:
                r.append(l)
                continue
        rows.append([l])
    ordered = [l for r in rows for l in sorted(r, key=lambda l: l["bbox"][0])]
    return sep.join(l["text"] for l in ordered if l["text"])


def merge_table(tokens: Sequence[str], cell_boxes: np.ndarray, cell_probs: Sequence[float],
                lines: List[Dict], min_overlap: float = 0.5, straddle_frac: float = 0.2,
                review_thresh: float = 0.8, reread: Optional[RereadFn] = None) -> Dict:
    """
    tokens/cell_boxes/cell_probs: table model output, cell_boxes already in page pixels (n,4 xyxy).
    Returns {"html", "cells", "grid", "used_line_ids", "straddling_line_ids"}.
    """
    layout = tokens_to_cells(tokens)
    n = min(len(layout), len(cell_boxes))
    per_cell: List[List[Tuple[Dict, float]]] = [[] for _ in range(n)]
    reread_cells, used, straddling = set(), set(), set()
    for li, line in enumerate(lines):
        fracs = np.array([_inter_frac(line["bbox"], cell_boxes[ci]) for ci in range(n)])
        if fracs.size == 0:
            continue
        hit = np.nonzero(fracs >= straddle_frac)[0]
        if len(hit) >= 2 and reread is not None:
            straddling.add(li)
            used.add(li)
            reread_cells.update(int(c) for c in hit)
            continue
        best = int(np.argmax(fracs))
        if fracs[best] >= min_overlap:
            per_cell[best].append((line, float(fracs[best])))
            used.add(li)
    cells, texts = [], []
    for ci in range(n):
        ls = [l for l, _ in per_cell[ci]]
        text = _join_lines(ls)
        rec = min([l["conf"] for l in ls], default=1.0)
        asg = min([f for _, f in per_cell[ci]], default=1.0)
        if ci in reread_cells:
            rt, rc = reread([float(v) for v in cell_boxes[ci]])
            text = (text + " " + rt).strip() if text else rt
            rec = min(rec, rc) if ls else rc
        struct = float(cell_probs[ci]) if ci < len(cell_probs) else 1.0
        conf = min(rec, asg, struct)
        cells.append({**layout[ci], "bbox": [float(v) for v in cell_boxes[ci]], "text": text,
                      "empty": not text, "confidence": conf, "reread": ci in reread_cells,
                      "conf_parts": {"recognition": rec, "assignment": asg, "structure": struct},
                      "needs_review": conf < review_thresh})
        texts.append(text)
    return {"html": tokens_to_html(tokens, texts), "cells": cells, "grid": cells_to_grid(cells),
            "used_line_ids": used, "straddling_line_ids": straddling}


def cells_to_grid(cells: List[Dict]) -> List[List[str]]:
    """2D list for CSV/Excel. Spanned cells put text at the origin; covered positions are ''."""
    if not cells:
        return []
    nr = max(c["row"] + c["rowspan"] for c in cells)
    nc = max(c["col"] + c["colspan"] for c in cells)
    grid = [[""] * nc for _ in range(nr)]
    for c in cells:
        grid[c["row"]][c["col"]] = c["text"]
    return grid


def grid_to_csv(grid: List[List[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf).writerows(grid)
    return buf.getvalue()


def order_blocks(blocks: List[Dict], page_w: float, full_width_frac: float = 0.6,
                 center_margin: float = 0.05) -> List[Dict]:
    """
    Reading order for 1-2 column pages. Separators (read in place, top to bottom) are full-width blocks and any
    block that clearly crosses the page centre (e.g. a narrow centred heading). Between separators the left
    column is read before the right column.
    """
    def cx(b): return (b["bbox"][0] + b["bbox"][2]) / 2
    def cy(b): return b["bbox"][1]
    def is_sep(b):
        x1, x2 = b["bbox"][0], b["bbox"][2]
        crosses = x1 < page_w * (0.5 - center_margin) and x2 > page_w * (0.5 + center_margin)
        return (x2 - x1) >= full_width_frac * page_w or crosses

    seps = sorted([b for b in blocks if is_sep(b)], key=cy)
    cols = [b for b in blocks if not is_sep(b)]
    out, last_y = [], -1e18
    bounds = [cy(s) for s in seps] + [1e18]
    for i, s in enumerate(seps + [None]):
        seg = [b for b in cols if last_y <= cy(b) < bounds[i]]
        out += sorted([b for b in seg if cx(b) < page_w / 2], key=cy)
        out += sorted([b for b in seg if cx(b) >= page_w / 2], key=cy)
        if s is not None:
            out.append(s)
            last_y = cy(s)
    return out


# --------------------------------------------------------------------------------------
# Output in the ground-truth text format
# --------------------------------------------------------------------------------------
LIST_START = re.compile(r"^\s*(?:[०-९0-9]{1,3}[.)]|\([^)\s]{1,3}\)|[क-ह][.)]|[a-zA-Z][.)]|[-•●▪◦*])\s")


def _lines_in_rows(lines: List[Dict]) -> List[Dict]:
    """Lines sorted top-to-bottom, left-to-right (same row grouping as _join_lines)."""
    rows: List[List[Dict]] = []
    for l in sorted(lines, key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2):
        cy = (l["bbox"][1] + l["bbox"][3]) / 2
        if rows and min(x["bbox"][1] for x in rows[-1]) <= cy <= max(x["bbox"][3] for x in rows[-1]):
            rows[-1].append(l)
        else:
            rows.append([l])
    return [l for r in rows for l in sorted(r, key=lambda l: l["bbox"][0])]


def block_text(block: Dict, heading_level: int = 2) -> str:
    lines = [l for l in _lines_in_rows(block.get("lines", [])) if l["text"]]
    if not lines:
        return ""
    t = block.get("type", "text")
    if t == "heading":
        return "#" * heading_level + " " + " ".join(l["text"] for l in lines)
    if t == "list":
        items: List[str] = []
        for l in lines:
            if LIST_START.match(l["text"]) or not items:
                items.append(l["text"])
            else:
                items[-1] += " " + l["text"]
        return "\n".join(items)
    if t in ("header", "footer", "caption", "stamp", "signature", "form"):
        return "\n".join(l["text"] for l in lines)
    return " ".join(l["text"] for l in lines)


def serialize_page(ordered_blocks: List[Dict]) -> str:
    parts = []
    for b in ordered_blocks:
        if "table" in b:
            parts.append(b["table"]["html"])
        else:
            s = block_text(b)
            if s:
                parts.append(s)
    return "\n\n".join(parts)
