"""Suggest a table layout from a box drawn on the scan (no HTML, no OCR).

    from table_suggest import suggest_table
    s = suggest_table(gray, [x0, y0, x1, y1])
    s["ok"], s["num_rows"], s["num_cols"], s["cells"], s["html"], s["warnings"], s["skew_deg"]

Built on app/services/measure.py (grid, cells_from_grid, estimate_skew, deskew) and adds what a suggestion needs:
  * deskew first (a tilted page otherwise loses its horizontal rules);
  * faint rules much lighter than the table's own rules (bleed-through from the back of the page)
    are not columns / rows;
  * a cell boundary that is only partly printed (a toner gap, 20-50 % coverage) is merged as
    measure.py decides, but flagged so the annotator can check it;
  * no ruling lines -> ok False with a reason, never an empty grid.
Cells come back as quads / bboxes in the ORIGINAL (possibly tilted) image, text empty.
skew_deg is the page tilt in degrees, OpenCV sign (+ = rotated counter-clockwise).
"""
import cv2
import numpy as np

from app.services import measure as ms


def _darkness(gray, mask, origin, pos, axis, half=2):
    """Median grey value of the rule pixels of the line at `pos` (smaller = darker)."""
    ox, oy = origin
    if axis == "v":
        sl = (slice(None), slice(max(0, pos - ox - half), pos - ox + half + 1))
        g = gray[oy:oy + mask.shape[0], ox:ox + mask.shape[1]][sl]
    else:
        sl = (slice(max(0, pos - oy - half), pos - oy + half + 1), slice(None))
        g = gray[oy:oy + mask.shape[0], ox:ox + mask.shape[1]][sl]
    m = mask[sl].astype(bool)
    return float(np.median(g[m])) if m.any() else 255.0


def _drop_faint(gray, mask, origin, pos, axis, bg, keep_ratio=0.45):
    """Keep lines whose contrast to the paper is at least keep_ratio of the table's typical rule."""
    if len(pos) <= 2:
        return pos, []
    contrast = [bg - _darkness(gray, mask, origin, x, axis) for x in pos]
    ref = float(np.median(contrast))
    keep, dropped = [], []
    for i, (x, c) in enumerate(zip(pos, contrast)):
        outer = i in (0, len(pos) - 1)
        (keep if outer or c >= keep_ratio * ref else dropped).append(x)
    return keep, dropped


def _cells(p, xs, ys, hm, vm, origin):
    """measure.cells_from_grid on given xs / ys, plus the coverage of every merged boundary."""
    ox, oy = origin
    nr, nc = len(ys) - 1, len(xs) - 1
    parent = {(r, c): (r, c) for r in range(nr) for c in range(nc)}
    weak = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    w = max(2, p.px(2))

    def cover_v(x, ya, yb):
        band = vm[ya - oy + p.px(4):yb - oy - p.px(4), max(0, x - ox - w):x - ox + w + 1].any(1)
        return float(band.mean()) if band.size else 1.0

    def cover_h(y, xa, xb):
        band = hm[max(0, y - oy - w):y - oy + w + 1, xa - ox + p.px(4):xb - ox - p.px(4)].any(0)
        return float(band.mean()) if band.size else 1.0

    for r in range(nr):
        for c in range(nc):
            if c + 1 < nc:
                cv_ = cover_v(xs[c + 1], ys[r], ys[r + 1])
                if cv_ < 0.5:
                    parent[find((r, c + 1))] = find((r, c))
                    if cv_ >= 0.2:
                        weak.append(f"row {r + 1}, columns {c + 1}-{c + 2}: merged, but the rule between "
                                    f"them is {round(cv_ * 100)}% printed - check (a gap in the line?)")
            if r + 1 < nr:
                ch = cover_h(ys[r + 1], xs[c], xs[c + 1])
                if ch < 0.5:
                    parent[find((r + 1, c))] = find((r, c))
                    if ch >= 0.2:
                        weak.append(f"column {c + 1}, rows {r + 1}-{r + 2}: merged, but the rule between "
                                    f"them is {round(ch * 100)}% printed - check (a gap in the line?)")
    groups = {}
    for k in parent:
        groups.setdefault(find(k), []).append(k)
    cells = []
    for g in groups.values():
        r0, r1 = min(r for r, _ in g), max(r for r, _ in g)
        c0, c1 = min(c for _, c in g), max(c for _, c in g)
        if len(g) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise ValueError(f"non-rectangular merged cell at row {r0 + 1}, column {c0 + 1}; "
                             "check the ruling lines")
        cells.append({"row": r0, "col": c0, "rowspan": r1 - r0 + 1, "colspan": c1 - c0 + 1,
                      "box": [xs[c0], ys[r0], xs[c1 + 1], ys[r1 + 1]]})
    cells.sort(key=lambda c: (c["row"], c["col"]))
    return cells, weak


def suggest_table(gray, box, pad=40, min_skew=0.2):
    h, w = gray.shape[:2]
    box = [int(v) for v in box]
    big = [max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad)]
    warnings = []

    ang = ms.estimate_skew(gray, big)
    if abs(ang) >= min_skew:
        D, M, Minv = ms.deskew(gray, ang)
        pts = np.array([[big[0], big[1], 1], [big[2], big[1], 1], [big[2], big[3], 1], [big[0], big[3], 1]],
                       float) @ M.T
        big = [int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min())),
               int(min(w, pts[:, 0].max())), int(min(h, pts[:, 1].max()))]
        back = lambda b: ms.to_original(b, Minv)  # noqa: E731
    else:
        D, ang = gray, 0.0
        back = lambda b: ([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], list(b))  # noqa: E731

    p = ms.Page(D)
    xs, ys, hm, vm, origin = ms.grid(p, big)
    if len(xs) < 2 or len(ys) < 2:
        return {"ok": False, "skew_deg": round(-ang, 2), "warnings": [
            f"found {len(xs)} vertical and {len(ys)} horizontal ruling lines - a borderless table? "
            "Draw the cells by hand, or type the HTML and use Snap table"]}

    bg = float(np.median(D[big[1]:big[3], big[0]:big[2]]))
    xs, dx = _drop_faint(D, vm, origin, xs, "v", bg)
    ys, dy = _drop_faint(D, hm, origin, ys, "h", bg)
    if dx or dy:
        warnings.append(f"ignored {len(dx) + len(dy)} faint line(s) much lighter than the table's rules "
                        "(bleed-through from the back of the page?)")

    try:
        cells, weak = _cells(p, xs, ys, hm, vm, origin)
    except ValueError as e:
        return {"ok": False, "skew_deg": round(-ang, 2), "warnings": warnings + [str(e)]}
    warnings += weak

    R, C = len(ys) - 1, len(xs) - 1
    out = []
    for i, c in enumerate(cells):
        q, bb = back(c.pop("box"))
        out.append({"id": i, **c, "bbox": bb, "quad": q, "text": "", "is_header": False,
                    "handwritten": False, "empty": True})
    tq, tb = back([xs[0], ys[0], xs[-1], ys[-1]])
    return {"ok": True, "skew_deg": round(-ang, 2), "num_rows": R, "num_cols": C,
            "region_bbox": tb, "region_quad": tq, "cells": out,
            "html": ms.cells_to_html(out, 0), "warnings": warnings}
