"""measure.py - reference implementation of the box-measuring functions used to build the
gov_005 example labels. Pure numpy + OpenCV, no OCR. All boxes are [x0, y0, x1, y1] in image pixels.

Constants were tuned on a 1191 px wide page; every size is multiplied by `s` = image_width / 1191,
so the same code works on the 2481 px DevaLabel images.
"""
import cv2
import numpy as np

BASE_W = 1191


class Page:
    def __init__(self, gray):
        self.g = gray
        self.h, self.w = gray.shape
        self.s = self.w / BASE_W
        # black text only; light-grey ruling lines and checkbox borders (> 150) are excluded
        self.bw = (gray < 150).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(self.bw, 8)
        lim = max(3, round(6 * self.s))
        small = np.zeros(n, bool)
        small[1:] = (st[1:, cv2.CC_STAT_WIDTH] <= lim) & (st[1:, cv2.CC_STAT_HEIGHT] <= lim)
        self.dots = small[lab].astype(np.uint8)          # periods, colon dots, leader dots
        self.letters = self.bw & (1 - self.dots)           # everything except small dots

    def px(self, v):
        return max(1, round(v * self.s))


def runs(mask, gap):
    """[(start, end)] of True runs in a 1-D mask, merging runs separated by <= gap."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out, s0, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            out.append((int(s0), int(p)))
            s0 = i
        p = i
    out.append((int(s0), int(p)))
    return out


def clip(p, box):
    x0, y0, x1, y1 = box
    return [max(0, int(x0)), max(0, int(y0)), min(p.w, int(x1)), min(p.h, int(y1))]


def ink_box(p, box, pad=3, mask=None):
    """Tight box around the ink inside `box` (None if empty)."""
    x0, y0, x1, y1 = clip(p, box)
    m = (p.bw if mask is None else mask)[y0:y1, x0:x1]
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    pd = p.px(pad)
    return [x0 + int(xs.min()) - pd, y0 + int(ys.min()) - pd, x0 + int(xs.max()) + pd, y0 + int(ys.max()) + pd]


def label_box(p, box, pad=3):
    """Label box: first..last letter, plus a trailing colon/period within ~7 px. Never runs into a dot leader."""
    x0, y0, x1, y1 = clip(p, box)
    xs = np.where(p.letters[y0:y1, x0:x1].any(0))[0]
    if len(xs) == 0:
        return None
    a, b = x0 + int(xs.min()), x0 + int(xs.max())
    return ink_box(p, [a, y0, min(b + p.px(7), x1), y1], pad)


def leader_runs(p, box, min_len=25, gap=12):
    """Value areas over dot leaders inside `box`: runs of small dots longer than min_len px."""
    x0, y0, x1, y1 = clip(p, box)
    col = p.dots[y0:y1, x0:x1].any(0)
    out = []
    for a, b in runs(col, p.px(gap)):
        if b - a > p.px(min_len):
            ys = np.where(p.dots[y0:y1, x0 + a:x0 + b + 1].any(1))[0]
            out.append([x0 + a, y0 + int(ys.min()) - p.px(12), x0 + b + p.px(2), y0 + int(ys.max()) + p.px(3)])
    return out


def text_lines(p, box, expected=None, gap=2, min_h=6):
    """Printed lines inside `box`, top to bottom, each with its own x-extent.
    Lines whose matras/descenders touch are split at the thinnest row (if `expected` is given, until
    there are that many; otherwise any band taller than 1.7x the median band height is split)."""
    x0, y0, x1, y1 = clip(p, box)
    crop = p.bw[y0:y1, x0:x1]
    bands = [(a, b) for a, b in runs(crop.any(1), p.px(gap)) if b - a >= p.px(min_h)]
    if not bands:
        return []

    def split_tallest():
        i = max(range(len(bands)), key=lambda k: bands[k][1] - bands[k][0])
        a, b = bands[i]
        m6 = p.px(6)
        prof = crop[a:b + 1].sum(1)
        if len(prof) <= 2 * m6 + 1:
            return False
        m = a + m6 + int(np.argmin(prof[m6:-m6]))
        bands[i:i + 1] = [(a, m - 1), (m + 1, b)]
        return True

    if expected:
        while len(bands) < expected and split_tallest():
            pass
    else:
        med = float(np.median([b - a for a, b in bands]))
        while max(b - a for a, b in bands) > 1.7 * med and split_tallest():
            pass
    return [ink_box(p, [x0, y0 + a, x1, y0 + b + 1]) for a, b in bands]


def words(p, box, gap=14):
    """Horizontal segments of letters on one line (gap in px). Use gap ~14 for label/option splits."""
    x0, y0, x1, y1 = clip(p, box)
    return [ink_box(p, [x0 + a, y0, x0 + b + 1, y1]) for a, b in runs(p.letters[y0:y1, x0:x1].any(0), p.px(gap))]


def squares(p, box, thr=None):
    """Printed checkbox squares (light or dark border, empty inside) in `box`."""
    thr = line_threshold(p, box) if thr is None else thr
    x0, y0, x1, y1 = clip(p, box)
    m = (p.g[y0:y1, x0:x1] < thr).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if p.px(25) <= w <= p.px(90) and p.px(25) <= h <= p.px(90) and 0.75 <= w / h <= 1.6:
            out.append([x0 + x, y0 + y, x0 + x + w, y0 + y + h])
    out.sort()
    keep = []                                  # drop inner contours of the same square
    for b in out:
        if not any(abs(b[0] - k[0]) < p.px(6) and abs(b[1] - k[1]) < p.px(6) for k in keep):
            keep.append(b)
    return keep


def tick_ratio(p, sq):
    """Ink fraction inside a square, border excluded (> 0.08 = probably ticked)."""
    m = p.px(5)
    x0, y0, x1, y1 = sq
    inner = p.bw[y0 + m:y1 - m, x0 + m:x1 - m]
    return float(inner.mean()) if inner.size else 0.0


# ------------------------------------------------------------------------------------------ tables
def _group(vals, tol):
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(round(np.mean(g))) for g in out]


def line_threshold(p, box):
    """Ruling lines are often light grey: anything clearly darker than the page background counts."""
    x0, y0, x1, y1 = clip(p, box)
    bg = float(np.median(p.g[y0:y1, x0:x1]))
    return min(245.0, bg - 12)


def grid(p, box, thr=None):
    """Ruling-line grid of the table in `box`.
    Returns xs, ys (sorted line positions), the line masks, and the box origin.
    Horizontal lines must be longer than any word headline (shirorekha) and touch >= 2 vertical lines."""
    x0, y0, x1, y1 = clip(p, box)
    thr = line_threshold(p, box) if thr is None else thr
    m = (p.g[y0:y1, x0:x1] < thr).astype(np.uint8)
    hk = max(p.px(60), int(0.12 * (x1 - x0)))
    hm = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vm = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, p.px(20))))
    xs = _group(list(np.where(vm.sum(0) > p.px(15))[0]), p.px(3))
    ys = []
    for y in _group(list(np.where(hm.sum(1) > hk)[0]), p.px(3)):
        row = hm[max(0, y - 2):y + 3].any(0)
        touching = sum(1 for x in xs if row[max(0, x - 3):x + 4].any() and
                       vm[max(0, y - p.px(6)):y + p.px(6), max(0, x - 2):x + 3].any())
        if touching >= 2:
            ys.append(y)
    return [x0 + x for x in xs], [y0 + y for y in ys], hm, vm, (x0, y0)


def cells_from_grid(p, box, thr=None):
    """Cells with rowspan/colspan inferred from MISSING ruling segments (no text needed).
    Two neighbouring grid cells are merged when the boundary between them has < 50% line coverage."""
    xs, ys, hm, vm, (ox, oy) = grid(p, box, thr)
    nr, nc = len(ys) - 1, len(xs) - 1
    parent = {(r, c): (r, c) for r in range(nr) for c in range(nc)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def cover_v(x, ya, yb):                     # vertical boundary at x between ya..yb
        xi = x - ox
        w = max(2, p.px(2))                     # tolerance: lines wander a few px on scans
        band = vm[ya - oy + p.px(4):yb - oy - p.px(4), max(0, xi - w):xi + w + 1].any(1)
        return band.mean() if band.size else 1.0

    def cover_h(y, xa, xb):
        yi = y - oy
        w = max(2, p.px(2))
        band = hm[max(0, yi - w):yi + w + 1, xa - ox + p.px(4):xb - ox - p.px(4)].any(0)
        return band.mean() if band.size else 1.0

    for r in range(nr):
        for c in range(nc):
            if c + 1 < nc and cover_v(xs[c + 1], ys[r], ys[r + 1]) < 0.5:
                parent[find((r, c + 1))] = find((r, c))
            if r + 1 < nr and cover_h(ys[r + 1], xs[c], xs[c + 1]) < 0.5:
                parent[find((r + 1, c))] = find((r, c))
    groups = {}
    for k in parent:
        groups.setdefault(find(k), []).append(k)
    cells = []
    for g in groups.values():
        r0, r1 = min(r for r, _ in g), max(r for r, _ in g)
        c0, c1 = min(c for _, c in g), max(c for _, c in g)
        if len(g) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise ValueError(f"non-rectangular merged cell at row {r0} col {c0}; check the ruling lines")
        cells.append({"row": r0, "col": c0, "rowspan": r1 - r0 + 1, "colspan": c1 - c0 + 1,
                      "bbox": [xs[c0], ys[r0], xs[c1 + 1], ys[r1 + 1]]})
    cells.sort(key=lambda c: (c["row"], c["col"]))
    for i, c in enumerate(cells):
        c["id"] = i
    return xs, ys, cells


def cell_text_lines(p, cell_bbox, expected=None):
    """Line boxes inside one cell (inset 3 px so the ruling lines are never included)."""
    x0, y0, x1, y1 = cell_bbox
    i = p.px(3)
    return text_lines(p, [x0 + i, y0 + i, x1 - i, y1 - i], expected)


def cells_to_html(cells, header_rows):
    rows = {}
    for c in sorted(cells, key=lambda c: (c["row"], c["col"])):
        tag = "th" if c["row"] < header_rows else "td"
        at = (f' rowspan="{c["rowspan"]}"' if c["rowspan"] > 1 else "") + \
             (f' colspan="{c["colspan"]}"' if c["colspan"] > 1 else "")
        txt = c.get("text", "").replace("\n", "<br>")
        rows.setdefault(c["row"], []).append(f"<{tag}{at}>{txt}</{tag}>")
    return "<table>" + "".join("<tr>" + "".join(rows[r]) + "</tr>" for r in sorted(rows)) + "</table>"


def ruling_segments(xs, ys, cells, table_index=0, width=1):
    """ruling_lines[] entries: only the segments that are really printed (not crossing a merged cell)."""
    out = []
    nr, nc = len(ys) - 1, len(xs) - 1
    inside = lambda r, c: next(k for k in cells if k["row"] <= r < k["row"] + k["rowspan"]
                               and k["col"] <= c < k["col"] + k["colspan"])
    for j, y in enumerate(ys):
        seg = None
        for c in range(nc):
            printed = j in (0, nr) or inside(j - 1, c) is not inside(j, c)
            if printed:
                seg = [seg[0], xs[c + 1]] if seg and seg[1] == xs[c] else seg
                if seg is None or seg[1] != xs[c + 1]:
                    if seg:
                        out.append(("h", seg, y))
                    seg = [xs[c], xs[c + 1]]
            elif seg:
                out.append(("h", seg, y))
                seg = None
        if seg:
            out.append(("h", seg, y))
    for i, x in enumerate(xs):
        seg = None
        for r in range(nr):
            printed = i in (0, nc) or inside(r, i - 1) is not inside(r, i)
            if printed:
                if seg and seg[1] == ys[r]:
                    seg[1] = ys[r + 1]
                else:
                    if seg:
                        out.append(("v", seg, x))
                    seg = [ys[r], ys[r + 1]]
            elif seg:
                out.append(("v", seg, x))
                seg = None
        if seg:
            out.append(("v", seg, x))
    res = []
    for o, (a, b), k in out:
        p0, p1 = ([a, k], [b, k]) if o == "h" else ([k, a], [k, b])
        bb = [a, k, b, k + width] if o == "h" else [k, a, k + width, b]
        res.append({"table_index": table_index, "orientation": o, "width": width, "p0": p0, "p1": p1, "bbox": bb})
    return res


# ------------------------------------------------------------------------------------------ JSON repair
def repair_structure(rec):
    """Renumber layout ids 0..N-1 (in reading order) and remap every reference; drop empty tables."""
    L = sorted(rec["layout"], key=lambda b: (b.get("reading_order") is None, b.get("reading_order") or 0,
                                             b["bbox"][1], b["bbox"][0]))
    idmap = {b["id"]: i for i, b in enumerate(L)}
    tmap, tables = {}, []
    for i, t in enumerate(rec.get("tables", [])):
        if t.get("cells"):
            tmap[i] = len(tables)
            tables.append(t)
    k = 0
    for i, b in enumerate(L):
        b["id"] = i
        if b.get("reading_order") is not None:
            b["reading_order"] = k
            k += 1
        if "table_index" in b:
            b["table_index"] = tmap.get(b["table_index"], b["table_index"])
        for f in b.get("fields", []):
            if f.get("label") and "region_id" in f["label"]:
                f["label"]["region_id"] = idmap[f["label"]["region_id"]]
    for l in rec.get("lines", []):
        l["block_id"] = idmap[l["block_id"]]
        if "table_index" in l:
            l["table_index"] = tmap[l["table_index"]]
    for r in rec.get("ruling_lines", []):
        r["table_index"] = tmap.get(r["table_index"], r["table_index"])
    rec["layout"], rec["tables"] = L, tables
    rec["blocks"] = [{"type": b["type"], "bbox": b["bbox"], "quad": b.get("quad"), "block_id": b["id"],
                      "font_id": None} for b in L]
    return rec


def grid_errors(cells, num_rows, num_cols):
    """Every grid slot covered exactly once, or a list of problems."""
    occ = {}
    errs = []
    for c in cells:
        for r in range(c["row"], c["row"] + c["rowspan"]):
            for k in range(c["col"], c["col"] + c["colspan"]):
                if (r, k) in occ:
                    errs.append(f"slot ({r},{k}) covered by cells {occ[(r, k)]} and {c['id']}")
                occ[(r, k)] = c["id"]
    for r in range(num_rows):
        for k in range(num_cols):
            if (r, k) not in occ:
                errs.append(f"slot ({r},{k}) not covered - add colspan/rowspan or a cell")
    return errs


# ------------------------------------------------------------------------------------------ skew
def estimate_skew(gray, box=None, max_deg=2.0, step=0.02, dark=160):
    """Skew angle in degrees (OpenCV sign: pass it straight to deskew). Projection-profile method on the
    dark pixels of `box` (use the table area when there is one); works on a half-size copy for speed."""
    g = gray if box is None else gray[box[1]:box[3], box[0]:box[2]]
    m = (g < dark).astype(np.uint8)
    m = cv2.resize(m, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = m.shape
    best = (-1.0, 0.0)
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1)
        r = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
        score = float((r.sum(1).astype(np.float64) ** 2).sum())
        if score > best[0]:
            best = (score, float(a))
    return round(best[1], 2)


def deskew(gray, angle):
    """Rotate the whole page by `angle` around its centre. Returns (deskewed image, M, Minv);
    measure on the deskewed image, then map boxes back with to_original(box, Minv)."""
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    border = 255 if gray.ndim == 2 else (255, 255, 255)
    D = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)
    return D, M, cv2.invertAffineTransform(M)


def to_original(box, Minv):
    """Box measured on the deskewed page -> (quad, bbox) in the original image."""
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float) @ np.asarray(Minv).T
    q = [[int(round(x)), int(round(y))] for x, y in pts]
    return q, [min(c[0] for c in q), min(c[1] for c in q), max(c[0] for c in q), max(c[1] for c in q)]


def remove_rules(p):
    """Take ruling lines out of the text masks (call after Page(), before measuring text inside tables)."""
    hl = cv2.morphologyEx(p.bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (p.px(60), 1)))
    vl = cv2.morphologyEx(p.bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, p.px(18))))
    r = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    p.bw, p.letters, p.dots = p.bw & (1 - r), p.letters & (1 - r), p.dots & (1 - r)
    return p
