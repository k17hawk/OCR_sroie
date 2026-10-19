"""Detect a ruled table's grid and merged cells from its printed lines.

usage: python3 table_grid.py scan.png [out_prefix]
writes <out_prefix>_grid.json, <out_prefix>_skeleton.html, <out_prefix>_overlay.png

Rules of thumb it relies on
  * a real rule connects the two rules that bound it (a word's शिरोरेखा or a
    glyph stem never touches both), so every cell edge is tested for that;
  * candidate lines are only positions; a candidate with no confirmed edge is dropped;
  * merged cells = grid cells joined across edges that have no rule.
"""
import json, math, sys
import cv2
import numpy as np


# ------------------------------------------------------------------ helpers
def binarize(gray, loose=None):
    """scans: Otsu on a background-flattened page (ignores paper texture/yellowing).
    loose: fixed cut for born-digital pages whose rules are very light grey."""
    if loose is not None:
        return (gray < loose).astype(np.uint8)
    bg = cv2.medianBlur(gray, 51)
    flat = cv2.divide(gray, bg, scale=255)
    _, b = cv2.threshold(flat, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return b.astype(np.uint8)


def text_height(bw):
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    h = st[1:, 3]
    h = h[(h > 5) & (h < 200)]
    return int(np.median(h)) if len(h) else 20


def deskew(gray, bw, loose=None):
    lines = cv2.HoughLinesP(cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, 80), np.uint8)) * 255,
                            1, np.pi / 1800, 200, minLineLength=gray.shape[1] // 4, maxLineGap=10)
    if lines is None:
        return gray, bw, 0.0, np.eye(2, 3)
    angs = [math.degrees(math.atan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
    ang = float(np.median([a for a in angs if abs(a) < 5] or [0]))
    M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), ang, 1.0)
    g2 = cv2.warpAffine(gray, M, gray.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=255)
    return g2, binarize(g2, loose), ang, M


def cluster(vals, tol):
    vals = sorted(vals)
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(round(np.mean(g))) for g in out]


def candidates(bw, th):
    """positions of rule pieces that belong to the table frame"""
    hk = max(40, 3 * th)
    vk = max(20, int(1.2 * th))
    hm = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vm = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    # the table is the largest connected network of rule pieces; text stems and
    # शिरोरेखा don't touch the rules, so they fall outside it
    net = cv2.dilate(((hm | vm) > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(net, 8)
    k = 1 + int(np.argmax(st[1:, 4]))
    inside = lab == k
    ys, xs = [], []
    for m, out, o in ((hm, ys, 'h'), (vm, xs, 'v')):
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n2):
            x, y, w, h, a = st2[i]
            if not inside[lab2 == i].any():
                continue
            out.append(y + h // 2 if o == 'h' else x + w // 2)
    tb = st[k, :4]
    return cluster(ys, max(6, th // 2)), cluster(xs, max(6, th // 2)), tb


def best_line(strip, gap):
    """strip: (thickness x length). Best single pixel-row (3-px wobble allowed).
    returns coverage, index of that row, and the closed row itself"""
    best, bi, brow = -1.0, 0, None
    for i in range(strip.shape[0]):
        row = strip[max(0, i - 1):i + 2].max(0).astype(np.uint8)
        row = cv2.morphologyEx(row[None, :], cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8))[0]
        if row.mean() > best:
            best, bi, brow = row.mean(), i, row
    return best, bi, brow


def reaches_ends(row, k=6):
    return row[:k].sum() >= k - 2 and row[-k:].sum() >= k - 2


def h_edge(bw, y, x0, x1, pad=4, trim=4):
    """horizontal rule from vertical x0 to vertical x1 near y.
    A rule (1) is one thin line inked along most of the edge, (2) runs into both
    verticals, and (3) has little ink just above/below it. A शिरोरेखा fails (2) and (3):
    its glyph bodies hang right under it."""
    xa, xb = x0 + trim, x1 - trim + 1
    if xb - xa < 8:
        return False
    s = bw[max(0, y - pad):y + pad + 1, xa:xb]
    cov, i, row = best_line(s, 15)
    if cov < (0.70 if xb - xa >= 60 else 0.88) or not reaches_ends(row):
        return False
    yy = max(0, y - pad) + i
    below = bw[yy + 4:yy + 8, xa:xb].any(0).mean()
    above = bw[max(0, yy - 7):yy - 3, xa:xb].any(0).mean()
    return min(below, above) < 0.35


def v_edge(bw, x, y0, y1, pad=5, trim=3):
    """vertical rule from horizontal y0 to y1 near x. A rule is a thin straight line:
    the columns 3-5 px beside it are nearly empty. A glyph that happens to sit on the
    position (e.g. सु in 'ख. सुविधा') spreads ink across the band, so it fails."""
    xa = max(0, x - pad)
    s = bw[y0 + trim:y1 - trim + 1, xa:x + pad + 1].T
    if s.shape[1] < 8:
        return False
    cov, i, row = best_line(s, 3)
    if cov < 0.75:
        return False
    k = 4
    if not (row[:k].sum() >= k - 2 or row[-k:].sum() >= k - 2):
        return False
    side = [s[j].mean() for j in list(range(0, max(0, i - 2))) + list(range(i + 3, s.shape[0]))]
    return (float(np.mean(side)) if side else 0.0) < 0.3


# ------------------------------------------------------------------ main
def detect(path, loose=None):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bw = binarize(gray, loose)
    gray, bw, ang, M = deskew(gray, bw, loose)
    th = text_height(bw)
    Ys, Xs, (tx, ty, tw, tht) = candidates(bw, th)
    Ys = [y for y in Ys if ty - 3 <= y <= ty + tht + 3]
    Xs = [x for x in Xs if tx - 3 <= x <= tx + tw + 3]

    # iterate: confirm edges, drop candidate lines that never confirm, repeat
    for _ in range(4):
        R, C = len(Ys) - 1, len(Xs) - 1
        H = np.array([[h_edge(bw, Ys[j], Xs[i], Xs[i + 1]) for i in range(C)] for j in range(R + 1)])
        V = np.array([[v_edge(bw, Xs[i], Ys[j], Ys[j + 1]) for i in range(C + 1)] for j in range(R)])
        keepY = [j for j in range(R + 1) if j in (0, R) or H[j].any()]
        keepX = [i for i in range(C + 1) if i in (0, C) or V[:, i].any()]
        if len(keepY) == R + 1 and len(keepX) == C + 1:
            break
        Ys = [Ys[j] for j in keepY]
        Xs = [Xs[i] for i in keepX]

    # merge grid cells across missing edges (union-find)
    R, C = len(Ys) - 1, len(Xs) - 1
    par = list(range(R * C))
    find = lambda a: a if par[a] == a else find(par[a])
    def uni(a, b):
        a, b = find(a), find(b)
        if a != b:
            par[b] = a
    for r in range(R):
        for c in range(C):
            if c + 1 < C and not V[r, c + 1]:
                uni(r * C + c, r * C + c + 1)
            if r + 1 < R and not H[r + 1, c]:
                uni(r * C + c, (r + 1) * C + c)
    groups = {}
    for r in range(R):
        for c in range(C):
            groups.setdefault(find(r * C + c), []).append((r, c))
    cells, warnings = [], []
    for g in groups.values():
        rs = [p[0] for p in g]; cs = [p[1] for p in g]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if len(g) != (r1 - r0 + 1) * (c1 - c0 + 1):
            warnings.append(f"non-rectangular merge at rows {r0}-{r1}, cols {c0}-{c1}")
        cells.append({"row": r0, "col": c0, "rowspan": r1 - r0 + 1, "colspan": c1 - c0 + 1,
                      "bbox_deskewed": [Xs[c0], Ys[r0], Xs[c1 + 1], Ys[r1 + 1]]})
    cells.sort(key=lambda c: (c["row"], c["col"]))
    return dict(skew_deg=round(-ang, 3), text_height=th, rows=R, cols=C, ys=Ys, xs=Xs,
                cells=cells, warnings=warnings, M=M.tolist()), gray


def skeleton(res):
    rows = {}
    for c in res["cells"]:
        at = (f' rowspan="{c["rowspan"]}"' if c["rowspan"] > 1 else "") + \
             (f' colspan="{c["colspan"]}"' if c["colspan"] > 1 else "")
        rows.setdefault(c["row"], []).append(f"<td{at}></td>")
    return "<table>\n" + "\n".join("<tr>" + "".join(rows.get(r, [])) + "</tr>" for r in range(res["rows"])) + "\n</table>\n"


if __name__ == "__main__":
    src = sys.argv[1]
    pre = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0]
    res, g = detect(src)
    page_bg = np.median(cv2.imread(src, cv2.IMREAD_GRAYSCALE))
    if (res["rows"] < 2 or res["cols"] < 2) and page_bg >= 250:
        # born-digital page with faint rules: retry with a loose fixed threshold
        res, g = detect(src, loose=240)
    json.dump(res, open(pre + "_grid.json", "w"), indent=1)
    open(pre + "_skeleton.html", "w").write(skeleton(res))
    ov = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for k, c in enumerate(res["cells"]):
        x0, y0, x1, y1 = c["bbox_deskewed"]
        cv2.rectangle(ov, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (0, 0, 255), 2)
        cv2.putText(ov, f'{c["row"]},{c["col"]}' + (f' {c["rowspan"]}x{c["colspan"]}' if c["rowspan"] * c["colspan"] > 1 else ""),
                    (x0 + 6, y0 + 18), 0, 0.45, (200, 0, 0), 1)
    cv2.imwrite(pre + "_overlay.png", ov)
    print(f'{res["rows"]} rows x {res["cols"]} cols, {len(res["cells"])} cells, skew {res["skew_deg"]}°')
    for w in res["warnings"]:
        print("WARN", w)
