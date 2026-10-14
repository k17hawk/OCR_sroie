"""
Applies the synthetic-v2 export fixes to your OCR tool and writes the
complete patched file next to it.

Usage:
    python patch_ocr_tool.py ocr_tool.py            -> writes ocr_tool_patched.py
    python patch_ocr_tool.py ocr_tool.py --inplace  -> overwrites (keeps ocr_tool.py.bak)

Every patch checks that its anchor text occurs exactly once, so nothing is
changed silently. If an anchor is not found (e.g. you edited that part),
the script stops and tells you which fix could not be applied.

Fixes:
  1  Blocks inside a table (lists, paragraphs, headings, forms) are folded
     into the table, so table cells get their lines and nothing is duplicated.
  2  A text block that wraps 2+ other blocks (a layout-model "container"
     region) is dropped instead of being exported on top of them.
  3  A line box that contains another line box with the same text (a merged
     row like "श्री कार्यालय प्रमुखज्यू मिति....") is split: the duplicate part
     is dropped and the rest ("मिति....") is kept as its own, tightened line.
  4  estimate_block_lines ignores thin ink rows (underline / footnote-rule
     remnants), so words are no longer put into 2-3 px boxes.
  5  tighten_to_ink only erases an edge "vertical rule" if it continues
     outside the box - a danda "।" at the end of a line is kept.
  6  digit_style is "mixed" when both 0-9 and ०-९ occur.
  7  Block bbox/quad is shrunk to the union of its lines (+ small pad).
  8  Dot leaders ("मिति......", "......कार्यालय") become a label with an
     empty value instead of a label containing the dots.
  9  New checkbox "Re-tighten loaded lines": line boxes loaded from a JSON
     are fitted to the ink again on export (fixes loose saved boxes).
 10  tighten_to_ink ignores cell borders near the box edge and thin pieces
     of table rules, so line boxes inside table cells fit the text.
 11  2-3 px line boxes saved by older versions are repaired: joined to the
     line on the same row (word order taken from the block text) or refitted.
 12  Long table rules are detected even when text touches them.
 13  A table row/column edge is put on the rule between the two rows.
 14  Specks above/below the text (rule fragments) no longer stretch a line box.
"""

import os
import py_compile
import shutil
import sys

PATCHES = []


def patch(name, old, new):
    PATCHES.append((name, old, new))


# ---------------------------------------------------------------- 1 fold into table
patch("1 fold blocks into tables",
r'''    "form_field": {"form_field", "paragraph"},
}
SYNTH_NO_LINES = {"figure", "logo", "stamp"}''',
r'''    "form_field": {"form_field", "paragraph"},
    "table": {"list", "paragraph", "heading", "form_field"},
}
SYNTH_NO_LINES = {"figure", "logo", "stamp"}''')

# ---------------------------------------------------------------- 4 thin rows
patch("4 ignore thin ink rows in estimate_block_lines",
r'''    rows = [r for r in merged if r[1] - r[0] + 1 >= 3]''',
r'''    row_hs = sorted(r[1] - r[0] + 1 for r in merged)
    min_row_h = max(3.0, 0.4 * row_hs[len(row_hs) // 2]) if row_hs else 3.0
    rows = [r for r in merged if r[1] - r[0] + 1 >= min_row_h]''')

# ---------------------------------------------------------------- 5 danda kept
patch("5a continues_v helper in tighten_to_ink",
r'''    keep = np.zeros(n, dtype=bool)
''',
r'''    def continues_v(i):
        """True if the component's ink carries on above or below the box -
        a real table rule does, a danda at the end of a line does not."""
        _rr, cc = np.nonzero(lab == i)
        if not cc.size:
            return True
        c0, c1 = int(cc.min()) + ix, int(cc.max()) + ix + 1
        top, bot = iy, iy + ch
        above = top > 0 and bool(bw_out[max(0, top - 2):top, c0:c1].any())
        below = bot < bw_out.shape[0] and bool(bw_out[bot:bot + 2, c0:c1].any())
        return above or below

    keep = np.zeros(n, dtype=bool)
''')

patch("5b only drop vertical rules that continue outside the box",
r'''        if rule_h or rule_v:
            continue''',
r'''        if rule_h or (rule_v and continues_v(i)):
            continue''')

# ---------------------------------------------------------------- 6 mixed digits
patch("6 digit_style mixed",
r'''    if devd or latd:
        digits = "dev" if devd > latd else "latin"''',
r'''    if devd and latd:
        digits = "mixed"
    elif devd or latd:
        digits = "dev" if devd > latd else "latin"''')

# ---------------------------------------------------------------- 8 dot leaders
patch("8 dot leaders are empty values",
r'''def _split_form_line(L):
    t = L["text"]
    m = _FORM_SPLIT_RE.match(t)''',
r'''_LEADER_RE = re.compile(r"[.\u2026_]{4,}")


def _split_form_line(L):
    t = L["text"]
    leaders = list(_LEADER_RE.finditer(t))
    if leaders:
        rest = re.sub(r"\s+", " ", _LEADER_RE.sub(" ", t)).strip()
        if any(c.isalnum() for c in rest):
            inside = [False] * len(t)
            for lm in leaders:
                for k in range(lm.start(), lm.end()):
                    inside[k] = True
            idx = [k for k, c in enumerate(t) if not inside[k] and not c.isspace()]
            x0, y0, x1, y1 = L["xyxy"]
            n = float(max(1, len(t)))
            lx0 = x0 + (x1 - x0) * idx[0] / n
            lx1 = x0 + (x1 - x0) * (idx[-1] + 1) / n
            # the dots are the blank to fill in: label only, value stays null
            return [_make_line(rest, [round(lx0, 1), y0, round(lx1, 1), y1],
                               role="label", src=L)], False
        # only dots (+ danda): a blank line to fill in, not a missing separator
        return [dict(L, role="label")], False
    m = _FORM_SPLIT_RE.match(t)''')

patch("8b dot leaders also stripped from saved label lines",
r'''        if L["role"] is None:''',
r'''        if L["role"] is None or (L["role"] == "label" and _LEADER_RE.search(L["text"] or "")):''')

# ---------------------------------------------------------------- 9 re-tighten (signature)
patch("9a build_synth_page retighten parameter",
r'''def build_synth_page(dets, width, height, meta, page_img=None, tighten=True, rulings=True):''',
r'''def build_synth_page(dets, width, height, meta, page_img=None, tighten=True, rulings=True,
                     retighten=False):''')

patch("9b re-tighten loaded line boxes",
r'''    for d in lines:
        x, y, w, h = d["bbox"]
        xyxy, quad = [x, y, x + w, y + h], None
        if d.get("quad") or d.get("src_xyxy"):
            quad = det_quad(d)
            xyxy = det_xyxy(d)
        elif tighten and gray is not None:
            t = tighten_to_ink(gray, d["bbox"])
            if t:
                xyxy, quad, ang = t
                if quad:
                    angles.append(ang)
        text = _norm_text(d.get("text"))''',
r'''    for d in lines:
        x, y, w, h = d["bbox"]
        xyxy, quad = [x, y, x + w, y + h], None
        loaded = bool(d.get("quad") or d.get("src_xyxy"))
        t = None
        if tighten and gray is not None and (retighten or not loaded):
            t = tighten_to_ink(gray, d["bbox"])
        if t:
            xyxy, quad, ang = t
            if quad:
                angles.append(ang)
        elif loaded:
            quad = det_quad(d)
            xyxy = det_xyxy(d)
        text = _norm_text(d.get("text"))''')

# ---------------------------------------------------------------- 3 duplicate lines
patch("3 drop merged duplicate line boxes",
r'''    folded = set()
    for inner in sorted(blocks, key=lambda b: _area_xywh(b["bbox"])):''',
r'''    dup_lines, rem_lines = set(), []
    for big in L_all:
        if not big["text"]:
            continue
        for small in L_all:
            if small is big or id(small) in dup_lines or not small["text"]:
                continue
            sa = max(1, _area_xywh(small["bbox"]))
            if (_area_xywh(big["bbox"]) > 1.5 * sa
                    and _inter_xywh(small["bbox"], big["bbox"]) >= 0.8 * sa
                    and small["text"] in big["text"]):
                dup_lines.add(id(big))
                # keep the part of the merged line that is NOT the smaller line
                # ("श्री कार्यालय प्रमुखज्यू | मिति......") as its own line box
                bt_, st_, rem, side = big["text"], small["text"], "", None
                if bt_.startswith(st_):
                    rem, side = bt_[len(st_):].strip(), "right"
                elif bt_.endswith(st_):
                    rem, side = bt_[:-len(st_)].strip(), "left"
                note = "dropped as a merged duplicate"
                if rem:
                    bx0, by0, bx1, by1 = big["xyxy"]
                    sx0, _sy0, sx1, _sy1 = small["xyxy"]
                    box = [sx1 + 1, by0, bx1, by1] if side == "right" else [bx0, by0, sx0 - 1, by1]
                    if box[2] - box[0] >= 4:
                        xyxy_r, quad_r = box, None
                        if gray is not None:
                            tt = tighten_to_ink(gray, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
                            if tt:
                                xyxy_r, quad_r, _a = tt
                        rem_lines.append(_make_line(rem, xyxy_r, quad_r, big.get("role")))
                        note = f"split - '{rem[:20]}' kept as its own line"
                warns.append(f"line '{big['text'][:30]}' {_xyxy_ints(big['xyxy'])} contains the "
                             f"line '{small['text'][:20]}' - {note}")
                break
    L_all = [L for L in L_all if id(L) not in dup_lines] + rem_lines

    folded = set()
    for inner in sorted(blocks, key=lambda b: _area_xywh(b["bbox"])):''')

# ---------------------------------------------------------------- 2 container regions
patch("2 drop container blocks that wrap other blocks",
r'''    blocks = [b for b in blocks if id(b) not in folded]
''',
r'''    blocks = [b for b in blocks if id(b) not in folded]

    wrappers = set()
    for outer in sorted(blocks, key=lambda b: -_area_xywh(b["bbox"])):
        if outer["type"] in ("table", "header", "footer") or outer["type"] in SYNTH_NO_LINES:
            continue
        if outer["det"].get("label_fixed"):
            continue
        oa = _area_xywh(outer["bbox"])
        inner = [b for b in blocks
                 if b is not outer and id(b) not in wrappers
                 and oa > _area_xywh(b["bbox"])
                 and _inter_xywh(b["bbox"], outer["bbox"]) >= 0.85 * max(1, _area_xywh(b["bbox"]))]
        if len(inner) >= 2:
            wrappers.add(id(outer))
            warns.append(f"'{canonical_label(outer['det'].get('label'))}' box "
                         f"{_xyxy_ints(_bb(outer['det']))} wraps {len(inner)} other blocks - "
                         "dropped (layout-model container region)")
    blocks = [b for b in blocks if id(b) not in wrappers]
''')

# ---------------------------------------------------------------- 7 shrink block boxes
patch("7 shrink block bbox to its lines",
r'''    for bid, b in enumerate(ordered):
        typ, d, extra = b["type"], b["det"], b["extra"]
        x, y, w, h = b["bbox"]
        bxyxy = _xyxy_ints([x, y, x + w, y + h])
        bquad = det_quad(d) if d.get("quad") else _rect_quad(*bxyxy)''',
r'''    for bid, b in enumerate(ordered):
        typ, d, extra = b["type"], b["det"], b["extra"]
        shrunk = False
        if b["lines"] and typ != "table" and typ not in SYNTH_NO_LINES:
            lx0 = min(L["xyxy"][0] for L in b["lines"])
            ly0 = min(L["xyxy"][1] for L in b["lines"])
            lx1 = max(L["xyxy"][2] for L in b["lines"])
            ly1 = max(L["xyxy"][3] for L in b["lines"])
            lhs = sorted(L["xyxy"][3] - L["xyxy"][1] for L in b["lines"])
            pad = max(2, int(round(0.25 * lhs[len(lhs) // 2])))
            nx0, ny0 = max(0, int(lx0) - pad), max(0, int(ly0) - pad)
            nx1, ny1 = int(round(lx1)) + pad, int(round(ly1)) + pad
            if width:
                nx1 = min(int(width), nx1)
            if height:
                ny1 = min(int(height), ny1)
            b["bbox"] = (nx0, ny0, max(1, nx1 - nx0), max(1, ny1 - ny0))
            shrunk = True
        x, y, w, h = b["bbox"]
        bxyxy = _xyxy_ints([x, y, x + w, y + h])
        bquad = det_quad(d) if d.get("quad") and not shrunk else _rect_quad(*bxyxy)''')

# ---------------------------------------------------------------- 9 UI + call site
patch("9c pass retighten from the UI",
r'''rulings=self.synth_rules_var.get())''',
r'''rulings=self.synth_rules_var.get(),
                                              retighten=self.synth_retight_var.get())''')

patch("9d Re-tighten checkbox",
r'''                        variable=self.synth_rules_var).pack(side=tk.LEFT, padx=2)
''',
r'''                        variable=self.synth_rules_var).pack(side=tk.LEFT, padx=2)
        self.synth_retight_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(syn_row, text="Re-tighten loaded lines",
                        variable=self.synth_retight_var).pack(side=tk.LEFT, padx=2)
''')

# ---------------------------------------------------------------- 10 borders / rule fragments
patch("10 tighten_to_ink ignores cell borders and rule fragments",
r'''    n, lab, st, _c = cv2.connectedComponentsWithStats(bw, 8)

    def cut_at''',
r'''    # cell borders: an edge column that is mostly ink and continues outside the box
    colcov = (bw > 0).mean(axis=0)
    edge_ = min(15, max(3, int(0.05 * cw)))
    for c in sorted(set(list(range(min(edge_, cw))) + list(range(max(0, cw - edge_), cw)))):
        if colcov[c] >= 0.5:
            cx_ = c + ix
            up = iy > 0 and bool(bw_out[max(0, iy - 2):iy, cx_].any())
            down = iy + ch < bw_out.shape[0] and bool(bw_out[iy + ch:iy + ch + 2, cx_].any())
            if up or down:
                bw[:, max(0, c - 1):c + 2] = 0
    # pieces of a horizontal rule: a thin (<= 3 px) wide run of ink with blank
    # rows above and below. Kept if it is the only ink in the box (a dot leader).
    rowink = (bw > 0).sum(axis=1)
    thr = max(2.0, 0.02 * cw)
    runs_, a_ = [], None
    for r_, v_ in enumerate(list(rowink > thr) + [False]):
        if v_ and a_ is None:
            a_ = r_
        elif not v_ and a_ is not None:
            runs_.append((a_, r_ - 1))
            a_ = None
    if len(runs_) > 1:
        for a_, b_ in runs_:
            if b_ - a_ + 1 > 3 or rowink[a_:b_ + 1].max() < 0.15 * cw:
                continue
            above = a_ == 0 or bool((rowink[max(0, a_ - 2):a_] <= thr).all())
            below = b_ == ch - 1 or bool((rowink[b_ + 1:b_ + 3] <= thr).all())
            if above and below:
                bw[max(0, a_ - 1):b_ + 2, :] = 0
    n, lab, st, _c = cv2.connectedComponentsWithStats(bw, 8)

    def cut_at''')

# ---------------------------------------------------------------- 11 repair flat saved lines
patch("11 repair 2-3 px line boxes from older saves",
r'''            blocks.append(best)
        best["lines"].append(L)
''',
r'''            blocks.append(best)
        best["lines"].append(L)

    # Line boxes only 2-3 px tall (saved by older versions from underline /
    # rule remnants): join the text to the line on the same row, choosing the
    # word order that matches the block text, or refit the box to the ink.
    hs_all = sorted(L["xyxy"][3] - L["xyxy"][1] for L in L_all if L["text"])
    med_h = float(hs_all[len(hs_all) // 2]) if hs_all else 15.0
    for b in blocks:
        for L in list(b["lines"]):
            x0, y0, x1, y1 = L["xyxy"]
            if y1 - y0 >= max(4.0, 0.3 * med_h) or not L["text"]:
                continue
            cy = (y0 + y1) / 2.0
            others = [o for o in b["lines"] if o is not L
                      and o["xyxy"][1] - 0.5 * med_h <= cy <= o["xyxy"][3] + 0.5 * med_h]
            if others:
                o = min(others, key=lambda o: abs((o["xyxy"][1] + o["xyxy"][3]) / 2.0 - cy))
                want = _norm_text(strip_heading_prefix(b["det"].get("text") or ""))
                a1 = (L["text"] + " " + o["text"]).strip()
                a2 = (o["text"] + " " + L["text"]).strip()
                if a1 in want and a2 not in want:
                    o["text"] = a1
                elif a2 in want:
                    o["text"] = a2
                else:
                    o["text"] = a1 if x0 < o["xyxy"][0] else a2
                nx0, nx1 = min(o["xyxy"][0], x0), max(o["xyxy"][2], x1)
                if (nx0, nx1) != (o["xyxy"][0], o["xyxy"][2]):
                    o["xyxy"] = [nx0, o["xyxy"][1], nx1, o["xyxy"][3]]
                    o["bbox"] = (nx0, o["xyxy"][1], nx1 - nx0, o["xyxy"][3] - o["xyxy"][1])
                    o["quad"] = _rect_quad(*o["xyxy"])
                b["lines"].remove(L)
                warns.append(f"line '{L['text'][:25]}' was {y1 - y0:g} px tall - joined to "
                             f"'{o['text'][:30]}'")
            elif gray is not None:
                ex = [max(0, x0 - med_h), max(0, cy - 0.6 * med_h), x1 + med_h, cy + 0.6 * med_h]
                tt = tighten_to_ink(gray, (ex[0], ex[1], ex[2] - ex[0], ex[3] - ex[1]))
                if tt and tt[0][3] - tt[0][1] > y1 - y0:
                    L["xyxy"] = list(tt[0])
                    L["bbox"] = (tt[0][0], tt[0][1], tt[0][2] - tt[0][0], tt[0][3] - tt[0][1])
                    L["quad"] = tt[1] or _rect_quad(*tt[0])
                    warns.append(f"line '{L['text'][:25]}' was {y1 - y0:g} px tall - refitted "
                                 f"to {_xyxy_ints(L['xyxy'])}")
''')

# ---------------------------------------------------------------- 12 long table rules kept
patch("12a _rule_components long_ok parameter",
r'''def _rule_components(bw, orient, minlen, max_thick):''',
r'''def _rule_components(bw, orient, minlen, max_thick, long_ok=None):''')

patch("12b long rules are kept even when text touches them",
r'''        if near > 0.3:''',
r'''        if near > 0.3 and (long_ok is None or length < long_ok):''')

patch("12c detect_ruling_lines keeps long rules",
r'''        for lx, ly, lw, lh, a in _rule_components(bw, orient, minlen, max_thick):''',
r'''        long_ok = 0.5 * (cw if orient == "h" else ch)
        for lx, ly, lw, lh, a in _rule_components(bw, orient, minlen, max_thick, long_ok):''')

# ---------------------------------------------------------------- 13 snap rows to the rule between them
patch("13 table row/column edge = the rule between the two rows",
r'''            b[i] = (ext[i - 1][1] + ext[i][0]) / 2.0''',
r'''            mid = (ext[i - 1][1] + ext[i][0]) / 2.0
            between = [s for s in (snaps or []) if ext[i - 1][1] - 2 <= s <= ext[i][0] + 2]
            b[i] = min(between, key=lambda s: abs(s - mid)) if between else mid''')

# ---------------------------------------------------------------- 14 specks outside the text band
patch("14 tighten_to_ink drops specks above/below the text",
r'''    if not keep.any():''',
r'''    if keep.any():
        hs_ = [int(st[i][3]) for i in range(1, n) if keep[i]]
        big_h = max(hs_)
        main = [i for i in range(1, n) if keep[i] and st[i][3] >= 0.35 * big_h]
        if main:
            by0_ = min(int(st[i][1]) for i in main)
            by1_ = max(int(st[i][1] + st[i][3]) for i in main)
            gap_ = 0.25 * (by1_ - by0_)
            for i in range(1, n):
                if keep[i] and st[i][3] < 0.35 * big_h and (
                        st[i][1] + st[i][3] < by0_ - gap_ or st[i][1] > by1_ + gap_):
                    keep[i] = False
    if not keep.any():''')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    inplace = "--inplace" in sys.argv[2:]
    with open(src, "r", encoding="utf-8", newline="") as f:
        code = f.read()
    crlf = "\r\n" in code
    code = code.replace("\r\n", "\n")

    failed = []
    for name, old, new in PATCHES:
        if new in code and old not in code:
            print(f"  already applied: {name}")
            continue
        n = code.count(old)
        if n != 1:
            failed.append(f"{name}  (anchor found {n} times, expected 1)")
            continue
        code = code.replace(old, new)
        print(f"  ok: {name}")
    if failed:
        print("\nNOT patched - nothing was written. These anchors did not match:")
        for f_ in failed:
            print("  - " + f_)
        sys.exit(2)

    if crlf:
        code = code.replace("\n", "\r\n")
    if inplace:
        shutil.copy2(src, src + ".bak")
        out = src
    else:
        stem, ext = os.path.splitext(src)
        out = stem + "_patched" + ext
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(code)
    py_compile.compile(out, doraise=True)
    print(f"\nWrote {out} (syntax check passed).")


if __name__ == "__main__":
    main()
