#!/usr/bin/env python3
"""
Convert minimal manual annotations into the v2 training format
(v, image, layout, blocks, lines, tables, ...).

The annotator only labels what a human must decide. Everything else is
derived here: bbox, ids, blocks, line_in_block, list items, list_level,
item_index, table_index, cell_id, num_rows/num_cols, html, page text, font_size.

------------------------------------------------------------------------
MINIMAL INPUT (one JSON per page) -- what your annotation tool must export
------------------------------------------------------------------------
{
  "image": "images/page_0000010.jpg", "width": 1145, "height": 1600,
  "document_id": "doc_0000010", "page_number": 1, "page_count": 1,
  "language": "en", "digit_style": "latin", "doc_type": "report_page",

  "regions": [                       # one per layout element
    {"id": 0, "type": "header",  "reading_order": 0, "quad": [[x,y]x4]},
    {"id": 1, "type": "heading", "reading_order": 1, "level": 1, "quad": ...},
    {"id": 4, "type": "list",    "reading_order": 4, "quad": ...},
    {"id": 6, "type": "table",   "reading_order": 6, "quad": ...},
    {"id": 15, "type": "handwriting", "reading_order": null,
     "subtype": "note", "quad": ...}      # parent_id / parent_item optional
  ],                                      # any type label is accepted

  "lines": [                         # one per text line
    {"quad": ..., "text": "4. Recommendations"},   # region auto-detected
    {"quad": ..., "text": "OK", "region_id": 16}   # or set explicitly
  ],                                 # optional per line: handwritten,
                                     # font_size, list_level, is_heading
                                     # (font_size can also be set per region)

  "tables": [
    {"region_id": 6, "header_rows": 1, "ruling_style": "full",
     "cells": [{"row": 0, "col": 0, "rowspan": 1, "colspan": 1,
                "quad": ..., "text": "S.N."}]}   # text optional: taken
  ],                                             # from lines in the cell

  "extra_regions": [], "quality": {}, "annotation": {}   # passed through
}

Quads: 4 points, clockwise from the top-left of the text
(TL, TR, BR, BL), in pixels. Flat [x1,y1,...,x4,y4] is also accepted.

Usage:
  python to_training_format.py annotations/ -o converted/
  python to_training_format.py page.json -o converted/
"""
import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

HEADING_TYPES = {"heading", "title"}
HANDWRITING_TYPES = {"handwriting"}
MARKER_RE = re.compile(
    r"^\s*(\(?\d+[.)]|\(?[a-zA-Z][.)]|\(?[ivxlcdm]+[.)]|[•\-–—*·▪◦])\s+"
)


# ---------------------------------------------------------------- geometry
def norm_quad(q):
    if len(q) == 8 and not isinstance(q[0], (list, tuple)):
        q = [q[i:i + 2] for i in range(0, 8, 2)]
    if len(q) != 4:
        raise ValueError(f"quad needs 4 points, got {q}")
    return [[round(float(x), 1), round(float(y), 1)] for x, y in q]


def signed_area(q):
    return sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1]
               for i in range(4)) / 2


def fix_winding(q, label, warn):
    # In image coordinates (y down) clockwise gives a positive area.
    if signed_area(q) < 0:
        warn.append(f"{label}: corners were counter-clockwise, reordered")
        q = [q[0], q[3], q[2], q[1]]
    return q


def bbox(q):
    xs = [p[0] for p in q]
    ys = [p[1] for p in q]
    return [math.floor(min(xs)), math.floor(min(ys)),
            math.ceil(max(xs)), math.ceil(max(ys))]


def center(q):
    return (sum(p[0] for p in q) / 4, sum(p[1] for p in q) / 4)


def axes(q):
    """Unit vectors along the text direction (right) and across it (down)."""
    (x0, y0), (x1, y1), _, (x3, y3) = q
    rx, ry, dx, dy = x1 - x0, y1 - y0, x3 - x0, y3 - y0
    rn = math.hypot(rx, ry) or 1.0
    dn = math.hypot(dx, dy) or 1.0
    return (rx / rn, ry / rn), (dx / dn, dy / dn)


def quad_height(q):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = q
    return (math.hypot(x3 - x0, y3 - y0) + math.hypot(x2 - x1, y2 - y1)) / 2


def point_in_quad(pt, q):
    x, y = pt
    inside = False
    for i in range(4):
        x1, y1 = q[i]
        x2, y2 = q[(i + 1) % 4]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def dot(p, v):
    return p[0] * v[0] + p[1] * v[1]


# ------------------------------------------------------------------ helpers
def sort_lines(region_quad, lines):
    """Reading order inside a block, robust to skew/rotation."""
    if not lines:
        return lines
    right, down = axes(region_quad)
    keyed = sorted(lines, key=lambda l: dot(center(l["quad"]), down))
    rows, cur = [], [keyed[0]]
    for l in keyed[1:]:
        prev = cur[-1]
        gap = dot(center(l["quad"]), down) - dot(center(prev["quad"]), down)
        if gap < 0.5 * min(quad_height(l["quad"]), quad_height(prev["quad"])):
            cur.append(l)
        else:
            rows.append(cur)
            cur = [l]
    rows.append(cur)
    out = []
    for row in rows:
        out += sorted(row, key=lambda l: dot(center(l["quad"]), right))
    return out


def join_text(lines):
    return " ".join(l["text"].strip() for l in lines if l["text"].strip())


def build_list(region, lines, children, indent_tol):
    right, down = axes(region["quad"])
    origin = region["quad"][0]

    def indent(l):
        p = l["quad"][0]
        return dot((p[0] - origin[0], p[1] - origin[1]), right)

    marker_lines = [l for l in lines if MARKER_RE.match(l["text"])]
    levels = []
    for v in sorted(indent(l) for l in marker_lines):
        if not levels or v - levels[-1] > indent_tol:
            levels.append(v)

    def level_of(l):
        if l.get("list_level") is not None:
            return l["list_level"]
        if not levels:
            return 0
        v = indent(l)
        return min(range(len(levels)), key=lambda i: abs(levels[i] - v))

    items, item_pos = [], []
    for l in lines:
        m = MARKER_RE.match(l["text"])
        if m or not items:
            items.append({
                "marker": m.group(1) if m else "",
                "level": level_of(l),
                "text": l["text"][m.end():].strip() if m else l["text"].strip(),
            })
            item_pos.append(dot(center(l["quad"]), down))
        else:  # continuation line of the previous item
            items[-1]["text"] += " " + l["text"].strip()
        l["list_level"] = items[-1]["level"]
        l["item_index"] = len(items) - 1

    # attach handwritten notes etc. to the item they sit under
    for ch in children:
        if ch.get("parent_item") is None:
            cpos = dot(center(ch["quad"]), down)
            idx = 0
            for i, pos in enumerate(item_pos):
                if pos <= cpos:
                    idx = i
            ch["parent_item"] = idx
        if items:
            items[ch["parent_item"]].setdefault("children", []).append(ch["id"])

    text_lines = []
    for i, it in enumerate(items):
        pad = "  " * it["level"]
        sep = " " if it["marker"] else ""
        text_lines.append(f"{pad}{it['marker']}{sep}{it['text']}")
        for cid in it.get("children", []):
            ch = next(c for c in children if c["id"] == cid)
            text_lines.append("  " * (it["level"] + 1) + ch["text"])
    return items, "\n".join(text_lines)


def table_html(cells, num_rows):
    rows = [[] for _ in range(num_rows)]
    for c in cells:
        rows[c["row"]].append(c)
    out = "<table>"
    for row in rows:
        out += "<tr>"
        for c in sorted(row, key=lambda c: c["col"]):
            tag = "th" if c["is_header"] else "td"
            attrs = ""
            if c["rowspan"] > 1:
                attrs += f' rowspan="{c["rowspan"]}"'
            if c["colspan"] > 1:
                attrs += f' colspan="{c["colspan"]}"'
            out += f"<{tag}{attrs}>{html.escape(c['text'], quote=False)}</{tag}>"
        out += "</tr>"
    return out + "</table>"


# --------------------------------------------------------------- converter
def convert(a, font_scale=1.3, indent_tol=12.0):
    warn = []

    # ---- regions: order, renumber, normalise
    regions = []
    for r in a["regions"]:
        r = dict(r)
        r["quad"] = fix_winding(norm_quad(r["quad"]), f"region {r['id']}", warn)
        regions.append(r)
    ordered = sorted([r for r in regions if r.get("reading_order") is not None],
                     key=lambda r: r["reading_order"])
    floating = [r for r in regions if r.get("reading_order") is None]
    regions = ordered + floating
    idmap = {r["id"]: i for i, r in enumerate(regions)}
    if len(idmap) != len(regions):
        raise ValueError("duplicate region ids")
    for i, r in enumerate(regions):
        r["id"] = i
        if r in ordered:
            r["reading_order"] = ordered.index(r)
        if r.get("parent_id") is not None:
            r["parent_id"] = idmap[r["parent_id"]]
    by_id = {r["id"]: r for r in regions}

    def area(q):
        return abs(signed_area(q))

    # ---- lines: assign to regions
    lines = []
    for i, l in enumerate(a.get("lines", [])):
        l = dict(l)
        l["quad"] = fix_winding(norm_quad(l["quad"]), f"line {i}", warn)
        if l.get("region_id") is not None:
            l["block_id"] = idmap[l["region_id"]]
        else:
            c = center(l["quad"])
            hits = [r for r in regions if point_in_quad(c, r["quad"])]
            if not hits:
                warn.append(f"line {i} '{l['text'][:30]}' is outside every region, dropped")
                continue
            l["block_id"] = min(hits, key=lambda r: area(r["quad"]))["id"]
        lines.append(l)
    by_block = {r["id"]: [] for r in regions}
    for l in lines:
        by_block[l["block_id"]].append(l)

    # ---- handwriting regions: flags and parent detection
    for r in regions:
        if r["type"] in HANDWRITING_TYPES:
            r["handwritten"] = True
            for l in by_block[r["id"]]:
                l["handwritten"] = True
            if "parent_id" not in r:
                c = center(r["quad"])
                parents = [p for p in regions if p["type"] == "list"
                           and p["id"] != r["id"] and point_in_quad(c, p["quad"])]
                if parents:
                    r["parent_id"] = parents[0]["id"]

    # ---- tables
    tables_in = {idmap[t["region_id"]]: t for t in a.get("tables", [])}
    out_tables = []
    for r in regions:
        if r["type"] != "table":
            continue
        t = tables_in.get(r["id"])
        if t is None:
            warn.append(f"table region {r['id']} has no cell annotation")
            continue
        ti = len(out_tables)
        r["table_index"] = ti
        src = sorted(t["cells"], key=lambda c: (c["row"], c["col"]))
        nr = max(c["row"] + c.get("rowspan", 1) for c in src)
        nc = max(c["col"] + c.get("colspan", 1) for c in src)
        hr = t.get("header_rows", 1)
        grid = [[None] * nc for _ in range(nr)]
        cells = []
        for i, c in enumerate(src):
            rs, cs = c.get("rowspan", 1), c.get("colspan", 1)
            for rr in range(c["row"], c["row"] + rs):
                for cc in range(c["col"], c["col"] + cs):
                    if grid[rr][cc] is not None:
                        warn.append(f"table {ti}: cells overlap at row {rr}, col {cc}")
                    grid[rr][cc] = i
            q = fix_winding(norm_quad(c["quad"]), f"table {ti} cell {i}", warn)
            cells.append({
                "id": i, "row": c["row"], "col": c["col"],
                "rowspan": rs, "colspan": cs, "text": c.get("text"),
                "is_header": c.get("is_header", c["row"] < hr),
                "handwritten": c.get("handwritten", False), "empty": None,
                "bbox": bbox(q), "quad": q,
            })
        for rr in range(nr):
            for cc in range(nc):
                if grid[rr][cc] is None:
                    warn.append(f"table {ti}: no cell covers row {rr}, col {cc}")

        # lines -> cells
        cell_lines = {c["id"]: [] for c in cells}
        for l in by_block[r["id"]]:
            l["table_index"] = ti
            hit = [c for c in cells if point_in_quad(center(l["quad"]), c["quad"])]
            if hit:
                l["cell_id"] = hit[0]["id"]
                cell_lines[hit[0]["id"]].append(l)
            else:
                l["cell_id"] = None
                warn.append(f"table {ti}: line '{l['text'][:30]}' is in no cell")
        for c in cells:
            cl = sort_lines(c["quad"], cell_lines[c["id"]])
            if c["text"] is None:
                c["text"] = join_text(cl)
            c["empty"] = c["text"].strip() == ""
            if any(l.get("handwritten") for l in cl):
                c["handwritten"] = True
        by_block[r["id"]] = sorted(
            by_block[r["id"]],
            key=lambda l: (l["cell_id"] if l["cell_id"] is not None else 1e9,
                           dot(center(l["quad"]), axes(r["quad"])[1])))

        out_tables.append({
            "bbox": bbox(r["quad"]), "quad": r["quad"],
            "num_rows": nr, "num_cols": nc, "header_rows": hr,
            "ruling_style": t.get("ruling_style"),
            "theme": t.get("theme"), "col_types": t.get("col_types"),
            "html": table_html(cells, nr),
            "reading_order": [c["id"] for c in cells],
            "cells": cells,
            "ruling_lines": t.get("ruling_lines", []),
        })
        r["text"] = out_tables[-1]["html"]

    # ---- line order inside non-table blocks
    for r in regions:
        if r["type"] != "table":
            by_block[r["id"]] = sort_lines(r["quad"], by_block[r["id"]])

    # ---- region text (handwriting first, lists use child notes)
    for r in regions:
        if r["type"] != "table" and "text" not in r:
            r["text"] = join_text(by_block[r["id"]])
    for r in regions:
        t = r["type"]
        if t == "list":
            children = [c for c in regions if c.get("parent_id") == r["id"]]
            r["items"], built = build_list(r, by_block[r["id"]], children, indent_tol)
            r["text"] = built
            r.setdefault("list_style", a.get("digit_style", "latin"))
        elif t in HEADING_TYPES:
            r.setdefault("level", 1)
            if not r["text"].startswith("#"):
                r["text"] = "#" * r["level"] + " " + r["text"]

    # ---- assemble layout / blocks / lines
    layout, blocks, out_lines = [], [], []
    for r in regions:
        e = {"id": r["id"], "type": r["type"], "bbox": bbox(r["quad"]),
             "reading_order": r.get("reading_order"), "text": r["text"]}
        if r["type"] in HEADING_TYPES:
            e["level"] = r["level"]
        if r["type"] == "list":
            e["list_style"] = r["list_style"]
            e["items"] = r["items"]
        if r["type"] == "table" and "table_index" in r:
            e["table_index"] = r["table_index"]
        if r["type"] in HANDWRITING_TYPES:
            e["subtype"] = r.get("subtype")
            e["handwritten"] = True
            if r.get("parent_id") is not None:
                e["parent_id"] = r["parent_id"]
                e["parent_item"] = r.get("parent_item")
        if r.get("fields") is not None:
            e["fields"] = r["fields"]
        e["quad"] = r["quad"]
        layout.append(e)
        blocks.append({"type": r["type"], "bbox": bbox(r["quad"]),
                       "block_id": r["id"], "font_id": None, "quad": r["quad"]})

        # one size per block (median line height), like the training data
        hs = sorted(quad_height(l["quad"]) for l in by_block[r["id"]])
        block_fs = float(round(hs[len(hs) // 2] * font_scale)) if hs else None
        for k, l in enumerate(by_block[r["id"]]):
            fs = l.get("font_size")
            if fs is None:
                fs = r.get("font_size", block_fs)
            o = {"text": l["text"], "bbox": bbox(l["quad"]),
                 "block_id": r["id"], "line_in_block": k,
                 "is_heading": l.get("is_heading", r["type"] in HEADING_TYPES),
                 "font_size": float(fs), "font_id": None}
            if "item_index" in l:
                o["list_level"] = l["list_level"]
                o["item_index"] = l["item_index"]
            if "table_index" in l:
                o["table_index"] = l["table_index"]
                o["cell_id"] = l["cell_id"]
            if l.get("handwritten"):
                o["handwritten"] = True
            o["quad"] = l["quad"]
            out_lines.append(o)

    page_text = "\n\n".join(r["text"] for r in ordered if r["text"])

    out = {
        "v": 2,
        "image": a["image"], "width": a["width"], "height": a["height"],
        "document_id": a["document_id"],
        "page_number": a.get("page_number", 1),
        "page_count": a.get("page_count", 1),
        "language": a.get("language"), "digit_style": a.get("digit_style"),
        "doc_type": a.get("doc_type"),
        "text": page_text,
        "fonts": [], "font_ids": [],
        "layout": layout, "blocks": blocks, "lines": out_lines,
        "tables": out_tables,
        "extra_regions": a.get("extra_regions", []),
        "quality": a.get("quality", {}),
        "annotation": a.get("annotation", {}),
    }
    return out, warn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="annotation JSON file or folder of them")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder, or an output .json path for a single input file")
    ap.add_argument("--font-scale", type=float, default=1.3,
                    help="font_size = line height * this, when not annotated")
    ap.add_argument("--indent-tol", type=float, default=12.0,
                    help="px tolerance for grouping list indent levels")
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(src.glob("*.json")) if src.is_dir() else [src]
    out_arg = Path(args.out)
    single_file_out = (not src.is_dir()) and out_arg.suffix.lower() == ".json"
    if single_file_out:
        out_arg.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_arg.mkdir(parents=True, exist_ok=True)

    ok = failed = total_warn = 0
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        missing = [k for k in ("image", "width", "height", "document_id", "regions")
                   if k not in data]
        if missing:
            print(f"[ERROR] {f.name}: not in the minimal annotation format "
                  f"(missing {', '.join(missing)}). Top-level keys found: "
                  f"{', '.join(list(data)[:15]) if isinstance(data, dict) else type(data).__name__}",
                  file=sys.stderr)
            failed += 1
            continue
        try:
            out, warn = convert(data, args.font_scale, args.indent_tol)
        except Exception as e:
            print(f"[ERROR] {f.name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue
        dest = out_arg if single_file_out else out_arg / f.name
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        for w in warn:
            print(f"[WARN] {f.name}: {w}", file=sys.stderr)
        total_warn += len(warn)
        ok += 1
    print(f"converted {ok} file(s), failed {failed}, {total_warn} warning(s) -> {out_arg}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()