#!/usr/bin/env python3
"""
Side-by-side viewer: annotated page image (left) + extracted document rendering (right).

    python visualize_ocr.py <ocr_output_folder> --images <images_folder>
    python visualize_ocr.py <ocr_output_folder> --images <images_folder> --n 20 --seed 7
    python visualize_ocr.py <ocr_output_folder> --images <images_folder> --page 004

Produces <ocr_output_folder>/_viewer/index.html — open it in a browser.
"""
import argparse
import html as html_mod
import json
import re
from pathlib import Path

import cv2
import numpy as np

# ---- colours (BGR) ---------------------------------------------------------
BLOCK_COLORS = {
    "text": (60, 180, 60), "heading": (200, 0, 200), "list": (0, 160, 255),
    "table": (0, 0, 230), "figure": (0, 150, 0), "caption": (128, 128, 0),
    "header": (120, 60, 0), "footer": (120, 120, 0), "stamp": (60, 60, 200),
    "signature": (90, 90, 90), "form": (0, 120, 160), "handwriting": (160, 0, 100),
}
LINE_OK = (255, 80, 0)
LINE_LOW = (0, 0, 255)
LINE_EMPTY = (0, 165, 255)
CELL_COLOR = (0, 165, 255)


# ---- Unicode-capable text drawing -----------------------------------------
class TextRenderer:
    def __init__(self, size: int = 14):
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.pil = (Image, ImageDraw)
            for cand in (
                "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ):
                if Path(cand).exists():
                    self.font = ImageFont.truetype(cand, size)
                    break
            else:
                self.font = ImageFont.load_default()
        except ImportError:
            self.pil = None
            self.font = None

    def draw(self, img, text, x, y, color):
        if not text:
            return
        if self.pil is None:
            cv2.putText(img, text.encode("ascii", "replace").decode(),
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            return
        Image, ImageDraw = self.pil
        pil = Image.fromarray(img[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        rgb = (color[2], color[1], color[0])
        try:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            th = bbox[3] - bbox[1]
        except Exception:
            th = 12
        draw.text((x, max(0, y - th)), text, fill=rgb, font=self.font)
        img[:, :, :] = np.asarray(pil)[:, :, ::-1]


# ---- annotate one image ----------------------------------------------------
def annotate(img, r, renderer):
    vis = img.copy()
    for b in r.get("blocks", []):
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        if b.get("orphan"):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (140, 140, 140), 1)
            continue
        color = BLOCK_COLORS.get(b.get("type", ""), (180, 180, 180))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        tag = f"{b.get('type', '?')} {b.get('score', 0):.2f}"
        ty = y1 - 4 if y1 > 20 else y1 + 16
        renderer.draw(vis, tag, x1 + 2, ty, color)
    for l in r.get("lines", []):
        q = np.asarray(l["quad"], np.float32).astype(np.int32)
        txt = l.get("text", "")
        conf = l.get("conf", 0.0)
        color = LINE_EMPTY if not txt.strip() else (LINE_LOW if conf < 0.5 else LINE_OK)
        cv2.polylines(vis, [q], True, color, 1, cv2.LINE_AA)
        if txt:
            top = int(q[:, 1].min())
            left = int(q[:, 0].min())
            ty = top - 3 if top > 15 else top + 14
            renderer.draw(vis, txt, left, ty, color)
    for t in r.get("tables", []):
        for c in t.get("cells", []):
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), CELL_COLOR, 1)
    return vis


# ---- extracted document -> HTML --------------------------------------------
def _esc(s):
    return html_mod.escape(html_mod.unescape(s or ""))


def _cells_to_html(cells):
    """Reconstruct a real HTML table from OCR cells with row/col/rowspan/colspan."""
    if not cells:
        return "<p class='empty'><em>(table with no cells)</em></p>"
    cells = sorted(cells, key=lambda c: (c.get("row", 0), c.get("col", 0)))
    # skip cells covered by a previous cell's span
    occupied, kept = set(), []
    for c in cells:
        r, col = c.get("row", 0), c.get("col", 0)
        if (r, col) in occupied:
            continue
        rs, cs = c.get("rowspan", 1), c.get("colspan", 1)
        for dr in range(rs):
            for dc in range(cs):
                occupied.add((r + dr, col + dc))
        kept.append(c)
    rows = {}
    for c in kept:
        rows.setdefault(c.get("row", 0), []).append(c)
    out = ['<table class="ocr-table">']
    for r in sorted(rows):
        out.append("<tr>")
        for c in sorted(rows[r], key=lambda x: x.get("col", 0)):
            rs, cs = c.get("rowspan", 1), c.get("colspan", 1)
            attrs = ""
            if rs > 1: attrs += f' rowspan="{rs}"'
            if cs > 1: attrs += f' colspan="{cs}"'
            out.append(f"<td{attrs}>{_esc(c.get('text', ''))}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _block_text_html(block):
    """Render a single non-table block: heading / list / paragraph."""
    t = (block.get("text") or "").strip()
    if not t:
        return ""
    btype = block.get("type", "text")
    # multi-line: keep the breaks
    if btype == "heading":
        return f"<h3>{_esc(t)}</h3>"
    if btype == "list":
        items = [ln.strip() for ln in t.split("\n") if ln.strip()]
        return "<ul>" + "".join(f"<li>{_esc(x)}</li>" for x in items) + "</ul>"
    # paragraph / footer / header / caption / etc.
    return "<p>" + "<br>".join(_esc(ln) for ln in t.split("\n")) + "</p>"


def render_document(r):
    """Render the whole page's extracted content in reading order."""
    parts = []
    for b in r.get("blocks", []):
        if b.get("orphan"):
            continue
        if "table" in b and b["table"] is not None:
            t = b["table"]
            parts.append(f'<div class="table-wrap">{_cells_to_html(t.get("cells", []))}</div>')
            continue
        btype = b.get("type", "text")
        if btype == "figure":
            parts.append('<div class="fig">[figure]</div>')
            continue
        h = _block_text_html(b)
        if h:
            parts.append(h)
    if not parts:
        return "<p class='empty'><em>(no content extracted)</em></p>"
    return "\n".join(parts)


# ---- main -----------------------------------------------------------------
def load_pages(folder: Path, images_dir: Path):
    pages = []
    for jf in sorted(folder.glob("*.json")):
        if jf.name == "summary.json":
            continue
        try:
            r = json.load(open(jf, encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {jf.name}: {e}")
            continue
        stem = jf.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        pages.append((stem, jf, r, img_path))
    return pages


PAGE_CSS = """
:root { --bg:#f6f5f2; --card:#fff; --ink:#222; --muted:#666; --line:#ddd; --ok:#2e7d32; --warn:#c0392b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161616; --card:#222; --ink:#eee; --muted:#aaa; --line:#444; --ok:#66bb6a; --warn:#ef5350; }
}
body { margin:0; background:var(--bg); color:var(--ink);
       font-family:"Noto Sans Devanagari","Mukta",system-ui,sans-serif; font-size:15px; line-height:1.55; }
header { padding:16px 20px; }
header h1 { margin:0; font-size:20px; }
header p { color:var(--muted); margin:4px 0 0; font-size:14px; }
.legend { margin-top:8px; }
.legend span { display:inline-block; margin-right:14px; font-size:13px; }
.legend i { display:inline-block; width:12px; height:12px; margin-right:4px; vertical-align:-1px;
            border:1px solid rgba(0,0,0,.2); border-radius:2px; }
.page { background:var(--card); margin:16px; padding:16px; border-radius:8px; border:1px solid var(--line); }
.page h2 { margin:0 0 6px; font-size:16px; }
.meta { color:var(--muted); font-size:13px; margin-bottom:12px; }
.grid { display:grid; grid-template-columns: minmax(0,1.1fr) minmax(0,1fr); gap:16px; }
@media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
.grid img { width:100%; border:1px solid var(--line); border-radius:4px; display:block; }
.doc { overflow:auto; max-height:1400px; padding:8px 12px; border:1px solid var(--line); border-radius:4px;
       background:rgba(127,127,127,.03); }
.doc h3 { margin:14px 0 6px; font-size:17px; }
.doc p  { margin:6px 0; }
.doc ul { margin:6px 0; padding-left:22px; }
.doc .fig { border:1px dashed var(--muted); padding:16px; text-align:center; color:var(--muted); margin:8px 0; }
.doc .empty { color:var(--muted); }
table.ocr-table { border-collapse:collapse; margin:10px 0; width:100%; font-size:14px; }
table.ocr-table th, table.ocr-table td { border:1px solid var(--muted); padding:4px 8px; vertical-align:top; }
table.ocr-table td:empty::after { content:"\\00a0"; }
.table-wrap { overflow-x:auto; }
details { margin-top:10px; }
details pre { white-space:pre-wrap; font-size:12px; max-height:400px; overflow:auto;
  background:rgba(127,127,127,.06); padding:10px; border-radius:4px; }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="OCR output folder (with *.json per page)")
    ap.add_argument("--images", required=True, help="folder holding the original page images")
    ap.add_argument("--page", default=None, help="only this page stem (substring match)")
    ap.add_argument("--n", type=int, default=0, help="limit to first N pages (0 = all)")
    ap.add_argument("--font-size", type=int, default=14)
    a = ap.parse_args()

    folder = Path(a.folder).expanduser().resolve()
    images_dir = Path(a.images).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    if not images_dir.is_dir():
        raise SystemExit(f"not a folder: {images_dir}")

    pages = load_pages(folder, images_dir)
    if a.page:
        pages = [p for p in pages if a.page in p[0]]
    if a.n:
        pages = pages[:a.n]
    if not pages:
        raise SystemExit("no pages matched")

    out = folder / "_viewer"
    out.mkdir(exist_ok=True)
    renderer = TextRenderer(a.font_size)

    sections = []
    for stem, jf, r, img_path in pages:
        # render annotated image
        ann_name = f"{stem}.annotated.png"
        if img_path and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(out / ann_name), annotate(img, r, renderer))
            else:
                ann_name = None
        else:
            ann_name = None

        # meta line
        n_lines = len(r.get("lines", []))
        n_tables = len(r.get("tables", []))
        n_cells = sum(len(t.get("cells", [])) for t in r.get("tables", []))
        confs = [l.get("conf", 0) for l in r.get("lines", []) if l.get("text", "").strip()]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        min_conf = min(confs) if confs else 0.0
        types = {}
        for b in r.get("blocks", []):
            if not b.get("orphan"):
                types[b["type"]] = types.get(b["type"], 0) + 1
        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(types.items())) or "none"

        meta = (f"{n_lines} lines · {n_tables} tables ({n_cells} cells) · "
                f"conf min/mean {min_conf:.2f}/{mean_conf:.2f} · blocks: {type_str}")

        doc_html = render_document(r)
        img_html = (f'<a href="{ann_name}" target="_blank"><img src="{ann_name}" alt="{stem}"></a>'
                    if ann_name else '<div class="empty">(image not found)</div>')

        raw_text = r.get("text", "").strip() or "(empty)"
        raw_json = json.dumps({k: v for k, v in r.items() if k not in ("text",)},
                              ensure_ascii=False, indent=2)

        sections.append(f"""
<section class="page">
  <h2>{html_mod.escape(stem)}</h2>
  <div class="meta">{html_mod.escape(meta)}</div>
  <div class="grid">
    {img_html}
    <div class="doc">{doc_html}</div>
  </div>
  <details><summary>Serialized page text (raw)</summary><pre>{html_mod.escape(raw_text)}</pre></details>
  <details><summary>Full JSON</summary><pre>{html_mod.escape(raw_json)}</pre></details>
</section>""")

    legend = "".join(f"<span><i style='background:rgb({c[2]},{c[1]},{c[0]})'></i>{k}</span>"
                     for k, c in BLOCK_COLORS.items())
    legend += "<span><i style='background:rgb(255,80,0)'></i>line conf ≥ 0.5</span>"
    legend += "<span><i style='background:rgb(0,0,255)'></i>line conf &lt; 0.5</span>"
    legend += "<span><i style='background:rgb(0,165,255)'></i>empty / table cell</span>"
    legend += "<span><i style='background:rgb(140,140,140)'></i>orphan fallback</span>"

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR viewer — {html_mod.escape(folder.name)}</title><style>{PAGE_CSS}</style></head><body>
<header>
  <h1>OCR viewer</h1>
  <p>{len(sections)} pages · {html_mod.escape(str(folder))}</p>
  <div class="legend">{legend}</div>
</header>
{''.join(sections)}
</body></html>"""

    index = out / "index.html"
    index.write_text(doc, encoding="utf-8")
    print(f"[done] {len(sections)} pages -> {index}")
    print(f"       open: file://{index}")


if __name__ == "__main__":
    main()