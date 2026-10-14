#!/usr/bin/env python3
"""
Draw bounding boxes on each page image and write the extracted text of each region.

Outputs, per page, into <out>/_annotated/:
  <stem>.annotated.png   — image with coloured boxes and text labels drawn on
  <stem>.lines.txt       — one line per detected text line: "x1 y1 x2 y2  conf  text"
  <stem>.blocks.txt      — one line per layout block:    "type  score  x1 y1 x2 y2  text"
  <stem>.tables.txt      — one section per table: header row + one row per cell

Plus <out>/_annotated/all_lines.tsv   — every line from every page in one file
      <out>/_annotated/all_blocks.tsv — every layout block from every page in one file

Usage:
    python annotate.py <output_folder> --images <images_folder>
    python annotate.py <output_folder> --images <images_folder> --font-scale 0.8 --thickness 2
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# Colour per layout class (BGR). Lines use their own scheme.
BLOCK_COLORS = {
    "text": (60, 180, 60), "heading": (0, 128, 255), "list": (255, 128, 0),
    "table": (0, 0, 255), "figure": (128, 0, 128), "caption": (128, 128, 0),
    "header": (200, 200, 0), "footer": (200, 0, 200), "stamp": (0, 255, 255),
    "signature": (255, 0, 128), "form": (150, 150, 150), "handwriting": (60, 60, 60),
}
LINE_OK_COLOR = (255, 80, 0)       # blue-ish for confident lines
LINE_LOW_COLOR = (0, 0, 255)       # red for low-conf lines
LINE_EMPTY_COLOR = (0, 165, 255)   # orange for lines with no decoded text
TABLE_CELL_COLOR = (0, 165, 255)


# ---------------------------------------------------------------------------
# Text rendering with a Unicode-capable font fallback
# ---------------------------------------------------------------------------
class TextRenderer:
    """
    cv2.putText only draws ASCII (its Hershey fonts are Latin-1). For Nepali we use PIL when available.
    Falls back to ASCII-only rendering of the ASCII portions if PIL is not installed.
    """
    def __init__(self, font_path: str = None, size: int = 14):
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.pil = (Image, ImageDraw, ImageFont)
            if font_path:
                self.font = ImageFont.truetype(font_path, size)
            else:
                # try a few common Devanagari fonts, else use PIL default
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

    def draw(self, img_bgr: np.ndarray, text: str, x: int, y: int, color_bgr: Tuple[int, int, int]):
        """Draw `text` with its baseline-left at (x, y). Falls back to cv2 ASCII if PIL is missing."""
        if not text:
            return
        if self.pil is None:
            cv2.putText(img_bgr, text.encode("ascii", "replace").decode(),
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)
            return
        Image, ImageDraw, _ = self.pil
        pil = Image.fromarray(img_bgr[:, :, ::-1])  # BGR -> RGB
        draw = ImageDraw.Draw(pil)
        rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        # PIL draws with the y as the TOP of the text, not the baseline; adjust by the font's ascent
        try:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            th = bbox[3] - bbox[1]
        except Exception:
            th = self.font.size if hasattr(self.font, "size") else 12
        draw.text((x, max(0, y - th)), text, fill=rgb, font=self.font)
        img_bgr[:, :, :] = np.asarray(pil)[:, :, ::-1]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_annotation(img: np.ndarray, r: dict, renderer: TextRenderer,
                    draw_lines: bool = True, draw_blocks: bool = True,
                    draw_tables: bool = True, font_scale: float = 1.0, thickness: int = 1) -> np.ndarray:
    vis = img.copy()

    if draw_blocks:
        for b in r.get("blocks", []):
            if b.get("orphan"):
                # dashed grey for the fallback blocks
                x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
                continue
            x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
            color = BLOCK_COLORS.get(b.get("type", ""), (180, 180, 180))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness + 1)
            label = f"{b.get('type', '?')} {b.get('score', 0):.2f}"
            # draw the label above the box, or inside the top-left if there's no room above
            ty = y1 - 4 if y1 > 20 else y1 + 16
            renderer.draw(vis, label, x1 + 2, ty, color)

    if draw_lines:
        for l in r.get("lines", []):
            q = np.asarray(l["quad"], np.float32).astype(np.int32)
            txt = l.get("text", "")
            conf = l.get("conf", 0.0)
            if not txt.strip():
                color = LINE_EMPTY_COLOR
            elif conf < 0.5:
                color = LINE_LOW_COLOR
            else:
                color = LINE_OK_COLOR
            cv2.polylines(vis, [q], True, color, thickness, cv2.LINE_AA)
            if txt:
                # place the text just above the top edge of the quad
                top_y = int(q[:, 1].min())
                left_x = int(q[:, 0].min())
                ty = top_y - 3 if top_y > 15 else top_y + 14
                renderer.draw(vis, f"{txt}", left_x, ty, color)

    if draw_tables:
        for t in r.get("tables", []):
            for c in t.get("cells", []):
                x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), TABLE_CELL_COLOR, 1)
                txt = c.get("text", "")
                if txt:
                    renderer.draw(vis, txt, x1 + 2, y1 + 14, TABLE_CELL_COLOR)

    return vis


# ---------------------------------------------------------------------------
# Text files
# ---------------------------------------------------------------------------
def write_lines_txt(r: dict, path: Path):
    """One line per detected text line: x1 y1 x2 y2  conf  <tab>  text"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# x1\ty1\tx2\ty2\tconf\ttext\n")
        for l in r.get("lines", []):
            x1, y1, x2, y2 = l["bbox"]
            conf = l.get("conf", 0.0)
            text = l.get("text", "").replace("\t", " ").replace("\n", " ")
            f.write(f"{x1:.1f}\t{y1:.1f}\t{x2:.1f}\t{y2:.1f}\t{conf:.3f}\t{text}\n")


def write_blocks_txt(r: dict, path: Path):
    """One line per layout block: type  score  x1 y1 x2 y2  <tab>  full block text"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# type\tscore\tx1\ty1\tx2\ty2\ttext\n")
        for b in r.get("blocks", []):
            if b.get("orphan"):
                continue
            x1, y1, x2, y2 = b["bbox"]
            btype = b.get("type", "?")
            score = b.get("score", 0.0)
            text = b.get("text", "").replace("\t", " ").replace("\n", " ")
            f.write(f"{btype}\t{score:.3f}\t{x1:.1f}\t{y1:.1f}\t{x2:.1f}\t{y2:.1f}\t{text}\n")


def write_tables_txt(r: dict, path: Path):
    """Per table: header + one row per cell in row-major order."""
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(r.get("tables", [])):
            x1, y1, x2, y2 = t["bbox"]
            f.write(f"# table {i}  bbox={x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}  "
                    f"struct_conf={t.get('structure_conf', 0):.3f}  "
                    f"cells={len(t.get('cells', []))}\n")
            for c in t.get("cells", []):
                r_, col = c.get("row", -1), c.get("col", -1)
                rs, cs = c.get("rowspan", 1), c.get("colspan", 1)
                text = c.get("text", "").replace("\n", " ").replace("\t", " ")
                conf = c.get("confidence", 0.0)
                f.write(f"  row={r_} col={col} span={rs}x{cs} conf={conf:.3f}\t{text}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_pages(out_folder: Path, images_dir: Path):
    pages = []
    for jf in sorted(out_folder.glob("*.json")):
        if jf.name == "summary.json":
            continue
        r = json.load(open(jf, encoding="utf-8"))
        stem = jf.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        pages.append((stem, jf, r, img_path))
    return pages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="output folder from test_real_pages.py")
    ap.add_argument("--images", required=True, help="folder holding the original page images")
    ap.add_argument("--font", default=None, help="path to a TTF font (default: Noto Sans Devanagari, else DejaVu)")
    ap.add_argument("--font-size", type=int, default=14)
    ap.add_argument("--thickness", type=int, default=1)
    ap.add_argument("--no-lines", action="store_true", help="skip line boxes")
    ap.add_argument("--no-blocks", action="store_true", help="skip layout block boxes")
    ap.add_argument("--no-tables", action="store_true", help="skip table cell boxes")
    ap.add_argument("--no-images", action="store_true", help="only write the text files, no annotated PNGs")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser().resolve()
    images_dir = Path(a.images).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    if not images_dir.is_dir():
        raise SystemExit(f"not a folder: {images_dir}")

    out = folder / "_annotated"
    out.mkdir(exist_ok=True)
    renderer = TextRenderer(a.font, a.font_size)

    pages = load_pages(folder, images_dir)
    print(f"[annotate] {len(pages)} pages -> {out}")

    all_lines = open(out / "all_lines.tsv", "w", encoding="utf-8")
    all_lines.write("# page\tx1\ty1\tx2\ty2\tconf\ttext\n")
    all_blocks = open(out / "all_blocks.tsv", "w", encoding="utf-8")
    all_blocks.write("# page\ttype\tscore\tx1\ty1\tx2\ty2\ttext\n")

    total_lines = total_blocks = total_cells = 0
    for stem, jf, r, img_path in pages:
        # ---- text files ----
        write_lines_txt(r, out / f"{stem}.lines.txt")
        write_blocks_txt(r, out / f"{stem}.blocks.txt")
        write_tables_txt(r, out / f"{stem}.tables.txt")

        for l in r.get("lines", []):
            x1, y1, x2, y2 = l["bbox"]
            text = l.get("text", "").replace("\t", " ").replace("\n", " ")
            all_lines.write(f"{stem}\t{x1:.1f}\t{y1:.1f}\t{x2:.1f}\t{y2:.1f}\t{l.get('conf', 0):.3f}\t{text}\n")
            total_lines += 1
        for b in r.get("blocks", []):
            if b.get("orphan"):
                continue
            x1, y1, x2, y2 = b["bbox"]
            text = b.get("text", "").replace("\t", " ").replace("\n", " ")
            all_blocks.write(f"{stem}\t{b.get('type', '?')}\t{b.get('score', 0):.3f}\t"
                             f"{x1:.1f}\t{y1:.1f}\t{x2:.1f}\t{y2:.1f}\t{text}\n")
            total_blocks += 1
        for t in r.get("tables", []):
            total_cells += len(t.get("cells", []))

        # ---- annotated image ----
        if not a.no_images and img_path and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                vis = draw_annotation(
                    img, r, renderer,
                    draw_lines=not a.no_lines,
                    draw_blocks=not a.no_blocks,
                    draw_tables=not a.no_tables,
                    thickness=a.thickness,
                )
                cv2.imwrite(str(out / f"{stem}.annotated.png"), vis)

        print(f"  {stem:<45}  {len(r.get('lines', [])):>4} lines  "
              f"{len(r.get('tables', [])):>2} tables  "
              f"{sum(len(t.get('cells', [])) for t in r.get('tables', [])):>4} cells")

    all_lines.close()
    all_blocks.close()
    print(f"\n[done] {len(pages)} pages, {total_lines} lines, {total_blocks} blocks, {total_cells} table cells")
    print(f"       annotated PNGs:  {out}/*.annotated.png")
    print(f"       per-page text:   {out}/*.lines.txt  {out}/*.blocks.txt  {out}/*.tables.txt")
    print(f"       combined:        {out}/all_lines.tsv  {out}/all_blocks.tsv")


if __name__ == "__main__":
    main()