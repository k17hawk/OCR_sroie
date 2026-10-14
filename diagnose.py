#!/usr/bin/env python3
"""
Surface every kind of failure in the OCR output.

    python diagnose.py ~/Documents/real_pages/_ocr_results
    python diagnose.py ~/Documents/real_pages/_ocr_results --page page004
    python diagnose.py ~/Documents/real_pages/_ocr_results --open
"""
import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def load_pages(folder: Path, images_dir: Path = None):
    """folder: output folder with *.json. images_dir: where the page images live (default: folder.parent)."""
    if images_dir is None:
        images_dir = folder.parent
    images_dir = Path(images_dir).expanduser().resolve()
    out = []
    for jf in sorted(folder.glob("*.json")):
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
        out.append((stem, jf, r, img_path))
    return out

# ----------------------------------------------------------------------
# Reporters
# ----------------------------------------------------------------------
def report_orphans(pages):
    """Orphan blocks = pipeline's fallback, meaning the layout model missed a text region."""
    print("\n" + "=" * 90)
    print("LAYOUT: orphan text blocks (layout model returned nothing here, pipeline made a 1-line block)")
    print("=" * 90)
    total_real = total_orphan = 0
    per_page = []
    for stem, jf, r, _ in pages:
        blocks = r.get("blocks", [])
        real = [b for b in blocks if not b.get("orphan")]
        orph = [b for b in blocks if b.get("orphan")]
        total_real += len(real)
        total_orphan += len(orph)
        if orph:
            per_page.append((stem, len(real), len(orph)))
    per_page.sort(key=lambda x: -x[2])
    print(f"  {'page':<45} {'real blocks':>12} {'orphans':>9}")
    print("  " + "-" * 68)
    for stem, nr, no in per_page[:25]:
        flag = "  <-- layout barely fired" if nr <= 2 and no >= 5 else ""
        print(f"  {stem[:45]:<45} {nr:>12} {no:>9}{flag}")
    if len(per_page) > 25:
        print(f"  ... and {len(per_page) - 25} more pages")
    print(f"\n  TOTAL: {total_real} real layout blocks vs {total_orphan} orphans")
    if total_orphan > total_real:
        print(f"  -> layout model is missing most text regions on these pages")


def report_low_conf_lines(pages, thresh=0.5, n=30):
    """Low-confidence recognizer output. Show the text so you can judge if it's actually wrong."""
    print("\n" + "=" * 90)
    print(f"RECOGNIZER: lowest-confidence lines (conf_min < {thresh}) — judge by eye if text is wrong")
    print("=" * 90)
    rows = []
    for stem, jf, r, _ in pages:
        for l in r.get("lines", []):
            if l.get("conf", 1.0) < thresh:
                rows.append((l["conf"], stem, l.get("text", "")))
    rows.sort()
    print(f"  {len(rows)} lines below {thresh} (out of {sum(len(r.get('lines',[])) for _,_,r,_ in pages)} total)")
    print()
    print(f"  {'conf':>6}  {'page':<38}  text")
    print("  " + "-" * 86)
    for c, s, t in rows[:n]:
        print(f"  {c:>6.3f}  {s[:38]:<38}  {t[:50]}")
    return rows


def report_empty_lines(pages):
    """Detected a line box but the recognizer produced no characters — usually det false positives."""
    print("\n" + "=" * 90)
    print("DETECTOR+REC: line boxes the recognizer could not read (empty output)")
    print("=" * 90)
    count = 0
    for stem, jf, r, _ in pages:
        empty = [l for l in r.get("lines", []) if not l.get("text", "").strip()]
        if empty:
            count += len(empty)
            boxes = ", ".join(f"({int(b['bbox'][0])},{int(b['bbox'][1])})" for b in empty[:4])
            more = f" +{len(empty)-4} more" if len(empty) > 4 else ""
            print(f"  {stem[:45]:<45}  {len(empty):>3} empty  at {boxes}{more}")
    print(f"\n  TOTAL: {count} empty line boxes")
    print("  -> each is a detector false positive (or a real line the recognizer cannot read)")


def report_small_boxes(pages, min_h=8):
    """Line boxes shorter than min_h px — likely spurious (commas, noise, paper grain)."""
    print("\n" + "=" * 90)
    print(f"DETECTOR: suspiciously small line boxes (height < {min_h} px)")
    print("=" * 90)
    count = 0
    for stem, jf, r, _ in pages:
        small = []
        for l in r.get("lines", []):
            q = np.asarray(l["quad"], np.float32)
            h = max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))
            if h < min_h:
                small.append((h, l.get("text", "")))
        if small:
            count += len(small)
            samples = ", ".join(f"{h:.1f}px:'{t[:15]}'" for h, t in small[:3])
            print(f"  {stem[:45]:<45}  {len(small):>3} small  {samples}")
    print(f"\n  TOTAL: {count} tiny boxes")
    print("  -> lower db_postprocess min_size or filter them out post-hoc")


def report_tables(pages):
    """Table structure: does each detected table have cells with text?"""
    print("\n" + "=" * 90)
    print("TABLE: structure confidence and cell fill rate")
    print("=" * 90)
    print(f"  {'page':<38} {'#':>2} {'cells':>6} {'filled':>7} {'struct_conf':>12} {'used_lines':>11} {'straddle':>9}")
    print("  " + "-" * 88)
    issues = 0
    for stem, jf, r, _ in pages:
        for i, t in enumerate(r.get("tables", [])):
            cells = t.get("cells", [])
            filled = sum(1 for c in cells if c.get("text", "").strip())
            sc = t.get("structure_conf", 0.0)
            used = len(t.get("used_line_ids", []))
            straddle = len(t.get("straddling_line_ids", []))
            flag = ""
            if cells == 0:
                flag = "  <-- empty structure"; issues += 1
            elif filled == 0 and used == 0:
                flag = "  <-- no text assigned"; issues += 1
            elif sc < 0.5:
                flag = "  <-- low struct conf"; issues += 1
            print(f"  {stem[:38]:<38} {i:>2} {len(cells):>6} {filled:>7} {sc:>12.3f} {used:>11} {straddle:>9}{flag}")
    print(f"\n  {issues} tables with problems")


def report_missing_classes(pages):
    """Which layout classes fired? Compare with what you'd expect on the pages."""
    print("\n" + "=" * 90)
    print("LAYOUT: class distribution (real blocks only, orphans excluded)")
    print("=" * 90)
    counts = Counter()
    scores = {}
    for stem, jf, r, _ in pages:
        for b in r.get("blocks", []):
            if b.get("orphan"):
                continue
            t = b.get("type", "?")
            counts[t] += 1
            scores.setdefault(t, []).append(b.get("score", 0))
    print(f"  {'class':<12} {'count':>6} {'mean score':>12} {'pages present':>14}")
    print("  " + "-" * 46)
    all_classes = ["text", "heading", "list", "table", "figure", "caption", "header", "footer",
                   "stamp", "signature", "form", "handwriting"]
    pages_with = Counter()
    for stem, jf, r, _ in pages:
        seen = {b.get("type") for b in r.get("blocks", []) if not b.get("orphan")}
        for t in seen:
            pages_with[t] += 1
    for t in all_classes:
        n = counts.get(t, 0)
        m = sum(scores[t]) / len(scores[t]) if scores.get(t) else 0.0
        pw = pages_with.get(t, 0)
        flag = "  <-- never detected" if n == 0 else ("  <-- rare" if n < 5 else "")
        print(f"  {t:<12} {n:>6} {m:>12.3f} {pw:>11}/{len(pages)}{flag}")


# ----------------------------------------------------------------------
# Visual: produce a "problems overlay" per page
# ----------------------------------------------------------------------
def draw_problems(img, r):
    """Overlay: green = real layout block, grey = orphan, red = low-conf line, orange = empty line, blue = table cell."""
    vis = img.copy()
    for b in r.get("blocks", []):
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        if b.get("orphan"):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(vis, f"{b.get('type','?')} {b.get('score',0):.2f}",
                        (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
    for l in r.get("lines", []):
        q = np.asarray(l["quad"], np.float32).astype(np.int32)
        if not l.get("text", "").strip():
            col = (0, 165, 255)      # orange = empty
        elif l.get("conf", 1.0) < 0.5:
            col = (0, 0, 255)        # red = low conf
        else:
            col = (255, 0, 0)        # blue = fine
        cv2.polylines(vis, [q], True, col, 1, cv2.LINE_AA)
    for t in r.get("tables", []):
        for c in t.get("cells", []):
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 128, 255), 1)
    return vis


def make_overlays(pages, folder, only=None):
    out = folder / "_diagnostics"
    out.mkdir(exist_ok=True)
    for stem, jf, r, img_path in pages:
        if only and only not in stem:
            continue
        if img_path is None or not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imwrite(str(out / f"{stem}.problems.png"), draw_problems(img, r))
    print(f"\n  overlays written to {out}/")


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the output folder produced by test_real_pages.py")
    ap.add_argument("--images", default=None,
                    help="folder holding the original page images (default: parent of <folder>)")
    ap.add_argument("--page", default=None, help="only build overlay for pages matching this substring")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    pages = load_pages(folder, a.images)
    if not pages:
        raise SystemExit(f"no *.json found in {folder}")
    print(f"loaded {len(pages)} pages from {folder}")

    report_missing_classes(pages)
    report_orphans(pages)
    report_low_conf_lines(pages, thresh=a.conf)
    report_empty_lines(pages)
    report_small_boxes(pages)
    report_tables(pages)
    make_overlays(pages, folder, only=a.page)

    if a.open:
        out = folder / "_diagnostics"
        first = next(iter(sorted(out.glob("*.png"))), None)
        if first:
            os.system(f"xdg-open {first}")

if __name__ == "__main__":
    main()
