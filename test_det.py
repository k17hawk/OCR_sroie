"""
Test the trained text-line detector (DBNet++) on real pages: drawn lines + a problem report (+ optional line crops).

    python test_det.py --weights runs/det/main/checkpoints/best_ema_weights.pt --input real_pages/ --out det_test

--input can be a folder, a single file, or a glob. Images (png/jpg/jpeg/tif/tiff/bmp/webp) and PDFs are accepted
(PDFs need `pip install pymupdf`; pages are rendered at --dpi).

What it checks on every page
  merged  : one box holding two or more text lines (found from the ink profile inside the box, so a large title
            is NOT flagged, only real stacked lines)
  dup     : a box that mostly overlaps a higher-scoring box (that text would be read twice)
  low     : score between --box-thresh and --keep-thresh: found, but the pipeline (box_thresh 0.6) would DROP it
  tiny    : implausibly small box
  missed  : text-like ink that no kept box covers (blue in the drawing) - text the pipeline would silently lose

Writes:
  <out>/vis/<page>.png      green ok, red merged, magenta dup/tiny, orange low, BLUE = missed ink
  <out>/predictions.json    {page: {"size": [w, h], "lines": [{"quad", "score", "status"}], "missed": [[x1,y1,x2,y2]]}}
  <out>/report.json         per-page flags, worst pages first
  <out>/crops/<page>/*.png  with --save-crops: each kept line, cropped exactly as pipeline.py crops it for the
                            recognizer. Type the text for a few hundred of these and you have a real recognizer test set.

While another model is training, use --device cpu (about 1-3 s per page) to stay out of its GPU memory.
"""
import argparse
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np
import torch

from detector_dbnet import DBNetPP, detect_lines
from pipeline import crop_quad  # the exact crop the recognizer gets at inference

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_model(path: str, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    model = DBNetPP(pretrained_backbone=False)  # architecture is identical; the trained weights are loaded below
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{path} does not match DBNetPP: missing {missing[:4]}..., unexpected {unexpected[:4]}...")
    val = (ck.get("val") or {}) if isinstance(ck, dict) else {}
    print(f"[model] {path} | step {ck.get('step', '?')} | synthetic val f1={val.get('f1', '?')} "
          f"best unclip={val.get('best_unclip_ratio', '?')}")
    return model.to(device).eval(), cfg, val


def list_inputs(spec: str):
    files = [os.path.join(spec, f) for f in sorted(os.listdir(spec))] if os.path.isdir(spec) else \
        (sorted(glob.glob(spec)) or [spec])
    return [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXT | {".pdf"}]


def iter_pages(files, dpi: int):
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if path.lower().endswith(".pdf"):
            try:
                import pymupdf as fitz
            except ImportError:
                try:
                    import fitz
                except ImportError:
                    print(f"[skip] {path}: pip install pymupdf to read PDFs")
                    continue
            with fitz.open(path) as doc:
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=dpi)
                    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
                    yield f"{stem}_page{i + 1:03d}", cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)  # non-ASCII file names OK
            if img is None:
                print(f"[skip] cannot read {path}")
                continue
            yield stem, img


def quad_size(q):
    w = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2
    h = (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2
    return float(w), float(h)


def rect(q):
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink (255) robust to uneven lighting."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)


def count_text_bands(crop_bgr: np.ndarray, med_h: float) -> int:
    """
    Number of separate text lines stacked inside a line crop, from its horizontal ink profile. A band must be at
    least a quarter of a normal line tall and hold >= 15% of the ink, so thin underlines and stray matras don't count.
    """
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    prof = ink.sum(1).astype(np.float32)
    if prof.max() <= 0:
        return 0
    on = prof > 0.12 * prof.max()
    bands, start = [], None
    for y, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append((start, y))
            start = None
    total = prof.sum()
    real = [b for b in bands if (b[1] - b[0]) >= 0.25 * med_h and prof[b[0]:b[1]].sum() >= 0.15 * total]
    return len(real)


def find_missed(img: np.ndarray, kept_quads, med_h: float):
    """
    Text-like ink not covered by any kept line box. Long table rulings are removed first; very tall blobs (stamps,
    photos, logos) are ignored. Returns merged regions [x1, y1, x2, y2].
    """
    H, W = img.shape[:2]
    ink = ink_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 25, 15), 1)))
    lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 25, 15))))
    ink = cv2.subtract(ink, lines)
    cover = np.zeros((H, W), np.uint8)
    for q in kept_quads:
        cv2.fillPoly(cover, [q.astype(np.int32)], 255)
    pad = max(int(0.3 * med_h), 2) if med_h else 3
    cover = cv2.dilate(cover, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    min_h = max(0.3 * med_h, 4) if med_h else 6
    max_h = 3 * med_h if med_h else H / 10
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 or h < min_h or h > max_h or w > W * 0.9:
            continue  # specks, tall non-text blobs (stamps / photos), page-wide artefacts
        thin = max(3, 0.2 * med_h) if med_h else 3
        if (w <= thin and h >= 0.5 * (med_h or 10)) or (h <= thin and w >= 2 * (med_h or 10)):
            continue  # ruling fragments that survived line removal (skewed scans, short row separators)
        comp = lab[y:y + h, x:x + w] == i
        if (cover[y:y + h, x:x + w][comp] > 0).mean() < 0.3:
            keep[i] = True
    missed = (keep[lab] * 255).astype(np.uint8)
    if not missed.any():
        return []
    k = max(int(med_h) if med_h else 10, 3)
    joined = cv2.dilate(missed, cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(k // 3, 1))))
    n2, _, st2, _ = cv2.connectedComponentsWithStats(joined, 8)
    regions = [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h, a in st2[1:] if a >= (k * k) // 2]
    return regions


def classify(img, quads, scores, keep_thresh: float):
    """Per-line status: ok / merged / dup / tiny / low."""
    heights = np.array([quad_size(q)[1] for q in quads]) if quads else np.zeros(0)
    kept_h = heights[np.array(scores) >= keep_thresh] if len(heights) else heights
    med = float(np.median(kept_h)) if len(kept_h) else (float(np.median(heights)) if len(heights) else 0.0)
    rects = [rect(q) for q in quads]
    order = np.argsort(-np.array(scores)) if scores else []
    dup = set()
    for a_i, a in enumerate(order):
        for b in order[a_i + 1:]:
            if b in dup or a in dup:
                continue
            ra, rb = rects[a], rects[b]
            ix = max(0.0, min(ra[2], rb[2]) - max(ra[0], rb[0]))
            iy = max(0.0, min(ra[3], rb[3]) - max(ra[1], rb[1]))
            area_b = max((rb[2] - rb[0]) * (rb[3] - rb[1]), 1e-6)
            if ix * iy / area_b >= 0.6:
                dup.add(int(b))
    status = []
    for i, (q, s, h) in enumerate(zip(quads, scores, heights)):
        w = quad_size(q)[0]
        if w < 6 or h < 4:
            status.append("tiny")
        elif i in dup:
            status.append("dup")
        elif med and h > 1.5 * med and count_text_bands(crop_quad(img, q), med) >= 2:
            status.append("merged")
        elif s < keep_thresh:
            status.append("low")
        else:
            status.append("ok")
    return status, med


COLORS = {"ok": (40, 170, 40), "merged": (0, 0, 230), "low": (0, 140, 255), "tiny": (200, 0, 200),
          "dup": (200, 0, 200), "missed": (230, 90, 0)}


def draw(img, quads, status, missed):
    vis = img.copy()
    for (x1, y1, x2, y2) in missed:
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS["missed"], -1)
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["missed"], 2)
    for i, (q, st) in enumerate(zip(quads, status)):
        pts = q.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, COLORS[st], 3 if st != "ok" else 2)
        if st != "ok":
            x, y = int(q[:, 0].min()), int(q[:, 1].min())
            cv2.putText(vis, f"{i}:{st}", (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[st], 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/det/main/checkpoints/best_ema_weights.pt")
    ap.add_argument("--input", required=True, help="folder, file or glob of images / PDFs")
    ap.add_argument("--out", default="det_test")
    ap.add_argument("--device", default="auto", help="use cpu while another model trains")
    ap.add_argument("--long-side", type=int, default=None, help="default: the det_long_side used in training")
    ap.add_argument("--unclip", type=float, default=None, help="default: the best unclip ratio from validation")
    ap.add_argument("--box-thresh", type=float, default=0.4,
                    help="detect down to this score, so near-misses become visible as 'low'")
    ap.add_argument("--keep-thresh", type=float, default=0.6,
                    help="what the pipeline keeps (its box_thresh); lines below are flagged 'low'")
    ap.add_argument("--dpi", type=int, default=144)
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--no-vis", action="store_true")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    files = list_inputs(args.input)
    if not files:
        raise SystemExit(f"no images or PDFs found at {args.input}")
    model, cfg, val = load_model(args.weights, device)
    long_side = args.long_side or cfg.get("det_long_side", 1600)
    unclip = args.unclip or val.get("best_unclip_ratio") or 1.5
    print(f"[setup] device {device} | long side {long_side} | unclip_ratio {unclip} | "
          f"detect >= {args.box_thresh}, pipeline keeps >= {args.keep_thresh}")
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)

    preds, report, totals = {}, [], Counter()
    for name, img in iter_pages(files, args.dpi):
        H, W = img.shape[:2]
        quads, scores = detect_lines(model, img, device, long_side=long_side, unclip_ratio=unclip,
                                     box_thresh=args.box_thresh)
        quads = [np.asarray(q, np.float32).reshape(4, 2) for q in quads]
        scores = [float(s) for s in scores]
        status, med_h = classify(img, quads, scores, args.keep_thresh)
        kept = [q for q, st in zip(quads, status) if st in ("ok", "merged")]
        low_quads = [q for q, st in zip(quads, status) if st == "low"]
        missed = find_missed(img, kept, med_h)
        # how many missed regions a lower box_thresh would have caught
        recoverable = 0
        for (x1, y1, x2, y2) in missed:
            if any(rect(q)[0] < x2 and rect(q)[2] > x1 and rect(q)[1] < y2 and rect(q)[3] > y1 for q in low_quads):
                recoverable += 1
        c = Counter(status)
        totals.update(c)
        totals["lines"] += len(quads)
        totals["missed"] += len(missed)
        totals["recoverable"] += recoverable
        preds[name] = {"size": [W, H], "median_line_height": round(med_h, 1), "missed": missed,
                       "lines": [{"quad": q.round(1).tolist(), "score": round(s, 4), "status": st}
                                 for q, s, st in zip(quads, scores, status)]}
        flags = [f"{c[k]} {k}" for k in ("merged", "dup", "low", "tiny") if c[k]]
        if missed:
            flags.append(f"{len(missed)} missed-ink regions" + (f" ({recoverable} caught by low boxes)" if recoverable else ""))
        if not quads:
            flags.append("no lines found")
        n_flags = sum(c[k] for k in ("merged", "dup", "tiny")) + 2 * len(missed) + c["low"] // 2 + (not quads) * 10
        report.append({"page": name, "lines": len(quads), "median_line_height": round(med_h, 1),
                       "n_flags": n_flags, "flags": flags})
        if not args.no_vis:
            cv2.imencode(".png", draw(img, quads, status, missed))[1].tofile(os.path.join(args.out, "vis", f"{name}.png"))
        if args.save_crops and kept:
            d = os.path.join(args.out, "crops", name)
            os.makedirs(d, exist_ok=True)
            idx = [i for i, st in enumerate(status) if st in ("ok", "merged")]
            idx.sort(key=lambda i: (round(quads[i][:, 1].mean() / max(med_h, 1)), quads[i][:, 0].min()))
            for rank, i in enumerate(idx):
                cv2.imencode(".png", crop_quad(img, quads[i]))[1].tofile(os.path.join(d, f"{rank:04d}_{status[i]}.png"))
        print(f"  {name}: {len(quads)} lines, median height {med_h:.0f}px" + (f" | {', '.join(flags)}" if flags else ""),
              flush=True)

    with open(os.path.join(args.out, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    report.sort(key=lambda r: -r["n_flags"])
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    n, L = len(preds), max(totals["lines"], 1)
    print(f"\n===== {n} pages, {totals['lines']} line boxes ({totals['lines'] / max(n, 1):.1f} per page) =====")
    for k in ("ok", "merged", "dup", "low", "tiny"):
        print(f"  {k:7s} {totals[k]:6d}  ({100 * totals[k] / L:.1f}%)")
    print(f"  missed-ink regions: {totals['missed']}  ({totals['recoverable']} would be caught by keeping 'low' boxes)")
    print("\nworst pages (look at these first):")
    for r in [r for r in report if r["n_flags"]][:10]:
        print(f"  {r['page']}: {', '.join(r['flags'])}")
    print(f"\n-> {args.out}/vis/, {args.out}/predictions.json, {args.out}/report.json"
          + (f", {args.out}/crops/" if args.save_crops else ""))


if __name__ == "__main__":
    main()