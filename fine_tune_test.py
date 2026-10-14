#!/usr/bin/env python3
"""
Run the full OCR pipeline on a folder of page images.

Loads the fine-tuned models from runs_2/*/finetune_*/checkpoints/best_ema_weights.pt
and falls back to the originals in runs/ when a fine-tuned checkpoint is missing
(e.g. the rec fine-tune was skipped because of the charset mismatch).

Usage:
    python test_real_pages.py                                  # defaults below
    python test_real_pages.py /path/to/images
    python test_real_pages.py /path/to/images --limit 3 --no-table
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from detector_dbnet import DBNetPP
from layout_detector import LAYOUT_CLASSES, build_layout_detector
from pipeline import OCRPipeline
from recognizer_svtr import CTCCodec, SVTRRecognizer
from table_slanet import DEFAULT_S, TableStructureModel, TableVocab

# ---------------------------------------------------------------------------
# Defaults — edit if your paths differ
# ---------------------------------------------------------------------------
DEFAULT_IMAGES = Path("/home/lang-chain/Documents/OCR_sroie/real_pages")
DEFAULT_RUNS_ROOT = Path("/media/lang-chain/b5ee0e21-c51a-4025-949a-a7e888058222")

# (base_dir_name, run_name) pairs to try in order, per model
CKPT_CANDIDATES = {
    "layout": [("runs_2", "finetune_layout"), ("runs", "b8"), ("runs", "main")],
    "det":    [("runs_2", "finetune_det"),    ("runs", "main")],
    "table":  [("runs_2", "finetune_table"),  ("runs", "main")],
    "rec":    [("runs_2", "finetune_rec"),    ("runs", "main")],
}

LAYOUT_COLORS = {
    "text": (0, 200, 0), "heading": (0, 128, 255), "list": (255, 128, 0),
    "table": (0, 0, 255), "figure": (128, 0, 128), "caption": (128, 128, 0),
    "header": (200, 200, 0), "footer": (200, 0, 200), "stamp": (0, 255, 255),
    "signature": (255, 0, 128), "form": (128, 128, 128), "handwriting": (60, 60, 60),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_ckpt(root: Path, model: str) -> Path:
    for base, run in CKPT_CANDIDATES[model]:
        p = root / base / model / run / "checkpoints" / "best_ema_weights.pt"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no checkpoint for '{model}': looked in "
        + ", ".join(str(root / b / model / r) for b, r in CKPT_CANDIDATES[model])
    )


def load_into(model, path: Path, name: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {name}: missing keys {list(missing)[:5]}"
              + (f" ... +{len(missing)-5}" if len(missing) > 5 else ""))
    if unexpected:
        print(f"  [warn] {name}: unexpected keys {list(unexpected)[:5]}"
              + (f" ... +{len(unexpected)-5}" if len(unexpected) > 5 else ""))
    return ckpt


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(to_jsonable(v) for v in x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def draw_viz(img: np.ndarray, result: dict) -> np.ndarray:
    vis = img.copy()

    for b in result.get("blocks", []):
        if b.get("type") == "table" or "table" in b:
            continue  # tables drawn separately below
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        color = LAYOUT_COLORS.get(b.get("type", ""), (180, 180, 180))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{b.get('type', '?')} {b.get('score', 0):.2f}"
        cv2.putText(vis, label, (x1 + 3, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for l in result.get("lines", []):
        q = np.asarray(l["quad"], np.float32).astype(np.int32)
        cv2.polylines(vis, [q], True, (255, 0, 0), 1, cv2.LINE_AA)

    for t in result.get("tables", []):
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for c in t.get("cells", []):
            cx1, cy1, cx2, cy2 = [int(v) for v in c["bbox"]]
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 128, 255), 1)

    return vis


def page_summary(result: dict) -> dict:
    lines = result.get("lines", [])
    tables = result.get("tables", [])
    confs = [l["conf"] for l in lines]
    needs_review = sum(1 for l in lines if l.get("needs_review"))
    return {
        "n_lines": len(lines),
        "n_tables": len(tables),
        "n_blocks": len(result.get("blocks", [])),
        "text_chars": len(result.get("text", "")),
        "min_line_conf": float(min(confs)) if confs else None,
        "mean_line_conf": float(np.mean(confs)) if confs else None,
        "lines_needing_review": needs_review,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="?", default=str(DEFAULT_IMAGES),
                    help=f"folder of page images (default: {DEFAULT_IMAGES})")
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT),
                    help="root holding runs/ and runs_2/ (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: <images>/_ocr_results)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-layout", action="store_true")
    ap.add_argument("--no-table", action="store_true")
    ap.add_argument("--no-viz", action="store_true", help="skip drawing/saving the overlay images")
    ap.add_argument("--rec-ckpt", default=None, help="explicit rec checkpoint path (overrides the default lookup)")
    ap.add_argument("--review-thresh", type=float, default=0.8)
    a = ap.parse_args()

    images_dir = Path(a.images).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"not a folder: {images_dir}")
    runs_root = Path(a.runs_root).expanduser().resolve()
    out_dir = Path(a.out).expanduser().resolve() if a.out else images_dir / "_ocr_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if a.limit:
        images = images[:a.limit]
    if not images:
        raise SystemExit(f"no images found in {images_dir} (extensions: {sorted(exts)})")
    print(f"[data] {len(images)} image(s) in {images_dir}")

    device = 'cpu'
    print(f"[dev]  {device}")
    if device.startswith("cuda"):
        print(f"       {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")

    # ---- rec: charset comes from the checkpoint's `extra` field, not from a file ----
    rec_ckpt_path = Path(a.rec_ckpt) if a.rec_ckpt else find_ckpt(runs_root, "rec")
    print(f"[load] rec      <- {rec_ckpt_path}")
    rec_ckpt = torch.load(rec_ckpt_path, map_location="cpu", weights_only=False)
    charset = rec_ckpt.get("extra", {}).get("charset")
    if not charset:
        raise SystemExit(f"{rec_ckpt_path} has no `extra.charset`; can't rebuild the CTC codec")
    print(f"       charset: {len(charset)} characters (+ blank = {len(charset)+1} classes)")
    codec = CTCCodec(charset)
    rcfg = rec_ckpt.get("config", {}) or {}
    recognizer = SVTRRecognizer(
        codec.num_classes,
        in_height=rcfg.get("rec_height", 48),
        drop_path=rcfg.get("drop_path", 0.1),
        dropout=rcfg.get("dropout", 0.1),
    )
    load_into(recognizer, rec_ckpt_path, "rec")

    # ---- detector ----
    det_ckpt_path = find_ckpt(runs_root, "det")
    print(f"[load] det      <- {det_ckpt_path}")
    detector = DBNetPP(pretrained_backbone=False)
    load_into(detector, det_ckpt_path, "det")

    # ---- layout (optional) ----
    layout_model = None
    if not a.no_layout:
        try:
            layout_ckpt_path = find_ckpt(runs_root, "layout")
            print(f"[load] layout   <- {layout_ckpt_path}")
            lcfg = torch.load(layout_ckpt_path, map_location="cpu", weights_only=False).get("config", {}) or {}
            layout_model = build_layout_detector(
                pretrained_backbone=False,
                min_size=lcfg.get("layout_min_size", 1024),
                max_size=lcfg.get("layout_max_size", 1600),
            )
            load_into(layout_model, layout_ckpt_path, "layout")
        except FileNotFoundError as e:
            print(f"[warn] no layout model ({e}); page will be one text block")

    # ---- table (optional) ----
    table_model = None
    table_size = DEFAULT_S
    if not a.no_table:
        try:
            table_ckpt_path = find_ckpt(runs_root, "table")
            print(f"[load] table    <- {table_ckpt_path}")
            tckpt = torch.load(table_ckpt_path, map_location="cpu", weights_only=False)
            vocab = TableVocab(max_span=tckpt.get("extra", {}).get("table_vocab_max_span", 20))
            table_size = tckpt.get("extra", {}).get("table_size", DEFAULT_S)
            table_model = TableStructureModel(vocab, pretrained_backbone=False)
            load_into(table_model, table_ckpt_path, "table")
        except FileNotFoundError as e:
            print(f"[warn] no table model ({e}); tables will be read as plain text")

    # ---- pipeline ----
    pipe = OCRPipeline(
        layout=layout_model,
        detector=detector,
        recognizer=recognizer,
        codec=codec,
        table=table_model,
        device=device,
        review_thresh=a.review_thresh,
        table_size=table_size,
        do_preprocess=False,
        det_long_side=rcfg.get("det_long_side", 1600),
        rec_height=rcfg.get("rec_height", 48),
        rec_max_width=rcfg.get("rec_max_width", 2560),
    )
    print("[pipe] ready\n")

    # ---- run ----
    summary_all = {"images": [], "config": {
        "images_dir": str(images_dir),
        "runs_root": str(runs_root),
        "rec_ckpt": str(rec_ckpt_path),
        "det_ckpt": str(det_ckpt_path),
        "layout_ckpt": str(layout_ckpt_path) if layout_model is not None else None,
        "table_ckpt": str(table_ckpt_path) if table_model is not None else None,
        "charset_size": len(charset),
    }}

    t_all = time.time()
    for i, p in enumerate(images, 1):
        t0 = time.time()
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{i}/{len(images)}] {p.name}: FAILED to read")
            summary_all["images"].append({"file": p.name, "error": "unreadable"})
            continue
        H, W = img.shape[:2]
        try:
            result = pipe(img)
        except Exception as e:
            print(f"[{i}/{len(images)}] {p.name}: pipeline error: {type(e).__name__}: {e}")
            summary_all["images"].append({"file": p.name, "error": f"{type(e).__name__}: {e}"})
            continue
        dt = time.time() - t0
        s = page_summary(result)
        s["file"] = p.name
        s["size"] = [W, H]
        s["seconds"] = round(dt, 2)
        summary_all["images"].append(s)

        stem = p.stem
        (out_dir / f"{stem}.txt").write_text(result["text"], encoding="utf-8")
        (out_dir / f"{stem}.json").write_text(
            json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
        if not a.no_viz:
            cv2.imwrite(str(out_dir / f"{stem}.overlay.png"), draw_viz(img, result))

        review_note = f"  review: {s['lines_needing_review']}" if s["lines_needing_review"] else ""
        conf_note = f"  conf min {s['min_line_conf']:.3f}" if s["min_line_conf"] is not None else ""
        print(f"[{i}/{len(images)}] {p.name:<40} {W}x{H}  {s['n_lines']:>4} lines  "
              f"{s['n_tables']:>2} tables  {dt:>5.1f}s{conf_note}{review_note}")

    summary_all["total_seconds"] = round(time.time() - t_all, 1)
    (out_dir / "summary.json").write_text(
        json.dumps(summary_all, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [s for s in summary_all["images"] if "error" not in s]
    if ok:
        print(f"\n[done] {len(ok)}/{len(images)} pages in {summary_all['total_seconds']:.1f}s  "
              f"({summary_all['total_seconds']/max(len(ok),1):.2f}s/page)")
        print(f"       output: {out_dir}")
    else:
        print(f"\n[done] no pages succeeded; check the errors above")


if __name__ == "__main__":
    main()