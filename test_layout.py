"""
Test a trained layout model.

1) Draw predictions on your own pages (images, folders, or PDFs):
    python test_layout.py --weights runs/layout/b8/checkpoints/best_ema_weights.pt \
        --images real_pages/ scan1.jpg report.pdf --out layout_test

   Writes layout_test/<name>.png (boxes drawn, label + score) and layout_test/predictions.json.

2) Score it on validation pages with strict box metrics:
    python test_layout.py --weights runs/layout/b8/checkpoints/best_ema_weights.pt \
        --data /path/to/data_v5 --eval-val 500

   Reports mAP at IoU 0.5 (what training logged), 0.75 (tight boxes) and 0.5:0.95 (COCO-style average),
   per class, plus the worst pages so you can look at them.

Runs fine on CPU (about 1-2 s per page), so you can test while training uses the GPU: add --device cpu.
PDF input needs PyMuPDF: pip install pymupdf
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from layout_detector import LAYOUT_CLASSES, build_layout_detector, layout_targets_from_json, postprocess_layout

COLORS = {  # BGR, one per class
    "text": (180, 119, 31), "heading": (14, 127, 255), "list": (44, 160, 44), "table": (40, 39, 214),
    "figure": (189, 103, 148), "caption": (75, 86, 140), "header": (194, 119, 227), "footer": (127, 127, 127),
    "stamp": (34, 189, 188), "signature": (207, 190, 23), "form": (0, 128, 128), "handwriting": (0, 0, 180),
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def load_model(weights: str, device: str):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    # Rebuild exactly as trained: the 'pretrained' flag decides the backbone's norm layers (frozen BN vs BN).
    # With pretrained=True torchvision uses the ImageNet weights already cached by training (no download).
    model = build_layout_detector(pretrained_backbone=cfg.get("pretrained", True),
                                  min_size=cfg.get("layout_min_size", 1024), max_size=cfg.get("layout_max_size", 1600))
    missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weights don't match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    info = f"step {ck.get('step', '?')}"
    if ck.get("val"):
        info += ", val " + ", ".join(f"{k}={v:.4f}" for k, v in ck["val"].items()
                                     if k in ("map50", "recall", "precision"))
    print(f"[model] {weights} ({info})")
    return model.to(device).eval()


@torch.no_grad()
def predict(model, img_bgr: np.ndarray, device: str, post: bool = True, score: float = 0.5,
            nms_iou: float = 0.5) -> List[Dict]:
    """post=True: same post-processing as the pipeline (layout_detector.postprocess_layout).
    post=False: every raw detection (used for AP, which needs the low-score ones too)."""
    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).permute(2, 0, 1).float().div_(255.0)
    out = model([x.to(device)])[0]
    boxes, labels, scores = out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()
    if post:
        return postprocess_layout(boxes, labels, scores, score, nms_iou)
    return [{"type": LAYOUT_CLASSES[int(l) - 1], "bbox": [round(float(v), 1) for v in b], "score": round(float(s), 4)}
            for b, l, s in zip(boxes, labels, scores)]


def draw(img_bgr: np.ndarray, dets: List[Dict], score_thresh: float) -> np.ndarray:
    vis = img_bgr.copy()
    lw = max(2, round(max(vis.shape[:2]) / 600))
    for d in sorted(dets, key=lambda d: d["score"]):
        if d["score"] < score_thresh:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        c = COLORS.get(d["type"], (0, 0, 255))
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, -1)
        vis = cv2.addWeighted(overlay, 0.12, vis, 0.88, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, lw)
        label = f"{d['type']} {d['score']:.2f}"
        fs = 0.5 * lw / 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 6, ty), c, -1)
        cv2.putText(vis, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def iter_inputs(paths: List[str], dpi: int):
    """Yields (name, BGR image) for image files, folders of images, and PDFs (one image per page)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(IMG_EXT + (".pdf",)))
        else:
            files += sorted(glob.glob(p)) or [p]
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        if f.lower().endswith(".pdf"):
            try:
                import pymupdf as fitz
            except ImportError:
                print(f"[skip] {f}: PDF input needs PyMuPDF (pip install pymupdf)")
                continue
            doc = fitz.open(f)
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                yield f"{base}_p{i + 1:03d}", img
        else:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[skip] {f}: not a readable image")
                continue
            yield base, img


# --------------------------------------------------------------------------------------
# Strict evaluation on validation pages
# --------------------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def average_precision(scores: List[float], tps: List[bool], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    tp = np.array(tps, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(model, data: str, n_pages: int, device: str, val_permille: int, out_dir: str, score_thresh: float):
    from data import DataIndex
    index = DataIndex(os.path.join(data, "labels.jsonl"), val_permille)
    pages = index.pages("val")
    pages = pages[index.page_nlayout[pages] > 0]
    if len(pages) > n_pages:
        pages = pages[np.linspace(0, len(pages) - 1, n_pages).round().astype(int)]
    thrs = np.round(np.arange(0.5, 0.96, 0.05), 2)
    per_cls = {c: {"scores": [], "tp": {t: [] for t in thrs}, "n_gt": 0} for c in range(1, len(LAYOUT_CLASSES) + 1)}
    page_scores: List[Tuple[float, str, Dict]] = []
    for k, pi in enumerate(pages):
        rec = index.read(int(pi))
        img = index.image(rec)
        dets = predict(model, img, device, post=False)
        tg = layout_targets_from_json(rec)
        gb, gl = tg["boxes"].numpy(), tg["labels"].numpy()
        pb = np.array([d["bbox"] for d in dets], np.float32).reshape(-1, 4)
        pl = np.array([LAYOUT_CLASSES.index(d["type"]) + 1 for d in dets], int)
        ps = np.array([d["score"] for d in dets], float)
        ious = iou_matrix(pb, gb)
        for c in np.unique(np.concatenate([gl, pl])):
            gi, pidx = np.nonzero(gl == c)[0], np.nonzero(pl == c)[0]
            per_cls[c]["n_gt"] += len(gi)
            pidx = pidx[np.argsort(-ps[pidx])]
            per_cls[c]["scores"] += ps[pidx].tolist()
            for t in thrs:  # greedy matching by score, per IoU threshold
                used = set()
                for j in pidx:
                    best, bi = 0.0, -1
                    for i in gi:
                        if i not in used and ious[j, i] > best:
                            best, bi = ious[j, i], i
                    hit = best >= t
                    if hit:
                        used.add(bi)
                    per_cls[c]["tp"][t].append(hit)
        # page quality: mean best IoU of each GT block with a same-class prediction above the score threshold
        keep = ps >= score_thresh
        best_iou = [max([ious[j, i] for j in np.nonzero(keep & (pl == gl[i]))[0]], default=0.0) for i in range(len(gl))]
        page_scores.append((float(np.mean(best_iou)) if best_iou else 1.0, rec["image"], {"img": img, "dets": dets}))
        if (k + 1) % 100 == 0:
            print(f"[eval] {k + 1}/{len(pages)} pages")

    print(f"\n[eval] {len(pages)} validation pages, {sum(v['n_gt'] for v in per_cls.values())} blocks\n")
    print(f"{'class':<11}{'blocks':>8}{'AP@0.5':>9}{'AP@0.75':>9}{'AP@.5:.95':>11}")
    summary = {}
    for c, v in per_cls.items():
        if v["n_gt"] == 0:
            continue
        aps = {t: average_precision(v["scores"], v["tp"][t], v["n_gt"]) for t in thrs}
        name = LAYOUT_CLASSES[c - 1]
        summary[name] = {"blocks": v["n_gt"], "ap50": aps[0.5], "ap75": aps[0.75], "ap50_95": float(np.mean(list(aps.values())))}
        s = summary[name]
        print(f"{name:<11}{s['blocks']:>8}{s['ap50']:>9.4f}{s['ap75']:>9.4f}{s['ap50_95']:>11.4f}")
    m = {k: float(np.mean([s[k] for s in summary.values()])) for k in ("ap50", "ap75", "ap50_95")}
    print(f"{'mean (mAP)':<11}{'':>8}{m['ap50']:>9.4f}{m['ap75']:>9.4f}{m['ap50_95']:>11.4f}")

    worst_dir = os.path.join(out_dir, "worst_val_pages")
    os.makedirs(worst_dir, exist_ok=True)
    page_scores.sort(key=lambda x: x[0])
    for q, name, d in page_scores[:10]:
        cv2.imwrite(os.path.join(worst_dir, f"{q:.3f}_{os.path.splitext(os.path.basename(name))[0]}.png"),
                    draw(d["img"], d["dets"], score_thresh))
    print(f"\n[eval] 10 weakest pages drawn in {worst_dir}/ (file name starts with its mean box IoU)")
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump({"mAP": m, "per_class": summary, "pages": len(pages)}, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", nargs="*", default=[], help="image files, folders, globs or PDFs")
    ap.add_argument("--out", default="layout_test")
    ap.add_argument("--score", type=float, default=0.5, help="draw boxes with at least this score")
    ap.add_argument("--dpi", type=int, default=200, help="PDF render resolution")
    ap.add_argument("--nms-iou", type=float, default=0.5, help="cross-class NMS overlap (stamp/signature exempt)")
    ap.add_argument("--raw", action="store_true", help="write every raw detection instead of the post-processed set")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", help="dataset folder (for --eval-val)")
    ap.add_argument("--eval-val", type=int, default=0, help="score on this many validation pages")
    ap.add_argument("--val-permille", type=int, default=20, help="must match training")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    os.makedirs(a.out, exist_ok=True)
    model = load_model(a.weights, device)

    if a.images:
        results = {}
        for name, img in iter_inputs(a.images, a.dpi):
            dets = predict(model, img, device, post=not a.raw, score=a.score, nms_iou=a.nms_iou)
            results[name] = {"size": [img.shape[1], img.shape[0]], "blocks": dets}
            cv2.imwrite(os.path.join(a.out, f"{name}.png"), draw(img, dets, a.score))
            kept = [d for d in dets if d["score"] >= a.score]
            counts = {}
            for d in kept:
                counts[d["type"]] = counts.get(d["type"], 0) + 1
            unsure = [f"{d['type']} {d['score']:.2f} vs {d['alt']['type']} {d['alt']['score']:.2f}"
                      for d in kept if "alt" in d and d["score"] - d["alt"]["score"] < 0.15]
            print(f"[page] {name}: {len(kept)} blocks {counts}" + (f" | unsure: {'; '.join(unsure)}" if unsure else ""))
        with open(os.path.join(a.out, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"[done] drawn pages and predictions.json in {a.out}/")
    if a.eval_val:
        if not a.data:
            ap.error("--eval-val needs --data")
        evaluate(model, a.data, a.eval_val, device, a.val_permille, a.out, a.score)
    if not a.images and not a.eval_val:
        ap.error("give --images and/or --eval-val")


if __name__ == "__main__":
    main()