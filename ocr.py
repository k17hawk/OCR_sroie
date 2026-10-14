"""
Run the full OCR pipeline (layout -> line detection -> recognition -> tables -> merge) on real pages.

    python ocr.py --runs /media/lang-chain/b5ee0e21-c51a-4025-949a-a7e888058222/runs \
                  --input real_pages/ --out ocr_out

Models are found automatically: for each of layout / det / table / rec, the newest
<runs>/<model>/<run>/checkpoints/best_ema_weights.pt is used (override with --layout / --det / --table / --rec).
Each model is rebuilt exactly as it was trained, from the config saved in its checkpoint.

Output
  <out>/report.html               open in a browser: every page with drawn boxes, extracted text and tables
  <out>/summary.json              per-page counts, confidence and timing
  <out>/<page>/text.md            the page as text (headings "## ", lists one item per line, tables as HTML)
  <out>/<page>/table_<k>.html|csv each table (CSV opens in Excel / LibreOffice)
  <out>/<page>/result.json        every line (text, confidence, needs_review, box) and every block / table cell
  <out>/<page>/vis.png            layout blocks (coloured, labelled) + text lines (green ok, red needs review)

Images (png/jpg/jpeg/tif/tiff/bmp/webp) and PDFs (pip install pymupdf) are accepted.
"""
import argparse
import glob
import html
import json
import os
import time
import traceback
from typing import Dict, Optional

import cv2
import numpy as np
import torch

import layout_detector as LD
from detector_dbnet import DBNetPP
from merge import grid_to_csv
from pipeline import OCRPipeline
from recognizer_svtr import CTCCodec, SVTRRecognizer
from table_slanet import TableStructureModel, TableVocab

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
BLOCK_COLORS = {  # BGR
    "text": (180, 119, 31), "heading": (14, 127, 255), "list": (44, 160, 44), "table": (40, 39, 214),
    "figure": (189, 103, 148), "caption": (75, 86, 140), "header": (194, 119, 227), "footer": (127, 127, 127),
    "stamp": (34, 189, 188), "signature": (207, 190, 23), "form": (0, 128, 128), "handwriting": (0, 0, 180),
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def find_checkpoint(runs: str, model: str, override: Optional[str]) -> str:
    if override:
        return override
    cands = glob.glob(os.path.join(runs, model, "*", "checkpoints", "best_ema_weights.pt"))
    if not cands:
        raise SystemExit(f"no checkpoint for '{model}' under {os.path.join(runs, model)}/*/checkpoints/ "
                         f"(pass --{model} PATH)")
    return max(cands, key=os.path.getmtime)


def read_ckpt(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck, ck.get("config", {}), ck.get("extra", {}) or {}, ck.get("val") or {}


def load_strict(model: torch.nn.Module, sd: Dict, name: str, path: str):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name}: {path} does not match the model built from its config "
                           f"(missing {list(missing)[:3]}, unexpected {list(unexpected)[:3]})")


def describe(name, path, ck, val, keys):
    shown = ", ".join(f"{k}={val[k]:.4f}" for k in keys if isinstance(val.get(k), (int, float)))
    print(f"[{name:6s}] {path}  (step {ck.get('step', '?')}{', ' + shown if shown else ''})")


def load_models(a):
    models = {}
    # layout
    p = find_checkpoint(a.runs, "layout", a.layout)
    ck, cfg, extra, val = read_ckpt(p)
    sd = ck["model"]
    n_cls = sd["roi_heads.box_predictor.cls_score.weight"].shape[0] - 1
    classes = extra.get("layout_classes")
    if classes is not None and list(classes) != list(LD.LAYOUT_CLASSES):
        raise SystemExit(f"layout classes in the checkpoint {classes} differ from layout_detector.LAYOUT_CLASSES "
                         f"{LD.LAYOUT_CLASSES}")
    if n_cls != len(LD.LAYOUT_CLASSES):
        raise SystemExit(f"layout model has {n_cls} classes but layout_detector.LAYOUT_CLASSES has "
                         f"{len(LD.LAYOUT_CLASSES)}: use the layout_detector.py the model was trained with")
    m = LD.build_layout_detector(pretrained_backbone=cfg.get("pretrained", True),
                                 min_size=cfg.get("layout_min_size", 1024), max_size=cfg.get("layout_max_size", 1600))
    load_strict(m, sd, "layout", p)
    models["layout"] = m
    describe("layout", p, ck, val, ["map50"])

    # line detector
    p = find_checkpoint(a.runs, "det", a.det)
    ck, cfg, extra, val = read_ckpt(p)
    m = DBNetPP(pretrained_backbone=False)
    load_strict(m, ck["model"], "det", p)
    models["det"] = m
    models["det_long_side"] = a.det_long_side or cfg.get("det_long_side", 1600)
    models["unclip"] = a.unclip or val.get("best_unclip_ratio") or 1.5
    describe("det", p, ck, val, ["f1", "best_unclip_ratio"])

    # table structure
    p = find_checkpoint(a.runs, "table", a.table)
    ck, cfg, extra, val = read_ckpt(p)
    vocab = TableVocab(max_span=extra.get("table_vocab_max_span", 20))
    m = TableStructureModel(vocab, pretrained_backbone=cfg.get("pretrained", True))
    load_strict(m, ck["model"], "table", p)
    models["table"] = m
    models["table_size"] = extra.get("table_size", cfg.get("table_size", 640))
    describe("table", p, ck, val, ["exact_structure", "cell_box_iou"])

    # recognizer
    p = find_checkpoint(a.runs, "rec", a.rec)
    ck, cfg, extra, val = read_ckpt(p)
    if "charset" not in extra:
        raise SystemExit(f"{p} has no charset; use the best_ema_weights.pt written by train.py")
    codec = CTCCodec(extra["charset"])
    m = SVTRRecognizer(codec.num_classes, in_height=cfg.get("rec_height", 48),
                       drop_path=cfg.get("drop_path", 0.1), dropout=cfg.get("dropout", 0.1))
    load_strict(m, ck["model"], "rec", p)
    models["rec"], models["codec"] = m, codec
    models["rec_height"], models["rec_max_width"] = cfg.get("rec_height", 48), cfg.get("rec_max_width", 2560)
    describe("rec", p, ck, val, ["cer", "line_acc"])
    return models


# --------------------------------------------------------------------------------------
# Input pages
# --------------------------------------------------------------------------------------
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
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[skip] cannot read {path}")
                continue
            yield stem, img


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted(jsonable(v) for v in x)
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), 1).tolist()
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def draw(img, res, review_thresh):
    vis = img.copy()
    lw = max(2, round(max(vis.shape[:2]) / 700))
    for b in res["blocks"]:
        x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
        c = BLOCK_COLORS.get(b.get("type", "text"), (0, 0, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, lw + 1)
        label = b.get("type", "?") + (" (orphan)" if b.get("orphan") else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, max(y1, th + 6)), c, -1)
        cv2.putText(vis, label, (x1 + 3, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for l in res["lines"]:
        c = (40, 40, 220) if l["conf"] < review_thresh else (40, 170, 40)
        cv2.polylines(vis, [np.asarray(l["quad"]).reshape(-1, 1, 2).astype(np.int32)], True, c, 1 if c[1] > 100 else 2)
    return vis


def page_outputs(out_dir, name, img, res, seconds, review_thresh):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "text.md"), "w", encoding="utf-8") as f:
        f.write(res["text"] + "\n")
    for k, t in enumerate(res["tables"]):
        with open(os.path.join(d, f"table_{k + 1}.html"), "w", encoding="utf-8") as f:
            f.write(t["html"])
        with open(os.path.join(d, f"table_{k + 1}.csv"), "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel shows Devanagari
            f.write(grid_to_csv(t["grid"]))
    lines = [{"text": l["text"], "conf": l["conf"], "conf_mean": l.get("conf_mean"), "needs_review": l["needs_review"],
              "bbox": l["bbox"], "quad": l["quad"], "det_score": l.get("det_score")} for l in res["lines"]]
    blocks = [{"type": b.get("type"), "bbox": b["bbox"], "score": b.get("score"), "alt": b.get("alt"),
               "orphan": b.get("orphan", False), "text": b.get("text"), "conf": b.get("conf"),
               "n_lines": len(b.get("lines", [])), "is_table": "table" in b,
               "dropped_table": b.get("dropped_table")} for b in res["blocks"]]
    tables = [{"bbox": t["bbox"], "confidence": t.get("confidence"), "structure_conf": t.get("structure_conf"),
               "html": t["html"], "cells": [{k: c.get(k) for k in ("row", "col", "rowspan", "colspan", "header",
                                                                      "text", "confidence", "needs_review", "reread")}
                                            for c in t["cells"]]} for t in res["tables"]]
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable({"page": name, "size": [img.shape[1], img.shape[0]], "seconds": seconds,
                            "text": res["text"], "blocks": blocks, "lines": lines, "tables": tables}),
                  f, ensure_ascii=False, indent=1)
    vis = draw(img, res, review_thresh)
    cv2.imencode(".png", vis)[1].tofile(os.path.join(d, "vis.png"))
    n_rev = sum(l["needs_review"] for l in res["lines"])
    confs = [l["conf"] for l in res["lines"]]
    return {"page": name, "lines": len(res["lines"]), "needs_review": n_rev,
            "mean_conf": round(float(np.mean(confs)), 4) if confs else None,
            "blocks": len(res["blocks"]), "tables": len(res["tables"]), "seconds": round(seconds, 2)}


def write_report(out_dir, rows):
    css = """
    :root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--muted:#5d6675;--line:#dde1e7;--warn:#b42318;--ok:#1a7f37}
    @media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1c2028;--fg:#e6e9ee;--muted:#9aa3b2;--line:#2c323d;--warn:#ff7b72;--ok:#56d364}}
    body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Noto Sans Devanagari","Mukta","Nirmala UI",system-ui,sans-serif}
    main{max-width:1500px;margin:0 auto;padding:24px 16px}
    h1{font-size:22px;margin:0 0 4px} .muted{color:var(--muted)}
    table.sum{border-collapse:collapse;width:100%;margin:16px 0 32px;background:var(--card)}
    table.sum td,table.sum th{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;font-variant-numeric:tabular-nums}
    section{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:0 0 28px;padding:16px}
    .grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
    @media (max-width:900px){.grid{grid-template-columns:minmax(0,1fr)}}
    img{max-width:100%;border:1px solid var(--line)}
    pre{white-space:pre-wrap;word-break:break-word;margin:0;font:inherit}
    .tbl{overflow-x:auto;margin-top:12px} .tbl table{border-collapse:collapse} .tbl td,.tbl th{border:1px solid var(--line);padding:4px 8px}
    .warn{color:var(--warn)} .ok{color:var(--ok)}
    """
    tot_lines = sum(r.get("lines", 0) for r in rows)
    tot_rev = sum(r.get("needs_review", 0) for r in rows)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>OCR report</title><style>{css}</style></head><body><main>",
             f"<h1>OCR report</h1><div class='muted'>{len(rows)} pages, {tot_lines} lines, "
             f"{tot_rev} lines flagged for review ({100 * tot_rev / max(tot_lines, 1):.1f}%)</div>",
             "<table class='sum'><tr><th>page</th><th>lines</th><th>needs review</th><th>mean conf</th>"
             "<th>blocks</th><th>tables</th><th>seconds</th></tr>"]
    for r in rows:
        if "error" in r:
            parts.append(f"<tr><td><a href='#{html.escape(r['page'])}'>{html.escape(r['page'])}</a></td>"
                         f"<td colspan='6' class='warn'>error: {html.escape(r['error'])}</td></tr>")
            continue
        cls = "warn" if r["needs_review"] else "ok"
        parts.append(f"<tr><td><a href='#{html.escape(r['page'])}'>{html.escape(r['page'])}</a></td><td>{r['lines']}</td>"
                     f"<td class='{cls}'>{r['needs_review']}</td><td>{r['mean_conf'] if r['mean_conf'] is not None else '-'}</td>"
                     f"<td>{r['blocks']}</td><td>{r['tables']}</td><td>{r['seconds']}</td></tr>")
    parts.append("</table>")
    for r in rows:
        name = r["page"]
        parts.append(f"<section id='{html.escape(name)}'><h2>{html.escape(name)}</h2>")
        if "error" in r:
            parts.append(f"<pre class='warn'>{html.escape(r['error'])}</pre></section>")
            continue
        text = open(os.path.join(out_dir, name, "text.md"), encoding="utf-8").read()
        tables = sorted(glob.glob(os.path.join(out_dir, name, "table_*.html")))
        tbl_html = "".join(f"<div class='tbl'><div class='muted'>{os.path.basename(t)}</div>"
                           f"{open(t, encoding='utf-8').read()}</div>" for t in tables)
        parts.append(f"<div class='grid'><div><img loading='lazy' src='{html.escape(name)}/vis.png' alt='{html.escape(name)}'></div>"
                     f"<div><pre>{html.escape(text)}</pre>{tbl_html}</div></div></section>")
    parts.append("</main></body></html>")
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write("".join(parts))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder with layout/ det/ table/ rec/ run folders")
    ap.add_argument("--input", required=True, help="folder, file or glob of images / PDFs")
    ap.add_argument("--out", default="ocr_out")
    ap.add_argument("--layout"); ap.add_argument("--det"); ap.add_argument("--table"); ap.add_argument("--rec")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dpi", type=int, default=144, help="PDF render resolution")
    ap.add_argument("--det-long-side", type=int, default=None)
    ap.add_argument("--unclip", type=float, default=None, help="default: best unclip ratio from validation")
    ap.add_argument("--box-thresh", type=float, default=0.6, help="line detector score threshold")
    ap.add_argument("--review-thresh", type=float, default=0.8, help="lines below this confidence are flagged")
    ap.add_argument("--preprocess", action="store_true", help="deskew + contrast before OCR (off by default)")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    files = list_inputs(a.input)
    if not files:
        raise SystemExit(f"no images or PDFs found at {a.input}")
    M = load_models(a)
    pipe = OCRPipeline(detector=M["det"], recognizer=M["rec"], codec=M["codec"], layout=M["layout"], table=M["table"],
                       device=device, review_thresh=a.review_thresh, table_size=M["table_size"],
                       do_preprocess=a.preprocess, det_long_side=M["det_long_side"], rec_height=M["rec_height"],
                       rec_max_width=M["rec_max_width"],
                       det_kwargs={"unclip_ratio": M["unclip"], "box_thresh": a.box_thresh})
    print(f"[setup] device {device} | det long side {M['det_long_side']} | unclip {M['unclip']} | "
          f"box_thresh {a.box_thresh} | table size {M['table_size']} | {len(M['codec'].chars)} characters")
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for name, img in iter_pages(files, a.dpi):
        t0 = time.time()
        try:
            with torch.no_grad():
                res = pipe(img)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            row = page_outputs(a.out, name, img, res, time.time() - t0, a.review_thresh)
            print(f"  {name}: {row['lines']} lines ({row['needs_review']} to review), {row['blocks']} blocks, "
                  f"{row['tables']} tables, {row['seconds']} s", flush=True)
        except Exception as e:  # keep going; the report lists the failure
            traceback.print_exc()
            row = {"page": name, "error": f"{type(e).__name__}: {e}"}
            print(f"  {name}: ERROR {row['error']}", flush=True)
        rows.append(row)

    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    write_report(a.out, rows)
    ok = [r for r in rows if "error" not in r]
    n_lines = sum(r["lines"] for r in ok)
    n_rev = sum(r["needs_review"] for r in ok)
    print(f"\n===== {len(rows)} pages ({len(rows) - len(ok)} errors) | {n_lines} lines | {n_rev} flagged for review "
          f"({100 * n_rev / max(n_lines, 1):.1f}%) | {sum(r['seconds'] for r in ok) / max(len(ok), 1):.1f} s/page =====")
    print(f"open {os.path.join(a.out, 'report.html')} in a browser")


if __name__ == "__main__":
    main()