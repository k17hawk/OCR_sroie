"""
Test the trained table-structure model on real tables.

Two ways to feed it:
  1) Full pages: the layout model finds the tables first (default).
        python test_table.py --input real_pages/ --out table_test
  2) Table crops you cut yourself (each image = one table). This tests the table model ALONE, without layout
     mistakes (a wrong or loose table box) getting in the way:
        python test_table.py --input my_table_crops/ --crops --out table_test

Add --text to also read the cell text (loads det + rec too; rec can still be training).

Writes:
  <out>/index.html        open in a browser: each table crop next to the structure the model predicted
  <out>/tables/*.png      each crop with the predicted cells drawn (blue; red = low-confidence cell) and numbered
  <out>/summary.json      per table: rows, cells, structure confidence, flags
Weights are read from --runs (default: the runs folder on the data disk).
"""
import argparse
import html
import json
import os

import cv2
import numpy as np
import torch

from table_slanet import TableStructureModel, TableVocab, boxes_to_crop_pixels, letterbox
from test_pipeline import DEFAULT_RUNS, iter_pages, list_inputs, load_ckpt, load_into


def load_table(runs, run):
    sd, cfg, extra, val, step = load_ckpt(os.path.join(runs, "table", run, "checkpoints", "best_ema_weights.pt"))
    vocab = TableVocab()
    model = load_into(TableStructureModel(vocab, pretrained_backbone=cfg.get("pretrained", True)), sd, "table")
    size = extra.get("table_size", cfg.get("table_size", 640))
    print(f"[model] table step {step} | synthetic exact_structure {val.get('exact_structure')} | size {size}")
    return model, size


def load_layout(runs, run):
    from layout_detector import build_layout_detector
    sd, cfg, *_ = load_ckpt(os.path.join(runs, "layout", run, "checkpoints", "best_ema_weights.pt"))
    return load_into(build_layout_detector(pretrained_backbone=cfg.get("pretrained", True),
                                           min_size=cfg.get("layout_min_size", 1024),
                                           max_size=cfg.get("layout_max_size", 1600)), sd, "layout")


def make_text_pipeline(runs, table, size, device):
    from detector_dbnet import DBNetPP
    from pipeline import OCRPipeline
    from recognizer_svtr import CTCCodec, SVTRRecognizer
    sd, cfg, extra, *_ = load_ckpt(os.path.join(runs, "rec", "main", "checkpoints", "best_ema_weights.pt"))
    codec = CTCCodec(extra["charset"])
    rec = load_into(SVTRRecognizer(codec.num_classes, in_height=cfg.get("rec_height", 48)), sd, "rec")
    rec_h, rec_w = cfg.get("rec_height", 48), cfg.get("rec_max_width", 2560)
    sd, cfg, extra, val, _ = load_ckpt(os.path.join(runs, "det", "main", "checkpoints", "best_ema_weights.pt"))
    det = load_into(DBNetPP(pretrained_backbone=cfg.get("pretrained", True)), sd, "det")
    return OCRPipeline(detector=det, recognizer=rec, codec=codec, table=table, device=device, table_size=size,
                       det_long_side=cfg.get("det_long_side", 1600), rec_height=rec_h, rec_max_width=rec_w,
                       det_kwargs={"unclip_ratio": val.get("best_unclip_ratio") or 1.5})


def structure_html(tokens):
    """The predicted structure as an HTML table with numbered empty cells (number i = box i in the drawing).
    Handles '<td>', '<td></td>' and split tokens like '<td', ' colspan="2"', '>'."""
    out, i, open_td = [], 0, False
    for t in tokens:
        if t == "<td></td>":
            out.append(f"<td><small>{i}</small></td>"); i += 1
        elif t == "<td>":
            out.append(f"<td><small>{i}</small>"); i += 1
        elif t == "<td":
            out.append(t); open_td = True
        elif t == ">" and open_td:
            out.append(f"><small>{i}</small>"); i += 1; open_td = False
        else:
            out.append(t)
    s = "".join(out)
    return s if s.lstrip().startswith("<table") else f"<table>{s}</table>"


def flags_for(tokens, boxes, probs, w, h):
    flags = []
    n_rows = sum(t == "<tr>" for t in tokens)
    n_cells = len(boxes)
    if n_rows == 0 or n_cells == 0:
        flags.append("no structure")
        return flags, n_rows, n_cells
    if probs is not None and len(probs) and min(probs) < 0.8:
        flags.append(f"{sum(p < 0.8 for p in probs)} low-confidence cells")
    if len(boxes):
        b = np.asarray(boxes)
        out = ((b[:, 0] < -0.02 * w) | (b[:, 1] < -0.02 * h) | (b[:, 2] > 1.02 * w) | (b[:, 3] > 1.02 * h)).sum()
        if out:
            flags.append(f"{out} cells outside the table")
        tiny = (((b[:, 2] - b[:, 0]) < 4) | ((b[:, 3] - b[:, 1]) < 4)).sum()
        if tiny:
            flags.append(f"{tiny} degenerate cells")
    # cells per row (a row with far fewer cells than the others is usually a missed column)
    per_row, cur = [], 0
    for t in tokens:
        if t == "<tr>":
            cur = 0
        elif t.startswith("<td"):
            cur += 1
        elif t == "</tr>":
            per_row.append(cur)
    if per_row and max(per_row) - min(per_row) >= 2:
        flags.append(f"ragged rows ({min(per_row)}-{max(per_row)} cells per row; OK only if there are spans)")
    return flags, n_rows, n_cells


def draw_cells(crop, boxes, probs):
    vis = crop.copy()
    lw = max(1, round(max(vis.shape[:2]) / 700))
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        low = probs is not None and i < len(probs) and probs[i] < 0.8
        c = (0, 0, 230) if low else (220, 120, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, lw + low)
        cv2.putText(vis, str(i), (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder, file or glob of images / PDFs")
    ap.add_argument("--out", default="table_test")
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--table-run", default="main")
    ap.add_argument("--layout-run", default="b8")
    ap.add_argument("--crops", action="store_true", help="each input image is already one table")
    ap.add_argument("--min-score", type=float, default=0.5, help="layout score needed to treat a box as a table")
    ap.add_argument("--pad", type=int, default=8, help="pixels added around layout table boxes")
    ap.add_argument("--text", action="store_true", help="also read cell text (loads det + rec)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dpi", type=int, default=144)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    files = list_inputs(args.input)
    if not files:
        raise SystemExit(f"no images or PDFs found at {args.input}")
    table, size = load_table(args.runs, args.table_run)
    table = table.to(device)
    layout = None if args.crops else load_layout(args.runs, args.layout_run).to(device)
    pipe = make_text_pipeline(args.runs, table, size, device) if args.text else None
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    rows, cards = [], []
    for name, img in iter_pages(files, args.dpi):
        H, W = img.shape[:2]
        if args.crops:
            found = [([0.0, 0.0, float(W), float(H)], 1.0, None)]
        else:
            from layout_detector import detect_layout
            blocks = detect_layout(layout, img, device)
            found = [(b["bbox"], b["score"], b.get("alt")) for b in blocks
                     if b["type"] == "table" and b["score"] >= args.min_score]
            if not found:
                print(f"  {name}: no table found by layout")
                continue
        lines = None
        if pipe is not None:
            from detector_dbnet import detect_lines
            quads, sc = detect_lines(pipe.detector, img, device, long_side=pipe.det_long_side, **pipe.det_kwargs)
            lines = pipe.recognize_lines(img, quads, sc) if quads else []
        for k, (bbox, lscore, alt) in enumerate(found):
            p = 0 if args.crops else args.pad
            x1, y1 = max(int(bbox[0]) - p, 0), max(int(bbox[1]) - p, 0)
            x2, y2 = min(int(np.ceil(bbox[2])) + p, W), min(int(np.ceil(bbox[3])) + p, H)
            crop = img[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            if h < 8 or w < 8:
                continue
            with torch.no_grad():
                pred = table.predict(letterbox(crop, size)[None].to(device))[0]
            tokens = pred["tokens"]
            boxes = boxes_to_crop_pixels(np.array(pred["cell_boxes"]), w, h, size) if pred["cell_boxes"] else np.zeros((0, 4))
            probs = list(pred.get("cell_probs") or [])
            sconf = float(np.min(pred["token_probs"])) if pred.get("token_probs") else 0.0
            flags, n_rows, n_cells = flags_for(tokens, boxes, probs, w, h)
            if sconf < 0.8:
                flags.insert(0, f"low structure confidence {sconf:.2f}")
            if alt is not None and lscore - alt["score"] < 0.15:
                flags.append(f"layout unsure: table {lscore:.2f} vs {alt['type']} {alt['score']:.2f}")
            tid = f"{name}_t{k + 1}"
            cv2.imencode(".png", draw_cells(crop, boxes, probs))[1].tofile(os.path.join(args.out, "tables", f"{tid}.png"))
            table_html = structure_html(tokens)
            if pipe is not None:
                t = pipe.read_table(img, [x1, y1, x2, y2], lines)
                if t is not None:
                    table_html = t.get("html", table_html)
            row = {"table": tid, "rows": n_rows, "cells": n_cells, "structure_conf": round(sconf, 3),
                   "layout_score": round(float(lscore), 3), "bbox": [x1, y1, x2, y2], "flags": flags}
            rows.append(row)
            cards.append((tid, row, table_html))
            print(f"  {tid}: {n_rows} rows, {n_cells} cells, structure conf {sconf:.2f}"
                  + (f" | {'; '.join(flags)}" if flags else ""), flush=True)

    rows.sort(key=lambda r: (-len(r["flags"]), r["structure_conf"]))
    order = {r["table"]: i for i, r in enumerate(rows)}
    cards.sort(key=lambda c: order[c[0]])
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    page = ["<!doctype html><meta charset='utf-8'><title>Table test</title><style>"
            "body{font-family:'Noto Sans Devanagari',sans-serif;margin:20px}"
            ".card{display:flex;gap:20px;align-items:flex-start;border-bottom:1px solid #ccc;padding:16px 0;flex-wrap:wrap}"
            ".card img{max-width:620px;width:100%;border:1px solid #ddd}.pred{flex:1;min-width:300px;overflow:auto}"
            "table{border-collapse:collapse}td,th{border:1px solid #666;padding:3px 6px;min-width:20px;vertical-align:top}"
            "small{color:#999}.flag{color:#b00}h3{margin:0 0 6px}</style>",
            f"<h2>{len(rows)} tables (worst first)</h2>"]
    for tid, r, th in cards:
        page.append(f"<div class='card'><div><h3>{html.escape(tid)}</h3><img src='tables/{html.escape(tid)}.png'></div>"
                    f"<div class='pred'><div>{r['rows']} rows · {r['cells']} cells · structure conf "
                    f"{r['structure_conf']:.2f}</div>"
                    + "".join(f"<div class='flag'>{html.escape(fl)}</div>" for fl in r["flags"])
                    + f"<div style='margin-top:8px'>{th}</div></div></div>")
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(page))

    n = len(rows)
    print(f"\n===== {n} tables | {sum(bool(r['flags']) for r in rows)} flagged | mean structure conf "
          f"{np.mean([r['structure_conf'] for r in rows]) if rows else 0:.3f} =====")
    for r in rows[:10]:
        if r["flags"]:
            print(f"  {r['table']}: {'; '.join(r['flags'])}")
    print(f"\n-> open {args.out}/index.html")


if __name__ == "__main__":
    main()