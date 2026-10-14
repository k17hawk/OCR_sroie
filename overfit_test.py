"""
Overfit test: can each model MEMORISE a handful of your real training samples?

If a model cannot drive its error to ~0 on 8-128 samples it has seen hundreds of times, something is broken
(labels misaligned with crops, wrong normalisation, a charset gap, a bad target) and full training would only
waste GPU days. Augmentation, dropout and drop-path are switched off on purpose.

    python overfit_test.py --data path/to/folder                         # all four models
    python overfit_test.py --data path/to/folder --models rec,table      # a subset
    python overfit_test.py --data path/to/folder --no-pretrained         # offline (no ImageNet download)

Pass criteria (on the SAME samples it trained on):
    rec    : character error rate <= 1%
    det    : F1 >= 0.90 at IoU 0.5 (also reports the best unclip_ratio; copy it into OCRPipeline(det_kwargs=...))
    table  : >= 90% of tables with exactly the right structure tokens, mean cell-box IoU >= 0.75
    layout : recall >= 0.90 at IoU 0.5 with the correct class (score >= 0.5)

Rough GPU times: rec 3-8 min, det 3-6 min, table 2-5 min, layout 3-6 min. CPU works but is 10-30x slower.
Writes overfit_report.json next to labels.jsonl. Exit code is non-zero if any model fails.
"""
import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from data import (DataIndex, DetDataset, LayoutDataset, RecConfig, RecDataset, TableDataset, check_layout_types,
                  det_collate, layout_collate, load_or_build_charset, make_table_collate, rec_collate, usable_lines)
from detector_dbnet import DBNetPP, _quad_iou, db_loss, db_postprocess
from layout_detector import LAYOUT_CLASSES, build_layout_detector
from recognizer_svtr import CTCCodec, SVTRRecognizer, ctc_loss_fn
from table_slanet import TableStructureModel, TableVocab, table_loss


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Autocast:
    """bf16 autocast on GPUs that support it, fp16 + GradScaler otherwise, nothing on CPU."""

    def __init__(self, device: str, enabled: bool):
        self.on = enabled and device.startswith("cuda")
        self.dtype = torch.bfloat16 if self.on and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.on and self.dtype == torch.float16)

    def ctx(self):
        return torch.autocast("cuda", dtype=self.dtype, enabled=self.on)

    def step(self, loss, opt, params, clip=5.0):
        opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, clip)
        self.scaler.step(opt)
        self.scaler.update()


def warmup(opt, steps):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / steps))


def run_loop(name: str, model, opt, batches: List, loss_fn: Callable, eval_fn: Callable, passed: Callable,
             steps: int, eval_every: int, max_minutes: float, amp: Autocast) -> Dict:
    """Cycles over pre-built batches; evaluates every eval_every steps; stops early on pass."""
    sched = warmup(opt, min(100, steps // 10 + 1))
    params = [p for p in model.parameters() if p.requires_grad]
    t0, hist, metrics, step = time.time(), [], {}, 0
    order = np.random.default_rng(0)
    model.train()
    while step < steps:
        for bi in order.permutation(len(batches)):
            with amp.ctx():
                loss = loss_fn(batches[bi])
            if not torch.isfinite(loss):
                return {"status": "FAIL", "reason": f"non-finite loss at step {step}", "steps": step}
            amp.step(loss, opt, params)
            sched.step()
            hist.append(loss.item())
            step += 1
            if step % 25 == 0:
                print(f"  [{name}] step {step:5d}  loss {np.mean(hist[-25:]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
            if step % eval_every == 0 or step >= steps:
                model.eval()
                with torch.no_grad():
                    metrics = eval_fn()
                model.train()
                print(f"  [{name}] eval @ {step}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in metrics.items() if k != "examples"), flush=True)
                if passed(metrics):
                    return {"status": "PASS", "steps": step, "seconds": round(time.time() - t0), **metrics}
            if step >= steps or (time.time() - t0) / 60 > max_minutes:
                break
        if (time.time() - t0) / 60 > max_minutes:
            break
    reason = "time budget reached" if step < steps else "step budget reached"
    return {"status": "FAIL", "reason": reason, "steps": step, "seconds": round(time.time() - t0), **metrics,
            "first_loss": hist[0] if hist else None, "last_loss": float(np.mean(hist[-25:])) if hist else None}


def first_pages(index: DataIndex, mask: np.ndarray, n: int) -> np.ndarray:
    """First n train pages satisfying mask (deterministic)."""
    pages = index.pages("train")
    return pages[mask[pages]][:n]


# --------------------------------------------------------------------------------------
# Recognizer
# --------------------------------------------------------------------------------------
def test_rec(index, args, device, amp) -> Dict:
    codec = CTCCodec(load_or_build_charset(index, os.path.join(index.root, "charset.json")))
    cfg = RecConfig()
    ids = usable_lines(index, "train", codec, cfg)
    if not len(ids):
        return {"status": "SKIP", "reason": "no usable training lines"}
    # spread the sample over pages (max 8 lines per page) so it is not one font / one layout
    picked, per_page = [], {}
    for lid in ids:
        p = int(index.line_page[lid])
        if per_page.get(p, 0) < 8:
            per_page[p] = per_page.get(p, 0) + 1
            picked.append(lid)
        if len(picked) >= args.rec_lines:
            break
    ds = RecDataset(index, "train", codec, cfg, augment=False, line_ids=np.array(picked), verbose=False)
    samples = [ds[i] for i in range(len(ds))]
    samples.sort(key=lambda s: s["image"].shape[1])
    bs = args.rec_batch
    batches = [rec_collate(samples[i:i + bs]) for i in range(0, len(samples), bs)]
    batches = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()} for b in batches]
    n_chars = sum(len(s["text"]) for s in samples)
    print(f"[rec] {len(samples)} lines from {len(per_page)} pages, {n_chars} chars, widths "
          f"{samples[0]['image'].shape[1]}-{samples[-1]['image'].shape[1]} px, {codec.num_classes - 1} classes")

    model = SVTRRecognizer(codec.num_classes, drop_path=0.0, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.rec_lr, weight_decay=0.0)
    skipped = {"n": 0}

    def loss_fn(b):
        logits = model(b["images"], b["widths"])
        loss, st = ctc_loss_fn(logits, b["targets"], b["target_lengths"], SVTRRecognizer.out_lengths(b["widths"]))
        skipped["n"] += st["skipped"]
        return loss

    def eval_fn():
        errs, worst = 0, []
        for b in batches:
            logits = model(b["images"], b["widths"]).float().cpu()
            for d, gt in zip(codec.decode_greedy(logits, SVTRRecognizer.out_lengths(b["widths"]).cpu()), b["texts"]):
                e = levenshtein(d["text"], gt)
                errs += e
                if e:
                    worst.append((e, gt, d["text"]))
        worst.sort(key=lambda x: -x[0])
        return {"cer": errs / max(n_chars, 1), "wrong_lines": len(worst), "ctc_skipped": skipped["n"],
                "examples": [{"gt": g, "pred": p} for _, g, p in worst[:5]]}

    res = run_loop("rec", model, opt, batches, loss_fn, eval_fn, lambda m: m["cer"] <= 0.01,
                   args.rec_steps, 100, args.max_minutes, amp)
    for ex in res.get("examples", []):
        print(f"    GT  : {ex['gt']}\n    PRED: {ex['pred']}")
    return res


# --------------------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------------------
def det_scores(prob: np.ndarray, gts: List[np.ndarray], ratio: float, iou_thr=0.5):
    preds, _ = db_postprocess(prob, unclip_ratio=ratio)
    used, tp = set(), 0
    for g in gts:
        best, bj = 0.0, -1
        for j, p in enumerate(preds):
            if j in used:
                continue
            iou = _quad_iou(np.asarray(g, np.float32), p)
            if iou > best:
                best, bj = iou, j
        if best >= iou_thr:
            tp += 1
            used.add(bj)
    return tp, len(preds), len(gts)


def test_det(index, args, device, amp) -> Dict:
    pages = first_pages(index, index.page_nlines > 0, args.det_pages)
    if not len(pages):
        return {"status": "SKIP", "reason": "no pages with lines"}
    ds = DetDataset(index, "train", long_side=args.det_long_side, crop=None, augment=False, pages=pages)
    samples = [ds[i] for i in range(len(ds))]
    batches = [det_collate(samples[i:i + 2]) for i in range(0, len(samples), 2)]
    batches = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()} for b in batches]
    print(f"[det] {len(samples)} pages at long side {args.det_long_side}, "
          f"{sum(len(s['quads']) for s in samples)} text lines")
    model = DBNetPP(pretrained_backbone=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.det_lr, weight_decay=1e-4)
    keys = ("gt", "gt_mask", "thresh_map", "thresh_mask")

    def loss_fn(b):
        out = model(b["image"])
        return db_loss(out, {k: b[k] for k in keys})[0]

    ratios = (1.2, 1.5, 1.8, 2.0, 2.5, 3.0)

    def eval_fn():
        probs = []
        for b in batches:
            p = model(b["image"])["prob"].float().cpu().numpy()[:, 0]
            probs += list(p)
        best = {"f1": 0.0, "unclip_ratio": None}
        for r in ratios:
            tp = npred = ngt = 0
            for p, s in zip(probs, samples):
                h, w = s["image"].shape[1:]
                a, b_, c = det_scores(p[:h, :w], s["quads"], r)
                tp, npred, ngt = tp + a, npred + b_, ngt + c
            prec, rec = tp / max(npred, 1), tp / max(ngt, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best["f1"]:
                best = {"f1": f1, "precision": prec, "recall": rec, "unclip_ratio": r}
        return best

    return run_loop("det", model, opt, batches, loss_fn, eval_fn, lambda m: m["f1"] >= 0.9,
                    args.det_steps, 100, args.max_minutes, amp)


# --------------------------------------------------------------------------------------
# Table structure
# --------------------------------------------------------------------------------------
def test_table(index, args, device, amp) -> Dict:
    pages = first_pages(index, index.page_ntables > 0, args.table_n)
    pairs = [(int(p), t) for p in pages for t in range(int(index.page_ntables[p]))][:args.table_n]
    if not pairs:
        return {"status": "SKIP", "reason": "no tables"}
    vocab = TableVocab()
    ds = TableDataset(index, "train", vocab, S=args.table_size, pairs=pairs)
    samples = [s for s in (ds[i] for i in range(len(ds))) if s is not None]
    if not samples:
        return {"status": "FAIL", "reason": f"all {len(pairs)} tables were unusable (see messages above)"}
    collate = make_table_collate(vocab.pad)
    batches = [collate(samples[i:i + 8]) for i in range(0, len(samples), 8)]
    batches = [{k: v.to(device) for k, v in b.items()} for b in batches]
    print(f"[table] {len(samples)} tables ({ds.skipped} unusable), up to "
          f"{max(len(s['tokens']) for s in samples)} tokens")
    model = TableStructureModel(vocab, pretrained_backbone=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.table_lr, weight_decay=0.0)

    ss_state = {"step": 0}

    def loss_fn(b):
        # Scheduled sampling: ramp ss_prob 0 -> table_ss_max over the first table_ss_warmup steps.
        # With prob ss, the decoder is fed its own previous argmax instead of the ground truth,
        # which trains it to recover from its own mistakes (fixes exposure bias).
        ss = min(args.table_ss_max, args.table_ss_max * ss_state["step"] / max(1, args.table_ss_warmup))
        ss_state["step"] += 1
        lg, bx = model(b["image"], b["tokens"], ss_prob=ss)
        return table_loss(lg, bx, b["tokens"], b["boxes"], b["box_mask"], vocab.pad)[0]

    def eval_fn():
        exact, ious = 0, []
        for b in batches:
            preds = model.predict(b["image"], max_len=b["tokens"].shape[1] + 20, beam_size=args.table_beam)
            for i, pr in enumerate(preds):
                ids = b["tokens"][i].tolist()
                gt = [vocab.tokens[t] for t in ids[1:ids.index(vocab.eos)]]
                if pr["tokens"] == gt:
                    exact += 1
                    m = b["box_mask"][i]
                    g = b["boxes"][i][m].cpu().numpy()
                    g = np.stack([g[:, 0] - g[:, 2] / 2, g[:, 1] - g[:, 3] / 2,
                                  g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2], 1)
                    ious += [box_iou(a, c) for a, c in zip(pr["cell_boxes"], g)]
        return {"exact_structure": exact / len(samples), "cell_box_iou": float(np.mean(ious)) if ious else 0.0}

    return run_loop("table", model, opt, batches, loss_fn, eval_fn,
                    lambda m: m["exact_structure"] >= 0.9 and m["cell_box_iou"] >= 0.75,
                    args.table_steps, 100, args.max_minutes, amp)


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------
def test_layout(index, args, device, amp) -> Dict:
    check_layout_types(index)
    pages = first_pages(index, index.page_nlayout > 0, args.layout_pages)
    if not len(pages):
        return {"status": "SKIP", "reason": "no layout blocks"}
    ds = LayoutDataset(index, "train", pages=pages)
    samples = [ds[i] for i in range(len(ds))]
    batches = [layout_collate(samples[i:i + 2]) for i in range(0, len(samples), 2)]
    batches = [([x.to(device) for x in ims], [{k: v.to(device) for k, v in t.items()} for t in tg]) for ims, tg in batches]
    print(f"[layout] {len(samples)} pages, {sum(len(t['labels']) for _, t in samples)} blocks")
    model = build_layout_detector(pretrained_backbone=args.pretrained, min_size=args.layout_min_size,
                                  max_size=args.layout_max_size).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.layout_lr, weight_decay=1e-4)

    def loss_fn(b):
        return sum(model(b[0], b[1]).values())

    def eval_fn():
        hit = tot = npred = 0
        for ims, tgs in batches:
            outs = model(ims)
            for o, t in zip(outs, tgs):
                keep = o["scores"] >= 0.5
                pb, pl = o["boxes"][keep].tolist(), o["labels"][keep].tolist()
                npred += len(pb)
                used = set()
                for gb, gl in zip(t["boxes"].tolist(), t["labels"].tolist()):
                    tot += 1
                    cands = [(box_iou(gb, b), j) for j, (b, l) in enumerate(zip(pb, pl)) if l == gl and j not in used]
                    if cands and max(cands)[0] >= 0.5:
                        hit += 1
                        used.add(max(cands)[1])
        return {"recall": hit / max(tot, 1), "precision": hit / max(npred, 1)}

    return run_loop("layout", model, opt, batches, loss_fn, eval_fn, lambda m: m["recall"] >= 0.9,
                    args.layout_steps, 100, args.max_minutes, amp)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", default="labels.jsonl")
    ap.add_argument("--val-permille", type=int, default=20, help="must match what you use for training")
    ap.add_argument("--models", default="rec,det,table,layout")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--max-minutes", type=float, default=30.0, help="time budget per model")
    ap.add_argument("--rec-lines", type=int, default=128)
    ap.add_argument("--rec-batch", type=int, default=16)
    ap.add_argument("--rec-steps", type=int, default=3000)
    ap.add_argument("--rec-lr", type=float, default=5e-4)
    ap.add_argument("--det-pages", type=int, default=8)
    ap.add_argument("--det-long-side", type=int, default=1024)
    ap.add_argument("--det-steps", type=int, default=1500)
    ap.add_argument("--det-lr", type=float, default=1e-3)
    ap.add_argument("--table-n", type=int, default=16)
    ap.add_argument("--table-size", type=int, default=640, help="letterbox size S (use the same for training)")
    ap.add_argument("--table-steps", type=int, default=1500)
    ap.add_argument("--table-lr", type=float, default=1e-3)
    ap.add_argument("--table-ss-max", type=float, default=0.5,
                    help="max scheduled-sampling probability (0 disables it)")
    ap.add_argument("--table-ss-warmup", type=int, default=1000,
                    help="steps to ramp ss_prob from 0 to --table-ss-max")
    ap.add_argument("--table-beam", type=int, default=1,
                    help="beam size at eval time (1 = greedy)")
    ap.add_argument("--layout-pages", type=int, default=8)
    ap.add_argument("--layout-min-size", type=int, default=1024)
    ap.add_argument("--layout-max-size", type=int, default=1600)
    ap.add_argument("--layout-steps", type=int, default=1500)
    ap.add_argument("--layout-lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(0)
    amp = Autocast(device, args.amp)
    index = DataIndex(os.path.join(args.data, args.labels), args.val_permille)
    print(f"[setup] device {device} | amp {amp.dtype if amp.on else 'off'} | pretrained {args.pretrained} | {index.summary()}")
    tests = {"rec": test_rec, "det": test_det, "table": test_table, "layout": test_layout}
    report = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n===== {name} =====")
        try:
            report[name] = tests[name](index, args, device, amp)
        except Exception as e:  # report and continue with the other models
            import traceback
            traceback.print_exc()
            report[name] = {"status": "ERROR", "reason": f"{type(e).__name__}: {e}"}
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n===== summary =====")
    for name, r in report.items():
        keys = [k for k in r if k not in ("status", "examples", "reason")]
        detail = ", ".join(f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys)
        print(f"{name:7s} {r['status']:5s} {detail}" + (f"  ({r['reason']})" if r.get("reason") else ""))
    out = os.path.join(index.root, "overfit_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report -> {out}")
    sys.exit(0 if all(r["status"] in ("PASS", "SKIP") for r in report.values()) else 1)


if __name__ == "__main__":
    main()