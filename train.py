"""
Full training for all four models, sized for a 6 GB GPU by default.

    python train.py --model rec    --data /path/to/data_v5
    python train.py --model det    --data /path/to/data_v5
    python train.py --model table  --data /path/to/data_v5
    python train.py --model layout --data /path/to/data_v5
    tensorboard --logdir runs           # then open http://localhost:6006

What it does
- LR: linear warmup (from 1e-3 x lr) then cosine to lr/50, as torch SequentialLR (state saved in checkpoints).
- EMA of the weights, updated after every optimizer step; validation, best.pt selection and the exported weights
  all use the EMA model (ema.module). --eval-raw also evaluates the raw model so you can compare.
- AdamW without weight decay on biases / norm / embedding parameters; gradient clipping; bf16 or fp16 autocast;
  optional gradient accumulation (--accum). A batch that runs out of GPU memory is skipped and counted.
- Checkpoints in runs/<model>/<run>/checkpoints/: last.pt (every --ckpt-every steps, every --ckpt-minutes, on
  Ctrl+C and on SIGTERM), best.pt (best validation metric), step_XXXXXXX.pt (last --keep kept), and
  best_ema_weights.pt (EMA weights + config only, for inference).
- Auto-resume: rerunning the same command (same --model and --run) continues from last.pt. --fresh starts over.
- Logging: TensorBoard (train/*, val/*, sample predictions) and metrics.jsonl (one JSON line per log/eval).

Validation metrics (on a fixed subset of the validation split, so numbers are comparable across evals)
  rec    : CER overall and per character class (Devanagari letters, matras/virama/anusvara marks, Devanagari
           digits, Latin, ASCII digits), line accuracy. Best = lowest CER.
  det    : precision / recall / F1 at IoU 0.5 for several unclip ratios. Best = highest F1.
  table  : exact structure, token similarity, cell-box IoU, teacher-forced token accuracy. Best = exact structure.
  layout : mAP@0.5, per-class AP, recall / precision at score 0.5. Best = mAP@0.5.
"""
import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (DataIndex, DetDataset, LayoutDataset, RecConfig, RecDataset, RecStream, TableDataset,
                  WidthBucketBatchSampler, check_layout_types, det_collate, layout_collate, load_or_build_charset,
                  make_table_collate, rec_collate, usable_lines)
from detector_dbnet import DBNetPP, _quad_iou, db_loss, db_postprocess
from layout_detector import LAYOUT_CLASSES, build_layout_detector
from recognizer_svtr import CTCCodec, SVTRRecognizer, ctc_loss_fn
from table_slanet import TableStructureModel, TableVocab, table_loss

try:
    from rapidfuzz.distance import Levenshtein as _RF
except ImportError:  # pip install rapidfuzz: ~100x faster validation for rec/table
    _RF = None

# --------------------------------------------------------------------------------------
# Defaults per model (6 GB GPU). Every key can be overridden on the command line.
# --------------------------------------------------------------------------------------
DEFAULTS = {
    "rec": dict(steps=200_000, warmup=2_000, lr=5e-4, wd=0.05, batch=64, max_batch_pixels=40_000, accum=1,
                ema_decay=0.9995, eval_every=5_000, ckpt_every=5_000, val_n=5_000, log_every=50,
                drop_path=0.1, dropout=0.1, rec_height=48, rec_max_width=2560, buffer_lines=2048),
    "det": dict(steps=60_000, warmup=1_000, lr=1e-3, wd=1e-4, batch=4, accum=1, ema_decay=0.999,
                eval_every=2_000, ckpt_every=2_000, val_n=200, log_every=50, det_long_side=1600, det_crop=960,
                unclip_ratios=[1.2, 1.5, 1.8, 2.0]),
    "table": dict(steps=80_000, warmup=1_000, lr=3e-4, wd=1e-4, batch=8, accum=1, ema_decay=0.999,
                  eval_every=2_000, ckpt_every=2_000, val_n=1_000, log_every=50, table_size=640),
    "layout": dict(steps=60_000, warmup=1_000, lr=1e-4, wd=1e-4, batch=2, accum=1, ema_decay=0.999,
                   eval_every=2_000, ckpt_every=2_000, val_n=300, log_every=50, layout_min_size=1024,
                   layout_max_size=1600),
}
# Early stopping: stop after `patience` evals with no gain larger than `min_delta` in the primary metric.
EARLY_STOP = {"rec": dict(patience=4, min_delta=0.0005),     # CER, absolute (0.05 percentage points)
              "det": dict(patience=4, min_delta=0.001),      # F1
              "table": dict(patience=4, min_delta=0.002),    # exact-structure rate
              "layout": dict(patience=4, min_delta=0.002)}   # mAP@0.5
for _m, _es in EARLY_STOP.items():
    DEFAULTS[_m].update(_es)
PRIMARY = {"rec": ("cer", "min"), "det": ("f1", "max"), "table": ("exact_structure", "max"),
           "layout": ("map50", "max")}


# --------------------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------------------
class EMA:
    """
    Exponential moving average of model weights. ema.update(model) after every optimizer step; evaluate ema.module.
    Floating-point parameters AND buffers (BatchNorm running stats) are averaged; integer buffers are copied.
    The decay ramps up early (min(decay, (1+n)/(10+n))) so the EMA does not stay stuck near the random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0
        self._pairs = None

    def _build_pairs(self, model):
        src, dst = model.state_dict(), self.module.state_dict()
        f_src, f_dst, i_src, i_dst = [], [], [], []
        for k, v in dst.items():
            (f_dst if v.dtype.is_floating_point else i_dst).append(v)
            (f_src if v.dtype.is_floating_point else i_src).append(src[k])
        self._pairs = (f_src, f_dst, i_src, i_dst)

    @torch.no_grad()
    def update(self, model: nn.Module):
        if self._pairs is None:
            self._build_pairs(model)
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        f_src, f_dst, i_src, i_dst = self._pairs
        torch._foreach_lerp_(f_dst, [s.detach() for s in f_src], 1.0 - d)
        for s, t in zip(i_src, i_dst):
            t.copy_(s)

    def state_dict(self):
        return {"module": self.module.state_dict(), "decay": self.decay, "updates": self.updates}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self.updates = sd.get("updates", 0)
        self._pairs = None


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
class Autocast:
    def __init__(self, device: str, enabled: bool):
        self.on = enabled and device.startswith("cuda")
        self.dtype = torch.bfloat16 if self.on and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.on and self.dtype == torch.float16)

    def ctx(self):
        return torch.autocast("cuda", dtype=self.dtype, enabled=self.on)


class Logger:
    """TensorBoard if installed, plus metrics.jsonl always."""

    def __init__(self, run_dir: str):
        self.f = open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(run_dir, "tb"))
        except ImportError:
            self.tb = None
            print("[log] tensorboard not installed (pip install tensorboard); writing metrics.jsonl only")

    def scalars(self, step: int, d: Dict[str, float], prefix: str):
        clean = {k: float(v) for k, v in d.items() if isinstance(v, (int, float, np.floating, np.integer))}
        if self.tb:
            for k, v in clean.items():
                self.tb.add_scalar(f"{prefix}/{k}", v, step)
        self.f.write(json.dumps({"step": step, "split": prefix, **clean}, ensure_ascii=False) + "\n")
        self.f.flush()

    def text(self, step, tag, s):
        if self.tb:
            self.tb.add_text(tag, s, step)

    def image(self, step, tag, img_hw):
        if self.tb:
            self.tb.add_image(tag, img_hw[None], step, dataformats="CHW")

    def close(self):
        if self.tb:
            self.tb.close()
        self.f.close()


def param_groups(model: nn.Module, wd: float):
    """No weight decay on biases, norm weights and embeddings (1-D params / embedding tables)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "embed" in n.lower() else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Closed-form warmup + cosine (to 1/50 of peak). Depends only on the step, so resuming is exact."""
    warmup = max(1, min(warmup, total - 1))
    if step < warmup:
        return 1e-3 + (1 - 1e-3) * step / warmup
    p = min(1.0, (step - warmup) / max(total - warmup, 1))
    return 1 / 50 + (1 - 1 / 50) * 0.5 * (1 + math.cos(math.pi * p))


def make_scheduler(opt, lr: float, warmup: int, total: int, start_step: int = 0):
    """
    LambdaLR instead of SequentialLR(LinearLR, CosineAnnealingLR): those are 'chainable' (each lr is computed
    from the optimizer's CURRENT lr), so after a resume that reset the optimizer lr the schedule continued from a
    wrong value (det resumed at ~1e-6 and stayed there). The lambda form is recomputed from the base lr every step.
    """
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup, total))
    if start_step:
        sched.last_epoch = start_step
        for g, base in zip(opt.param_groups, sched.base_lrs):
            g["lr"] = base * lr_factor(start_step, warmup, total)
        sched._last_lr = [g["lr"] for g in opt.param_groups]
    return sched


def edit_distance(a, b) -> int:
    if _RF is not None:
        return _RF.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def gt_char_errors(gt: str, pred: str):
    """(set of gt indices that were substituted or deleted, number of inserted chars)."""
    if _RF is not None:
        bad, ins = set(), 0
        for op in _RF.editops(gt, pred):
            if op.tag == "insert":
                ins += 1
            else:
                bad.add(op.src_pos)
        return bad, ins
    n, m = len(gt), len(pred)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]))
    bad, ins, i, j = set(), 0, n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]):
            if gt[i - 1] != pred[j - 1]:
                bad.add(i - 1)
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            bad.add(i - 1)
            i -= 1
        else:
            ins += 1
            j -= 1
    return bad, ins


def char_class(c: str) -> str:
    o = ord(c)
    if 0x0966 <= o <= 0x096F:
        return "dev_digit"
    if (0x0900 <= o <= 0x0903) or (0x093A <= o <= 0x094F and o != 0x093D) or 0x0955 <= o <= 0x0957 \
            or 0x0962 <= o <= 0x0963:
        return "dev_mark"  # candrabindu, anusvara, visarga, nukta, matras, virama
    if 0x0900 <= o <= 0x097F:
        return "dev_letter"
    if c.isascii() and c.isalpha():
        return "latin"
    if c.isascii() and c.isdigit():
        return "ascii_digit"
    return "space" if c.isspace() else "other"


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def to_dev(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_dev(v, device) for v in x)
    return x


def subset(ids: np.ndarray, n: int) -> np.ndarray:
    """Evenly spaced, deterministic subset (covers many documents, same every eval)."""
    if len(ids) <= n:
        return ids
    return ids[np.linspace(0, len(ids) - 1, n).round().astype(int)]


# --------------------------------------------------------------------------------------
# Tasks: model, data, loss, evaluation
# --------------------------------------------------------------------------------------
class Task:
    def __init__(self, cfg: Dict, index: DataIndex, device: str, amp: Autocast):
        self.cfg, self.index, self.device, self.amp = cfg, index, device, amp

    extra_state: Dict = {}


class RecTask(Task):
    def __init__(self, *a):
        super().__init__(*a)
        c = self.cfg
        charset = load_or_build_charset(self.index, os.path.join(self.index.root, "charset.json"))
        self.codec = CTCCodec(charset)
        self.extra_state = {"charset": charset}
        self.rcfg = RecConfig(height=c["rec_height"], max_width=c["rec_max_width"])

    def build_model(self):
        c = self.cfg
        return SVTRRecognizer(self.codec.num_classes, in_height=c["rec_height"], drop_path=c["drop_path"],
                              dropout=c["dropout"])

    def train_loader(self, seed):
        ds = RecStream(self.index, "train", self.codec, self.rcfg, batch_size=self.cfg["batch"],
                       buffer_lines=self.cfg["buffer_lines"], max_batch_pixels=self.cfg["max_batch_pixels"],
                       augment=True, seed=seed)
        # persistent_workers=False: workers are re-created each epoch so they see the new epoch number (new order)
        return DataLoader(ds, batch_size=None, num_workers=self.cfg["workers"], pin_memory=self.device != "cpu",
                          persistent_workers=False, prefetch_factor=4 if self.cfg["workers"] else None)

    def val_loader(self):
        ids = subset(usable_lines(self.index, "val", self.codec, self.rcfg), self.cfg["val_n"])
        ds = RecDataset(self.index, "val", self.codec, self.rcfg, augment=False, line_ids=ids, verbose=False)
        sampler = WidthBucketBatchSampler(ds.natural_widths(), 32, shuffle=False)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=rec_collate, num_workers=min(2, self.cfg["workers"]))

    def loss(self, model, b):
        logits = model(b["images"], b["widths"])
        loss, st = ctc_loss_fn(logits, b["targets"], b["target_lengths"], SVTRRecognizer.out_lengths(b["widths"]))
        return loss, {"ctc_skipped": st["skipped"]}, len(b["texts"])

    @torch.no_grad()
    def evaluate(self, model, loader, log: Logger, step: int, tag: str):
        model.eval()
        n_err, n_ref, lines_ok, n_lines = 0, 0, 0, 0
        cls_err, cls_tot = {}, {}
        worst = []
        for b in loader:
            b = to_dev(b, self.device)
            with self.amp.ctx():
                logits = model(b["images"], b["widths"])
            dec = self.codec.decode_greedy(logits.float().cpu(), SVTRRecognizer.out_lengths(b["widths"]).cpu())
            for d, gt in zip(dec, b["texts"]):
                pred = d["text"]
                e = edit_distance(gt, pred)
                n_err, n_ref, n_lines = n_err + e, n_ref + len(gt), n_lines + 1
                lines_ok += e == 0
                bad, _ = gt_char_errors(gt, pred)
                for i, ch in enumerate(gt):
                    k = char_class(ch)
                    cls_tot[k] = cls_tot.get(k, 0) + 1
                    cls_err[k] = cls_err.get(k, 0) + (i in bad)
                if e:
                    worst.append((e / max(len(gt), 1), gt, pred))
        m = {"cer": n_err / max(n_ref, 1), "line_acc": lines_ok / max(n_lines, 1), "lines": n_lines}
        for k, t in cls_tot.items():
            if k not in ("space", "other") or t > 0:
                m[f"cer_{k}"] = cls_err.get(k, 0) / t
        worst.sort(key=lambda x: -x[0])
        rows = ["| CER | ground truth | prediction |", "|---|---|---|"] + \
               [f"| {r:.2f} | {g} | {p} |" for r, g, p in worst[:15]]
        log.text(step, f"{tag}/worst_lines", "\n".join(rows))
        return m


class DetTask(Task):
    def build_model(self):
        return DBNetPP(pretrained_backbone=self.cfg["pretrained"])

    def train_loader(self, seed):
        ds = DetDataset(self.index, "train", long_side=self.cfg["det_long_side"], crop=self.cfg["det_crop"], augment=True)
        return DataLoader(ds, batch_size=self.cfg["batch"], shuffle=True, drop_last=True, collate_fn=det_collate,
                          num_workers=self.cfg["workers"], pin_memory=self.device != "cpu",
                          persistent_workers=self.cfg["workers"] > 0)

    def val_loader(self):
        pages = self.index.pages("val")
        pages = subset(pages[self.index.page_nlines[pages] > 0], self.cfg["val_n"])
        ds = DetDataset(self.index, "val", long_side=self.cfg["det_long_side"], crop=None, augment=False, pages=pages)
        return DataLoader(ds, batch_size=1, collate_fn=det_collate, num_workers=min(2, self.cfg["workers"]))

    def loss(self, model, b):
        out = model(b["image"])
        loss, parts = db_loss(out, {k: b[k] for k in ("gt", "gt_mask", "thresh_map", "thresh_mask")})
        return loss, parts, b["image"].shape[0]

    @torch.no_grad()
    def evaluate(self, model, loader, log, step, tag):
        model.eval()
        ratios = self.cfg["unclip_ratios"]
        acc = {r: [0, 0, 0] for r in ratios}  # tp, n_pred, n_gt
        first = None
        for b in loader:
            img = b["image"].to(self.device)
            with self.amp.ctx():
                prob = model(img)["prob"].float()[0, 0].cpu().numpy()
            if first is None:
                first = prob
            gts = [np.asarray(q, np.float32) for q in b["quads"][0]]
            for r in ratios:
                preds, _ = db_postprocess(prob, unclip_ratio=r)
                used, tp = set(), 0
                for g in gts:
                    best, bj = 0.0, -1
                    for j, p in enumerate(preds):
                        if j not in used:
                            iou = _quad_iou(g, p)
                            if iou > best:
                                best, bj = iou, j
                    if best >= 0.5:
                        tp += 1
                        used.add(bj)
                acc[r][0] += tp; acc[r][1] += len(preds); acc[r][2] += len(gts)
        m = {}
        best = (-1.0, None)
        for r, (tp, npred, ngt) in acc.items():
            p, rc = tp / max(npred, 1), tp / max(ngt, 1)
            f1 = 2 * p * rc / max(p + rc, 1e-9)
            m[f"f1_unclip_{r}"] = f1
            if f1 > best[0]:
                best = (f1, r)
                m.update({"precision": p, "recall": rc})
        m["f1"], m["best_unclip_ratio"] = best
        if first is not None:
            log.image(step, f"{tag}/prob_map", np.clip(first, 0, 1).astype(np.float32))
        return m


class TableTask(Task):
    def __init__(self, *a):
        super().__init__(*a)
        self.vocab = TableVocab()
        self.extra_state = {"table_vocab_max_span": self.vocab.max_span, "table_size": self.cfg["table_size"]}

    def build_model(self):
        return TableStructureModel(self.vocab, pretrained_backbone=self.cfg["pretrained"])

    def train_loader(self, seed):
        ds = TableDataset(self.index, "train", self.vocab, S=self.cfg["table_size"])
        return DataLoader(ds, batch_size=self.cfg["batch"], shuffle=True, drop_last=True,
                          collate_fn=make_table_collate(self.vocab.pad), num_workers=self.cfg["workers"],
                          pin_memory=self.device != "cpu", persistent_workers=self.cfg["workers"] > 0)

    def val_loader(self):
        pages = self.index.pages("val")
        pages = pages[self.index.page_ntables[pages] > 0]
        pairs = [(int(p), t) for p in pages for t in range(int(self.index.page_ntables[p]))]
        pairs = [pairs[i] for i in subset(np.arange(len(pairs)), self.cfg["val_n"])]
        ds = TableDataset(self.index, "val", self.vocab, S=self.cfg["table_size"], pairs=pairs)
        return DataLoader(ds, batch_size=8, collate_fn=make_table_collate(self.vocab.pad),
                          num_workers=min(2, self.cfg["workers"]))

    def loss(self, model, b):
        lg, bx = model(b["image"], b["tokens"])
        loss, parts = table_loss(lg, bx, b["tokens"], b["boxes"], b["box_mask"], self.vocab.pad)
        return loss, parts, b["image"].shape[0]

    @torch.no_grad()
    def evaluate(self, model, loader, log, step, tag):
        model.eval()
        V = self.vocab
        n = exact = right = total = 0
        sims, ious = [], []
        for b in loader:
            if b is None:
                continue
            b = to_dev(b, self.device)
            with self.amp.ctx():
                lg, _ = model(b["image"], b["tokens"])
                preds = model.predict(b["image"], max_len=b["tokens"].shape[1] + 20)
            tgt = b["tokens"][:, 1:]
            msk = tgt != V.pad
            right += int(((lg.argmax(-1) == tgt) & msk).sum())
            total += int(msk.sum())
            for i, pr in enumerate(preds):
                ids = b["tokens"][i].tolist()
                gt = [V.tokens[t] for t in ids[1:ids.index(V.eos)]]
                n += 1
                d = edit_distance(gt, pr["tokens"])
                sims.append(1 - d / max(len(gt), len(pr["tokens"]), 1))
                if pr["tokens"] == gt:
                    exact += 1
                    g = b["boxes"][i][b["box_mask"][i]].float().cpu().numpy()
                    g = np.stack([g[:, 0] - g[:, 2] / 2, g[:, 1] - g[:, 3] / 2, g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2], 1)
                    ious += [box_iou(a, c) for a, c in zip(pr["cell_boxes"], g)]
        return {"exact_structure": exact / max(n, 1), "token_similarity": float(np.mean(sims)) if sims else 0.0,
                "cell_box_iou": float(np.mean(ious)) if ious else 0.0, "tf_acc": right / max(total, 1), "tables": n}


class LayoutTask(Task):
    def build_model(self):
        check_layout_types(self.index)
        return build_layout_detector(pretrained_backbone=self.cfg["pretrained"], min_size=self.cfg["layout_min_size"],
                                     max_size=self.cfg["layout_max_size"])

    def train_loader(self, seed):
        ds = LayoutDataset(self.index, "train")
        return DataLoader(ds, batch_size=self.cfg["batch"], shuffle=True, drop_last=True, collate_fn=layout_collate,
                          num_workers=self.cfg["workers"], pin_memory=self.device != "cpu",
                          persistent_workers=self.cfg["workers"] > 0)

    def val_loader(self):
        pages = self.index.pages("val")
        pages = subset(pages[self.index.page_nlayout[pages] > 0], self.cfg["val_n"])
        return DataLoader(LayoutDataset(self.index, "val", pages=pages), batch_size=2, collate_fn=layout_collate,
                          num_workers=min(2, self.cfg["workers"]))

    def loss(self, model, b):
        ims, tgs = b
        parts = model(ims, tgs)
        return sum(parts.values()), {k: v.item() for k, v in parts.items()}, len(ims)

    @torch.no_grad()
    def evaluate(self, model, loader, log, step, tag):
        model.eval()
        dets = {c: [] for c in range(1, len(LAYOUT_CLASSES) + 1)}  # (score, is_tp)
        n_gt = {c: 0 for c in dets}
        hit = tot = npred = 0
        for ims, tgs in loader:
            ims = [x.to(self.device) for x in ims]
            with self.amp.ctx():
                outs = model(ims)
            for o, t in zip(outs, tgs):
                gb, gl = t["boxes"].tolist(), t["labels"].tolist()
                for c in gl:
                    n_gt[c] += 1
                pb, pl, ps = o["boxes"].float().cpu().tolist(), o["labels"].cpu().tolist(), o["scores"].float().cpu().tolist()
                order = np.argsort(ps)[::-1]
                used = set()
                for j in order:  # AP matching: highest score first, same class, IoU >= 0.5
                    best, bi = 0.0, -1
                    for i, (g, c) in enumerate(zip(gb, gl)):
                        if c == pl[j] and i not in used:
                            iou = box_iou(pb[j], g)
                            if iou > best:
                                best, bi = iou, i
                    tp = best >= 0.5
                    if tp:
                        used.add(bi)
                    dets[pl[j]].append((ps[j], tp))
                    if ps[j] >= 0.5:
                        npred += 1
                        hit += tp
                tot += len(gb)
        aps = {}
        for c, d in dets.items():
            if n_gt[c] == 0:
                continue
            d.sort(key=lambda x: -x[0])
            tps = np.cumsum([x[1] for x in d]) if d else np.zeros(0)
            fps = np.cumsum([not x[1] for x in d]) if d else np.zeros(0)
            rec = tps / n_gt[c]
            prec = tps / np.maximum(tps + fps, 1)
            mrec = np.concatenate([[0], rec, [1]])
            mpre = np.concatenate([[0], prec, [0]])
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
            aps[LAYOUT_CLASSES[c - 1]] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        m = {"map50": float(np.mean(list(aps.values()))) if aps else 0.0,
             "recall": hit / max(tot, 1), "precision": hit / max(npred, 1)}
        m.update({f"ap50_{k}": v for k, v in aps.items()})
        return m


TASKS = {"rec": RecTask, "det": DetTask, "table": TableTask, "layout": LayoutTask}


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def atomic_save(obj, path) -> bool:
    """
    Write to <path>.tmp then rename, so an existing checkpoint is never left half-written. If the write fails
    (usually a full disk) the partial .tmp is deleted, a warning is printed and training CONTINUES: losing one
    checkpoint is better than killing a run that has many hours in it. The previous file at `path` stays intact.
    """
    import shutil
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return True
    except (RuntimeError, OSError) as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free / 2 ** 30
        print(f"[warn] could not save {path} ({type(e).__name__}); {free:.2f} GB free on that disk. "
              f"Training continues; free some space (old step_*.pt files) so the next save works.", flush=True)
        return False


class Checkpointer:
    def __init__(self, run_dir: str, keep: int):
        self.dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.dir, exist_ok=True)
        self.keep = keep

    def save(self, state: Dict, step: int, kind: str):
        if kind == "best":
            # inference weights first: they are small and the file you actually need
            atomic_save({"model": state["ema"]["module"] if state.get("ema") else state["model"],
                         "config": state["config"], "extra": state["extra"], "step": step,
                         "val": state.get("best_metrics")},
                        os.path.join(self.dir, "best_ema_weights.pt"))
            atomic_save(state, os.path.join(self.dir, "best.pt"))
            return
        atomic_save(state, os.path.join(self.dir, "last.pt"))
        if kind == "periodic" and self.keep > 0:
            # delete the oldest step files BEFORE writing a new one, so disk use never peaks above `keep` files
            olds = sorted(f for f in os.listdir(self.dir) if f.startswith("step_") and f.endswith(".pt"))
            for f in olds[:-(self.keep - 1)] if self.keep > 1 else olds:
                os.remove(os.path.join(self.dir, f))
            atomic_save(state, os.path.join(self.dir, f"step_{step:07d}.pt"))


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
def infinite(loader_fn: Callable[[int], DataLoader], start_epoch: int):
    epoch = start_epoch
    while True:
        loader = loader_fn(epoch)
        for b in loader:
            yield epoch, b
        epoch += 1


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(TASKS))
    ap.add_argument("--data", required=True, help="folder with labels.jsonl")
    ap.add_argument("--labels", default="labels.jsonl")
    ap.add_argument("--run", default="main", help="run name; output goes to <out>/<model>/<run>")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt and start over")
    ap.add_argument("--resume", default=None, help="resume from a specific checkpoint file")
    ap.add_argument("--init", default=None, help="start from these weights (model only, e.g. a best_ema_weights.pt)")
    ap.add_argument("--val-permille", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--eval-raw", action="store_true", help="also evaluate the non-EMA model")
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--keep", type=int, default=3, help="step_*.pt checkpoints to keep")
    ap.add_argument("--ckpt-minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-going", action="store_true",
                    help="continue a run that already early-stopped (resets the patience counter)")
    # overrides of DEFAULTS (None = use the model's default)
    for k, t in [("steps", int), ("warmup", int), ("lr", float), ("wd", float), ("batch", int), ("accum", int),
                 ("ema_decay", float), ("eval_every", int), ("ckpt_every", int), ("val_n", int), ("log_every", int),
                 ("max_batch_pixels", int), ("buffer_lines", int), ("drop_path", float), ("dropout", float),
                 ("rec_max_width", int), ("det_long_side", int), ("det_crop", int), ("table_size", int),
                 ("layout_min_size", int), ("layout_max_size", int), ("patience", int), ("min_delta", float)]:
        ap.add_argument("--" + k.replace("_", "-"), type=t, default=None)
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = dict(DEFAULTS[a.model])
    for k in list(cfg) + ["max_batch_pixels", "buffer_lines"]:
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    cfg.update(model=a.model, workers=a.workers, pretrained=a.pretrained, ema=a.ema, amp=a.amp, clip=a.clip,
               seed=a.seed, val_permille=a.val_permille, data=os.path.abspath(a.data))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = a.model != "rec"  # rec widths vary every batch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = os.path.join(a.out, a.model, a.run)
    os.makedirs(run_dir, exist_ok=True)
    ckpt = Checkpointer(run_dir, a.keep)
    resume = a.resume or (None if a.fresh else (os.path.join(ckpt.dir, "last.pt")
                                                if os.path.exists(os.path.join(ckpt.dir, "last.pt")) else None))
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if state is not None:
        old = state["config"]
        if old.get("data") and old["data"] != cfg["data"]:
            # the dataset / runs folder was moved: the stored path is only a record, the new one is used and
            # written into every checkpoint saved from now on
            print(f"[resume] data folder moved: {old['data']} -> {cfg['data']} (checkpoints updated on next save)")
        changed = {k: (old.get(k), v) for k, v in cfg.items()
                   if k in old and old.get(k) != v and k not in ("workers", "data")}
        if changed:
            print(f"[resume] config differs from the checkpoint (using NEW values): {changed}")

    index = DataIndex(os.path.join(a.data, a.labels), a.val_permille)
    amp = Autocast(device, a.amp)
    task = TASKS[a.model](cfg, index, device, amp)
    model = task.build_model().to(device)

    # --- --init: transfer weights tensor-by-tensor, skipping shape mismatches ---
    if a.init and state is None:
        sd = torch.load(a.init, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        target = model.state_dict()
        keep, skipped = {}, []
        for k, v in sd.items():
            if k in target and target[k].shape == v.shape:
                keep[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(target[k].shape) if k in target else None))
        if skipped:
            print(f"[init] {len(keep)}/{len(sd)} tensors transferred; {len(skipped)} skipped "
                  f"(shape mismatch or not in model):")
            for k, src, dst in skipped[:20]:
                print(f"        {k}: ckpt{src} -> model{dst}")
            if len(skipped) > 20:
                print(f"        ... and {len(skipped) - 20} more")
        else:
            print(f"[init] all {len(keep)} tensors transferred cleanly")
        missing = model.load_state_dict(keep, strict=False)
        if missing.missing_keys:
            print(f"[init] model params still at random init: {missing.missing_keys}")

    opt = torch.optim.AdamW(param_groups(model, cfg["wd"]), lr=cfg["lr"], betas=(0.9, 0.999))
    ema = EMA(model, cfg["ema_decay"]) if a.ema else None

    step, epoch, best_val, best_metrics = 0, 0, None, None
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if ema is not None and state.get("ema"):
            ema.load_state_dict(state["ema"])
        if state.get("scaler"):
            amp.scaler.load_state_dict(state["scaler"])
        step, epoch = state["step"], state.get("epoch", 0) + 1  # streaming loaders restart at a fresh epoch
        best_val, best_metrics = state.get("best_val"), state.get("best_metrics")
        print(f"[resume] {resume}: step {step}, best {PRIMARY[a.model][0]} = {best_val}")
    for g in opt.param_groups:  # schedule always starts from the CURRENT config's peak lr
        g["lr"] = g["initial_lr"] = cfg["lr"]
    # Closed-form schedule: on resume just jump to `step` (works whether or not lr/warmup/steps changed).
    sched = make_scheduler(opt, cfg["lr"], cfg["warmup"], cfg["steps"], start_step=step)
    if state is not None:
        print(f"[resume] lr at step {step:,} = {opt.param_groups[0]['lr']:.2e}")

    # Early stopping state (patience = number of evals without a significant improvement)
    es = {"ref": None, "bad": 0, "stopped": False}
    if state is not None:
        es.update(state.get("early_stop", {}))
        if es["stopped"] and not a.keep_going:
            print(f"[done] this run already early-stopped at step {step:,} (best {PRIMARY[a.model][0]} = {best_val}). "
                  f"Skipping. Use --keep-going to train further or --fresh to restart.")
            return
        if a.keep_going:
            es.update(stopped=False, bad=0)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    log = Logger(run_dir)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[setup] {a.model} | {n_params:.1f}M params | device {device} | amp {amp.dtype if amp.on else 'off'} | "
          f"ema {cfg['ema_decay'] if ema else 'off'} | {index.summary()}")
    print(f"[setup] run dir {run_dir} | steps {cfg['steps']:,} | batch {cfg['batch']} x accum {cfg['accum']} | "
          f"lr {cfg['lr']} | tensorboard --logdir {a.out}")
    val_loader = task.val_loader()
    key, mode = PRIMARY[a.model]

    def full_state():
        return {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "ema": ema.state_dict() if ema else None, "scaler": amp.scaler.state_dict(), "step": step,
                "epoch": epoch, "best_val": best_val, "best_metrics": best_metrics, "config": cfg,
                "extra": task.extra_state, "early_stop": dict(es)}

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    def run_eval():
        nonlocal best_val, best_metrics
        t0 = time.time()
        target = ema.module if ema else model
        m = task.evaluate(target, val_loader, log, step, "val")
        m["eval_seconds"] = time.time() - t0
        log.scalars(step, m, "val")
        if a.eval_raw and ema:
            log.scalars(step, task.evaluate(model, val_loader, log, step, "val_raw"), "val_raw")
        model.train()
        better = best_val is None or (m[key] < best_val if mode == "min" else m[key] > best_val)
        # Patience counts only SIGNIFICANT gains (> min_delta); tiny gains still update best.pt but don't reset it.
        d = cfg["min_delta"]
        significant = es["ref"] is None or (m[key] < es["ref"] - d if mode == "min" else m[key] > es["ref"] + d)
        if significant:
            es["ref"], es["bad"] = m[key], 0
        else:
            es["bad"] += 1
        shown = ", ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float) and not k.startswith(("ap50_", "f1_unclip")))
        tag = "  (new best)" if better else ""
        if cfg["patience"] > 0:
            tag += f"  [patience {es['bad']}/{cfg['patience']}]"
        print(f"[val] step {step:,}: {shown}{tag}", flush=True)
        if better:
            best_val, best_metrics = m[key], m
            ckpt.save(full_state(), step, "best")
        if cfg["patience"] > 0 and es["bad"] >= cfg["patience"]:
            es["stopped"] = True
            stop["flag"] = True
            print(f"[early-stop] no {key} gain > {d} in {cfg['patience']} evals; stopping at step {step:,}.", flush=True)

    model.train()
    data = infinite(task.train_loader, epoch)
    t_log, t_ckpt = time.time(), time.time()
    agg: Dict[str, float] = {}
    n_seen, oom, gn_sum = 0, 0, 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        while step < cfg["steps"] and not stop["flag"]:
            opt.zero_grad(set_to_none=True)
            ok_micro = 0
            for _ in range(cfg["accum"]):
                epoch, b = next(data)
                if b is None:  # table batch with only unusable samples
                    continue
                b = to_dev(b, device)
                try:
                    with amp.ctx():
                        loss, parts, n = task.loss(model, b)
                    if not torch.isfinite(loss):
                        print(f"[warn] non-finite loss at step {step}; batch skipped")
                        continue
                    amp.scaler.scale(loss / cfg["accum"]).backward()
                except torch.OutOfMemoryError:
                    oom += 1
                    opt.zero_grad(set_to_none=True)
                    del b
                    torch.cuda.empty_cache()
                    print(f"[warn] out of GPU memory at step {step}; batch skipped ({oom} so far). "
                          f"If this repeats, lower --batch" + (" or --max-batch-pixels" if a.model == "rec" else ""))
                    continue
                ok_micro += 1
                n_seen += n
                agg["loss"] = agg.get("loss", 0.0) + loss.item()
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                agg["_n"] = agg.get("_n", 0) + 1
            if ok_micro == 0:
                continue
            amp.scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            amp.scaler.step(opt)
            amp.scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            step += 1
            gn_sum += float(gnorm)

            if step % cfg["log_every"] == 0:
                dt = time.time() - t_log
                nb = max(agg.pop("_n", 1), 1)
                out = {k: v / nb for k, v in agg.items()}
                out.update(lr=opt.param_groups[0]["lr"], grad_norm=gn_sum / cfg["log_every"], samples_per_s=n_seen / dt,
                           steps_per_s=cfg["log_every"] / dt, epoch=epoch, oom_skipped=oom)
                if device.startswith("cuda"):
                    out["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                    torch.cuda.reset_peak_memory_stats()
                log.scalars(step, out, "train")
                eta = (cfg["steps"] - step) / max(out["steps_per_s"], 1e-9) / 3600
                print(f"[train] step {step:,}/{cfg['steps']:,} loss {out['loss']:.4f} lr {out['lr']:.2e} "
                      f"{out['samples_per_s']:.0f} samples/s" + (f" mem {out['gpu_mem_gb']:.1f}GB" if 'gpu_mem_gb' in out else "")
                      + f" eta {eta:.1f}h", flush=True)
                agg, n_seen, gn_sum, t_log = {}, 0, 0.0, time.time()

            if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                t_eval = time.time()
                run_eval()
                t_log += time.time() - t_eval  # keep eval time out of samples/s and eta
            if step % cfg["ckpt_every"] == 0:
                ckpt.save(full_state(), step, "periodic")
                t_ckpt = time.time()
            elif (time.time() - t_ckpt) / 60 > a.ckpt_minutes:
                ckpt.save(full_state(), step, "last")
                t_ckpt = time.time()
    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C: saving last.pt ...")
    finally:
        if step > 0:
            ckpt.save(full_state(), step, "last")
            print(f"[stop] saved {os.path.join(ckpt.dir, 'last.pt')} at step {step:,}. "
                  f"Run the same command again to resume.")
        log.close()
    if step >= cfg["steps"] or es["stopped"]:
        print(f"[done] best {key} = {best_val}. Inference weights: {os.path.join(ckpt.dir, 'best_ema_weights.pt')}")


if __name__ == "__main__":
    main()