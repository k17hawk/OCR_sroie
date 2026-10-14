"""
Data loaders for the OCR pipeline. All loaders read the labels.jsonl written by your generator (format v2);
image paths in each record are relative to the folder that holds labels.jsonl.

    index   = DataIndex("data/labels.jsonl")             # one pass, cached to labels.index.npz
    charset = load_or_build_charset(index, "data/charset.json")
    codec   = CTCCodec(charset)

    rec_train = make_rec_loader(index, "train", codec)         # streaming, width-bucketed, 1 decode per page
    rec_val   = make_rec_loader(index, "val", codec, stream=False, augment=False)
    det_train = make_det_loader(index, "train")
    lay_train = make_layout_loader(index, "train")
    tab_train = make_table_loader(index, "train", TableVocab())

Design
- DataIndex: ONE pass over labels.jsonl stores byte offsets plus small per-page and per-line arrays (sizes,
  CTC step needs, document split, char and layout-type counts) in labels.index.npz. Every dataset is built
  from the arrays, so nothing re-parses 16 GB, and every run sees the same split.
- Split by document_id (crc32 hash, deterministic): no pages of a validation document appear in training.
- Charset: built once from the whole corpus and saved to charset.json. Always load it from that file so index
  order (and your checkpoints) never change between runs. Unknown characters are reported, never dropped.
- Recognition crops use the line QUAD (perspective-rectified, as at inference) with random margins that cover
  what the detector's unclip produces. Lines that would be squeezed, or have too few CTC steps even with the
  largest margin, are filtered at index time and counted by reason.
- Recognition training streams pages: each worker decodes a page once, crops all its lines into a buffer,
  sorts the buffer by width and yields width-bucketed batches (little padding, no per-line page decode).

CLI check (builds index + charset, prints stats, writes preview images to <data>/_preview):
    python data.py --data path/to/folder
"""
import argparse
import json
import math
import os
import random
import zlib
from collections import Counter, OrderedDict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info

from detector_dbnet import make_db_targets, normalize_page, resize_for_detector
from layout_detector import LAYOUT_CLASSES, label_of, layout_targets_from_json
from pipeline import crop_quad
from recognizer_svtr import (CTCCodec, ctc_min_steps, line_to_tensor, nfc, resize_line_uint8, collate_lines,
                             PAD_VALUE)
from table_slanet import DEFAULT_S, TableVocab, collate_tables, table_sample_from_json

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pip install orjson makes the one-time index pass ~2-3x faster
    _loads = json.loads

INDEX_VERSION = 1
SPLITS = {"train": 0, "val": 1}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def clean_text(t: Optional[str]) -> str:
    """The exact label the recognizer is trained on. Used for filtering AND encoding."""
    return nfc(t or "").strip()


def doc_split(document_id: str, val_permille: int) -> int:
    """0 = train, 1 = val. Deterministic across machines and runs."""
    return int(zlib.crc32(str(document_id).encode("utf-8")) % 1000 < val_permille)


def line_quad(ln: Dict) -> np.ndarray:
    if ln.get("quad") is not None:
        return np.asarray(ln["quad"], np.float32).reshape(4, 2)
    x1, y1, x2, y2 = [float(v) for v in ln["bbox"]]
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def quad_size(q: np.ndarray) -> Tuple[float, float]:
    w = max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2]))
    h = max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))
    return float(w), float(h)


_RNGS: Dict[Tuple[int, int], random.Random] = {}


def _worker_rng() -> random.Random:
    """
    One generator per DataLoader worker, created on first use and then ADVANCED on every call. torch gives each
    worker its own initial_seed (new seeds each epoch unless persistent_workers), so workers never repeat each
    other. Creating a fresh Random(seed) per call would return the same numbers for every sample.
    """
    key = (os.getpid(), torch.initial_seed())
    rng = _RNGS.get(key)
    if rng is None:
        rng = _RNGS[key] = random.Random(torch.initial_seed() % (2 ** 63))
    return rng


# --------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------
class DataIndex:
    """
    Byte offsets + per-page/per-line metadata for a labels.jsonl, cached next to it as labels.index.npz.
    Rebuilt automatically when labels.jsonl changes (size/mtime) or val_permille changes.
    """

    def __init__(self, labels_path: str, val_permille: int = 20, rebuild: bool = False, verbose: bool = True):
        self.path = os.path.abspath(labels_path)
        self.root = os.path.dirname(self.path)
        self.cache = os.path.splitext(self.path)[0] + ".index.npz"
        st = os.stat(self.path)
        self._sig = np.array([st.st_size, int(st.st_mtime), val_permille, INDEX_VERSION], np.int64)
        self.verbose = verbose
        if not rebuild and os.path.exists(self.cache):
            z = np.load(self.cache, allow_pickle=False)
            if np.array_equal(z["sig"], self._sig):
                self._from_npz(z)
                self._fh = None
                return
        self._build(val_permille)
        self._fh = None

    # ---- build / load ------------------------------------------------------------------
    def _build(self, val_permille: int):
        if self.verbose:
            print(f"[index] scanning {self.path} (one time) ...")
        P = {k: [] for k in ("offset", "split", "nlines", "ntables", "nlayout", "width", "height")}
        L = {k: [] for k in ("page", "idx", "w", "h", "need", "nchars")}
        chars, types = Counter(), Counter()
        pos, n_bad = 0, 0
        with open(self.path, "rb") as f:
            for raw in f:
                start, pos = pos, pos + len(raw)
                if not raw.strip():
                    continue
                try:
                    rec = _loads(raw)
                except ValueError:
                    n_bad += 1
                    continue
                pi = len(P["offset"])
                lines, tables, layout = rec.get("lines") or [], rec.get("tables") or [], rec.get("layout") or []
                P["offset"].append(start)
                P["split"].append(doc_split(rec.get("document_id", rec.get("image", pi)), val_permille))
                P["nlines"].append(len(lines))
                P["ntables"].append(len(tables))
                P["nlayout"].append(len(layout))
                P["width"].append(int(rec.get("width", 0)))
                P["height"].append(int(rec.get("height", 0)))
                for b in layout:
                    types[b.get("type", "?")] += 1
                for li, ln in enumerate(lines):
                    t = clean_text(ln.get("text"))
                    chars.update(t)
                    try:
                        w, h = quad_size(line_quad(ln))
                    except (KeyError, ValueError, TypeError):
                        w, h = 0.0, 0.0
                    L["page"].append(pi); L["idx"].append(li); L["w"].append(w); L["h"].append(h)
                    L["need"].append(ctc_min_steps(t)); L["nchars"].append(len(t))
                if self.verbose and (pi + 1) % 50000 == 0:
                    print(f"[index]   {pi + 1:,} pages")
        self.page_offset = np.asarray(P["offset"], np.int64)
        self.page_split = np.asarray(P["split"], np.uint8)
        self.page_nlines = np.asarray(P["nlines"], np.int32)
        self.page_ntables = np.asarray(P["ntables"], np.int16)
        self.page_nlayout = np.asarray(P["nlayout"], np.int16)
        self.page_size = np.stack([np.asarray(P["width"], np.int32), np.asarray(P["height"], np.int32)], 1) \
            if P["width"] else np.zeros((0, 2), np.int32)
        self.line_page = np.asarray(L["page"], np.int32)
        self.line_idx = np.asarray(L["idx"], np.int32)
        self.line_w = np.asarray(L["w"], np.float32)
        self.line_h = np.asarray(L["h"], np.float32)
        self.line_need = np.asarray(L["need"], np.int32)
        self.line_nchars = np.asarray(L["nchars"], np.int32)
        self.char_counts = dict(chars)
        self.type_counts = dict(types)
        np.savez(self.cache, sig=self._sig, page_offset=self.page_offset, page_split=self.page_split,
                 page_nlines=self.page_nlines, page_ntables=self.page_ntables, page_nlayout=self.page_nlayout,
                 page_size=self.page_size, line_page=self.line_page, line_idx=self.line_idx, line_w=self.line_w,
                 line_h=self.line_h, line_need=self.line_need, line_nchars=self.line_nchars,
                 char_counts=json.dumps(self.char_counts, ensure_ascii=False),
                 type_counts=json.dumps(self.type_counts, ensure_ascii=False))
        if self.verbose:
            print(f"[index] {len(self.page_offset):,} pages, {len(self.line_page):,} lines"
                  + (f", {n_bad} unreadable records skipped" if n_bad else "") + f" -> {self.cache}")

    def _from_npz(self, z):
        for k in ("page_offset", "page_split", "page_nlines", "page_ntables", "page_nlayout", "page_size",
                  "line_page", "line_idx", "line_w", "line_h", "line_need", "line_nchars"):
            setattr(self, k, z[k])
        self.char_counts = json.loads(str(z["char_counts"]))
        self.type_counts = json.loads(str(z["type_counts"]))

    # ---- access --------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.page_offset)

    def read(self, pi: int) -> Dict:
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(int(self.page_offset[pi]))
        return _loads(self._fh.readline())

    def image(self, rec: Dict) -> np.ndarray:
        p = os.path.join(self.root, rec["image"])
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        return img

    def pages(self, split: str) -> np.ndarray:
        return np.nonzero(self.page_split == SPLITS[split])[0]

    def __getstate__(self):  # DataLoader workers: never share an open file handle
        d = self.__dict__.copy()
        d["_fh"] = None
        return d

    def summary(self) -> str:
        n_tr, n_va = int((self.page_split == 0).sum()), int((self.page_split == 1).sum())
        return (f"pages: {len(self):,} (train {n_tr:,} / val {n_va:,}) | lines: {len(self.line_page):,} | "
                f"tables: {int(self.page_ntables.sum()):,} | layout blocks: {int(self.page_nlayout.sum()):,}")


# --------------------------------------------------------------------------------------
# Charset
# --------------------------------------------------------------------------------------
def load_or_build_charset(index: DataIndex, path: str, min_count: int = 1, keep_joiners: bool = True) -> List[str]:
    """
    Load charset.json if it exists, otherwise build it from the WHOLE corpus and save it. Keep this file with your
    checkpoints: the model's output layer is tied to its order. Reports corpus characters missing from it.
    """
    if os.path.exists(path):
        chars = json.load(open(path, encoding="utf-8"))
    else:
        chars = sorted(c for c, n in index.char_counts.items() if n >= min_count)
        if not keep_joiners:
            chars = [c for c in chars if c not in ("‌", "‍")]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chars, f, ensure_ascii=False, indent=0)
        print(f"[charset] built {len(chars)} characters -> {path}")
    missing = {c: n for c, n in index.char_counts.items() if c not in set(chars)}
    if missing:
        top = sorted(missing.items(), key=lambda kv: -kv[1])[:20]
        print(f"[charset] WARNING {len(missing)} corpus characters are not in {path}; lines containing them are "
              f"excluded from recognition training: " + ", ".join(f"{c!r} U+{ord(c):04X} x{n}" for c, n in top))
    return chars


# --------------------------------------------------------------------------------------
# Recognition
# --------------------------------------------------------------------------------------
class RecConfig:
    """Crop + resize settings shared by every recognition loader. Margins are in units of line height."""

    def __init__(self, height=48, max_width=2560, vpad=(0.05, 0.30), hpad=(0.0, 0.40),
                 eval_vpad=0.15, eval_hpad=0.20, ctc_margin=1.2, min_h=4.0):
        self.height, self.max_width = height, max_width
        self.vpad, self.hpad, self.eval_vpad, self.eval_hpad = vpad, hpad, eval_vpad, eval_hpad
        self.ctc_margin, self.min_h = ctc_margin, min_h


def usable_lines(index: DataIndex, split: str, codec: CTCCodec, cfg: RecConfig,
                 verbose: bool = True) -> np.ndarray:
    """Line ids (into index.line_*) that can be trained on, with drop counts by reason."""
    in_split = index.page_split[index.line_page] == SPLITS[split]
    w, h, need, nchars = index.line_w, index.line_h, index.line_need, index.line_nchars
    h_safe = np.maximum(h, 1e-3)
    empty = nchars == 0
    tiny = (h < cfg.min_h) | (w < 2)
    # widest the crop can get: full horizontal margin; must fit max_width without squeezing
    natural_max = (w + 2 * cfg.hpad[1] * h) * cfg.height / h_safe
    too_wide = natural_max > cfg.max_width
    # fewest CTC steps the crop can get: largest vertical margin, no horizontal margin
    t_min = np.ceil(np.round(w * cfg.height / (h_safe * (1 + 2 * cfg.vpad[1]))) / 4)
    too_short = t_min < cfg.ctc_margin * need
    ok = in_split & ~empty & ~tiny & ~too_wide & ~too_short
    ids = np.nonzero(ok)[0]
    missing = set(index.char_counts) - set(codec.chars)
    n_unknown = 0
    if missing and len(ids):  # rare: only when charset.json is older than the data; needs a pass over pages
        keep = np.ones(len(ids), bool)
        by_page: Dict[int, List[int]] = {}
        for k, lid in enumerate(ids):
            by_page.setdefault(int(index.line_page[lid]), []).append(k)
        for pi, ks in by_page.items():
            lines = index.read(pi)["lines"]
            for k in ks:
                if set(clean_text(lines[int(index.line_idx[ids[k]])].get("text"))) & missing:
                    keep[k] = False
        n_unknown = int((~keep).sum())
        ids = ids[keep]
    if verbose:
        n = int(in_split.sum())
        print(f"[rec/{split}] usable {len(ids):,}/{n:,} lines | dropped: empty {int((in_split & empty).sum())}, "
              f"tiny {int((in_split & tiny & ~empty).sum())}, too wide {int((in_split & too_wide & ~tiny & ~empty).sum())}, "
              f"too few CTC steps {int((in_split & too_short & ~too_wide & ~tiny & ~empty).sum())}, "
              f"unknown chars {n_unknown}")
    return ids


def jitter_quad(q: np.ndarray, top: float, bottom: float, left: float, right: float) -> np.ndarray:
    """Grow a (tl,tr,br,bl) quad along its own axes. Margins are in pixels."""
    tl, tr, br, bl = q
    ux = (tr - tl + br - bl) / 2
    uy = (bl - tl + br - tr) / 2
    ux = ux / max(np.linalg.norm(ux), 1e-6)
    uy = uy / max(np.linalg.norm(uy), 1e-6)
    return np.array([tl - ux * left - uy * top, tr + ux * right - uy * top,
                     br + ux * right + uy * bottom, bl - ux * left + uy * bottom], np.float32)


def make_line_crop(page_bgr: np.ndarray, ln: Dict, cfg: RecConfig, rng: Optional[random.Random]) -> np.ndarray:
    """Quad crop with margins (random when rng is given) -> uint8 RGB resized to cfg.height."""
    q = line_quad(ln)
    _, h = quad_size(q)
    if rng is None:
        t = b = cfg.eval_vpad * h
        l = r = cfg.eval_hpad * h
    else:
        t, b = rng.uniform(*cfg.vpad) * h, rng.uniform(*cfg.vpad) * h
        l, r = rng.uniform(*cfg.hpad) * h, rng.uniform(*cfg.hpad) * h
    crop = crop_quad(page_bgr, jitter_quad(q, t, b, l, r))
    return resize_line_uint8(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), cfg.height, cfg.max_width, on_overflow="squeeze")


def rec_collate(samples: Sequence[Dict]) -> Dict:
    """Pad images with white; targets are FLAT (concatenated) as ctc_loss_fn expects."""
    batch, widths = collate_lines([line_to_tensor(s["image"]) for s in samples], [s["image"].shape[1] for s in samples])
    return {"images": batch, "widths": widths,
            "targets": torch.cat([s["ids"] for s in samples]),
            "target_lengths": torch.tensor([len(s["ids"]) for s in samples], dtype=torch.long),
            "texts": [s["text"] for s in samples]}


class RecDataset(Dataset):
    """Map-style (random access). Use for validation, debugging and the overfit test. Keeps a small page LRU."""

    def __init__(self, index: DataIndex, split: str, codec: CTCCodec, cfg: RecConfig = None, augment: bool = False,
                 line_ids: Optional[np.ndarray] = None, cache_pages: int = 8, verbose: bool = True):
        self.index, self.codec, self.cfg, self.augment = index, codec, cfg or RecConfig(), augment
        self.ids = usable_lines(index, split, codec, self.cfg, verbose) if line_ids is None else np.asarray(line_ids)
        self.cache_pages, self._cache = cache_pages, OrderedDict()

    def __len__(self):
        return len(self.ids)

    def natural_widths(self) -> np.ndarray:
        """Approximate resized widths (for width-bucketed batching)."""
        lid = self.ids
        h = np.maximum(self.index.line_h[lid], 1e-3)
        return (self.index.line_w[lid] + 2 * self.cfg.eval_hpad * h) * self.cfg.height / (h * (1 + 2 * self.cfg.eval_vpad))

    def _page(self, pi: int):
        if pi in self._cache:
            self._cache.move_to_end(pi)
            return self._cache[pi]
        rec = self.index.read(pi)
        item = (rec, self.index.image(rec))
        self._cache[pi] = item
        if len(self._cache) > self.cache_pages:
            self._cache.popitem(last=False)
        return item

    def __getitem__(self, i: int) -> Dict:
        lid = int(self.ids[i])
        rec, img = self._page(int(self.index.line_page[lid]))
        ln = rec["lines"][int(self.index.line_idx[lid])]
        text = clean_text(ln.get("text"))
        ids, _ = self.codec.encode([text])
        crop = make_line_crop(img, ln, self.cfg, _worker_rng() if self.augment else None)
        return {"image": crop, "ids": ids, "text": text}


class WidthBucketBatchSampler(Sampler):
    """Shuffle, then sort within chunks of `chunk` batches by width, so each batch has similar widths."""

    def __init__(self, widths: np.ndarray, batch_size: int, shuffle: bool = True, chunk: int = 50, seed: int = 0):
        self.widths, self.bs, self.shuffle, self.chunk, self.seed, self.epoch = np.asarray(widths), batch_size, shuffle, chunk, seed, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.widths)) if self.shuffle else np.arange(len(self.widths))
        span = self.bs * self.chunk
        batches = []
        for s in range(0, len(order), span):
            part = order[s:s + span]
            part = part[np.argsort(self.widths[part], kind="stable")]
            batches += [part[j:j + self.bs].tolist() for j in range(0, len(part), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.widths) / self.bs)


class RecStream(IterableDataset):
    """
    Streaming training loader. Each worker takes its share of pages (shuffled per epoch), decodes each page ONCE,
    crops all of its usable lines into a buffer, and when the buffer is full sorts it by width and yields batches
    in random order. Works with DistributedDataParallel (shards by rank too).
    Use with DataLoader(batch_size=None).
    """

    def __init__(self, index: DataIndex, split: str, codec: CTCCodec, cfg: RecConfig = None, batch_size: int = 64,
                 buffer_lines: int = 2048, max_batch_pixels: Optional[int] = None, augment: bool = True,
                 seed: int = 0, verbose: bool = True):
        self.index, self.codec, self.cfg = index, codec, cfg or RecConfig()
        self.bs, self.buffer, self.max_px, self.augment, self.seed, self.epoch = \
            batch_size, buffer_lines, max_batch_pixels, augment, seed, 0
        ids = usable_lines(index, split, codec, self.cfg, verbose)
        self.n_lines = len(ids)
        order = np.argsort(index.line_page[ids], kind="stable")
        ids = ids[order]
        pages, starts = np.unique(index.line_page[ids], return_index=True)
        self.pages = pages
        self.page_lines = np.split(index.line_idx[ids], starts[1:]) if len(ids) else []

    def set_epoch(self, e: int):
        self.epoch = e

    def approx_batches(self) -> int:
        return math.ceil(self.n_lines / self.bs)

    def _batches(self, buf: List[Dict], rng: random.Random) -> List[List[Dict]]:
        buf.sort(key=lambda s: s["image"].shape[1])
        out, cur, cur_w = [], [], 0
        for s in buf:
            w = s["image"].shape[1]
            new_w = max(cur_w, w)
            if cur and (len(cur) >= self.bs or (self.max_px and new_w * (len(cur) + 1) > self.max_px)):
                out.append(cur)
                cur, new_w = [], w
            cur.append(s)
            cur_w = new_w
        if cur:
            out.append(cur)
        rng.shuffle(out)
        return out

    def __iter__(self) -> Iterator[Dict]:
        wi = get_worker_info()
        n_workers, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        shard, n_shards = rank * n_workers + wid, world * n_workers
        rng = random.Random(self.seed * 1000003 + self.epoch * 7919 + shard)
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.pages))[shard::n_shards]
        buf: List[Dict] = []
        for k in order:
            pi = int(self.pages[k])
            rec = self.index.read(pi)
            try:
                img = self.index.image(rec)
            except FileNotFoundError as e:
                print(f"[rec] skipping missing image: {e}")
                continue
            for li in self.page_lines[k]:
                ln = rec["lines"][int(li)]
                text = clean_text(ln.get("text"))
                ids, _ = self.codec.encode([text])
                buf.append({"image": make_line_crop(img, ln, self.cfg, rng if self.augment else None),
                            "ids": ids, "text": text})
            if len(buf) >= self.buffer:
                for b in self._batches(buf, rng):
                    yield rec_collate(b)
                buf = []
        if buf:
            for b in self._batches(buf, rng):
                yield rec_collate(b)


# --------------------------------------------------------------------------------------
# Detection (DBNet++)
# --------------------------------------------------------------------------------------
_WHITE = normalize_page(np.full((1, 1, 3), 255, np.uint8))[:, 0, 0]  # normalised white, per channel


class DetDataset(Dataset):
    """
    One page per sample, resized with resize_for_detector (same as inference), then (training) a random crop.
    Lines cut by the crop edge, and lines too small to shrink, are marked "ignore" (masked out of the loss).
    crop=None returns the whole resized page (validation / overfit test; batch with det_collate).
    """

    def __init__(self, index: DataIndex, split: str, long_side: int = 1600, crop: Optional[int] = 960,
                 augment: bool = True, min_text_h: float = 6.0, text_focus: float = 0.8,
                 pages: Optional[np.ndarray] = None):
        self.index, self.long_side, self.crop, self.augment = index, long_side, crop, augment
        self.min_text_h, self.text_focus = min_text_h, text_focus
        self.pages = pages if pages is not None else \
            index.pages(split)[index.page_nlines[index.pages(split)] > 0]

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, i: int) -> Dict:
        rec = self.index.read(int(self.pages[i]))
        img, (sx, sy) = resize_for_detector(self.index.image(rec), self.long_side)
        quads = [line_quad(ln) / np.array([sx, sy], np.float32) for ln in rec.get("lines", [])]
        H, W = img.shape[:2]
        ox = oy = 0
        if self.crop:
            C = self.crop
            rng = _worker_rng() if self.augment else random.Random(i)
            if quads and rng.random() < self.text_focus:  # centre the crop near a random text line
                c = quads[rng.randrange(len(quads))].mean(0)
                ox = int(np.clip(c[0] - C / 2 + rng.uniform(-C / 4, C / 4), 0, max(W - C, 0)))
                oy = int(np.clip(c[1] - C / 2 + rng.uniform(-C / 4, C / 4), 0, max(H - C, 0)))
            else:
                ox, oy = rng.randint(0, max(W - C, 0)), rng.randint(0, max(H - C, 0))
            canvas = np.full((C, C, 3), 255, np.uint8)
            part = img[oy:oy + C, ox:ox + C]
            canvas[:part.shape[0], :part.shape[1]] = part
            img, H, W = canvas, C, C
            quads = [q - np.array([ox, oy], np.float32) for q in quads]
        polys, ignore = [], []
        for q in quads:
            x1, y1 = q.min(0)
            x2, y2 = q.max(0)
            if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
                continue
            inside = (min(x2, W) - max(x1, 0)) * (min(y2, H) - max(y1, 0)) / max((x2 - x1) * (y2 - y1), 1e-6)
            _, h = quad_size(q)
            polys.append(q)
            ignore.append(inside < 0.95 or h < self.min_text_h)
        t = make_db_targets(polys, ignore, H, W)
        return {"image": normalize_page(img), **{k: torch.from_numpy(v) for k, v in t.items()},
                "quads": [p for p, ig in zip(polys, ignore) if not ig]}


def det_collate(samples: Sequence[Dict]) -> Dict:
    """Stacks samples, padding different-sized pages bottom/right (image with white, gt_mask with 0)."""
    H = max(s["image"].shape[1] for s in samples)
    W = max(s["image"].shape[2] for s in samples)
    B = len(samples)
    out = {"image": _WHITE[None, :, None, None].repeat(B, 1, H, W).clone()}
    for k in ("gt", "gt_mask", "thresh_map", "thresh_mask"):
        out[k] = torch.zeros(B, 1, H, W)
    for i, s in enumerate(samples):
        h, w = s["image"].shape[1:]
        out["image"][i, :, :h, :w] = s["image"]
        for k in ("gt", "gt_mask", "thresh_map", "thresh_mask"):
            out[k][i, :, :h, :w] = s[k]
    out["quads"] = [s["quads"] for s in samples]
    return out


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------
def check_layout_types(index: DataIndex):
    """Fail at startup (not mid-epoch) if the generator produced a block type the model can't map."""
    bad = []
    for t in index.type_counts:
        try:
            label_of(t)
        except KeyError:
            bad.append(t)
    if bad:
        raise KeyError(f"layout types not in layout_detector.TYPE_ALIASES: {bad}")


class LayoutDataset(Dataset):
    """Returns (image float (3,H,W) in [0,1] RGB, target {boxes, labels}) for torchvision detectors."""

    def __init__(self, index: DataIndex, split: str, pages: Optional[np.ndarray] = None):
        check_layout_types(index)
        self.index = index
        self.pages = pages if pages is not None else \
            index.pages(split)[index.page_nlayout[index.pages(split)] > 0]

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, i: int):
        rec = self.index.read(int(self.pages[i]))
        img = self.index.image(rec)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
        return x, layout_targets_from_json(rec)


def layout_collate(samples):
    return [s[0] for s in samples], [s[1] for s in samples]


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
class TableDataset(Dataset):
    """One sample per table. Unusable tables return None (collate drops them); see .skipped for the count."""

    def __init__(self, index: DataIndex, split: str, vocab: TableVocab, S: int = DEFAULT_S, pad: int = 4,
                 pairs: Optional[List[Tuple[int, int]]] = None):
        self.index, self.vocab, self.S, self.pad = index, vocab, S, pad
        if pairs is None:
            pages = index.pages(split)
            pages = pages[index.page_ntables[pages] > 0]
            pairs = [(int(p), t) for p in pages for t in range(int(index.page_ntables[p]))]
        self.pairs = pairs
        self.skipped = 0
        self._cache = (None, None, None)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int) -> Optional[Dict]:
        pi, ti = self.pairs[i]
        if self._cache[0] != pi:
            rec = self.index.read(pi)
            self._cache = (pi, rec, self.index.image(rec))
        _, rec, img = self._cache
        try:
            return table_sample_from_json(img, rec["tables"][ti], self.vocab, S=self.S, pad=self.pad)
        except ValueError as e:
            self.skipped += 1
            if self.skipped <= 5:
                print(f"[table] skipping {rec.get('image')} table {ti}: {e}")
            return None


def make_table_collate(pad_id: int):
    def collate(samples):
        real = [s for s in samples if s is not None]
        return collate_tables(real, pad_id) if real else None  # training loop: `if batch is None: continue`
    return collate


# --------------------------------------------------------------------------------------
# Loader factories
# --------------------------------------------------------------------------------------
def _workers(n):
    return min(8, os.cpu_count() or 1) if n is None else n


def _dl_kwargs(num_workers):
    return {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0}


def make_rec_loader(index: DataIndex, split: str, codec: CTCCodec, batch_size: int = 64, stream: bool = True,
                    augment: bool = True, cfg: RecConfig = None, num_workers: Optional[int] = None, **kw) -> DataLoader:
    nw = _workers(num_workers)
    if stream:
        ds = RecStream(index, split, codec, cfg, batch_size=batch_size, augment=augment, **kw)
        return DataLoader(ds, batch_size=None, **_dl_kwargs(nw))
    ds = RecDataset(index, split, codec, cfg, augment=augment, **kw)
    sampler = WidthBucketBatchSampler(ds.natural_widths(), batch_size, shuffle=augment)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=rec_collate, **_dl_kwargs(nw))


def make_det_loader(index: DataIndex, split: str, batch_size: int = 8, num_workers: Optional[int] = None,
                    **kw) -> DataLoader:
    ds = DetDataset(index, split, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train", collate_fn=det_collate,
                      drop_last=split == "train", **_dl_kwargs(_workers(num_workers)))


def make_layout_loader(index: DataIndex, split: str, batch_size: int = 4, num_workers: Optional[int] = None,
                       **kw) -> DataLoader:
    ds = LayoutDataset(index, split, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train", collate_fn=layout_collate,
                      **_dl_kwargs(_workers(num_workers)))


def make_table_loader(index: DataIndex, split: str, vocab: TableVocab, batch_size: int = 16,
                      num_workers: Optional[int] = None, **kw) -> DataLoader:
    ds = TableDataset(index, split, vocab, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train", collate_fn=make_table_collate(vocab.pad),
                      **_dl_kwargs(_workers(num_workers)))


# --------------------------------------------------------------------------------------
# CLI check with preview images
# --------------------------------------------------------------------------------------
def _save_rec_preview(batch: Dict, path: str, n: int = 24):
    imgs = ((batch["images"][:n].permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    W = max(min(int(w), 1600) for w in batch["widths"][:n])
    rows = [np.full((imgs.shape[1] + 6, W, 3), 255, np.uint8) for _ in imgs]
    for r, im, w in zip(rows, imgs, batch["widths"][:n]):
        w = min(int(w), W)
        r[3:3 + im.shape[0], :w] = im[:, :w]
        r[-1:, :] = 200
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR))
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(batch["texts"][:n]))


def _save_det_preview(batch: Dict, path: str):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = ((batch["image"][0] * std + mean).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()
    gt = batch["gt"][0, 0].numpy() > 0.5
    ign = batch["gt_mask"][0, 0].numpy() < 0.5
    img[gt] = (0.5 * img[gt] + [0, 127, 0]).astype(np.uint8)
    img[ign] = (0.5 * img[ign] + [0, 0, 127]).astype(np.uint8)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(description="Build the index + charset and sanity-check every loader.")
    ap.add_argument("--data", required=True, help="folder with labels.jsonl and the images it references")
    ap.add_argument("--labels", default="labels.jsonl")
    ap.add_argument("--val-permille", type=int, default=20, help="validation share per 1000 documents")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    index = DataIndex(os.path.join(a.data, a.labels), a.val_permille, rebuild=a.rebuild)
    print("[index]", index.summary())
    print("[index] layout types:", dict(sorted(index.type_counts.items(), key=lambda kv: -kv[1])))
    check_layout_types(index)
    codec = CTCCodec(load_or_build_charset(index, os.path.join(a.data, "charset.json")))
    print(f"[charset] {codec.num_classes - 1} characters (+ blank)")
    prev = os.path.join(a.data, "_preview")
    os.makedirs(prev, exist_ok=True)

    for split in ("train", "val"):
        if not len(index.pages(split)):
            print(f"[{split}] no pages (small dataset?) - skipping")
            continue
        rl = make_rec_loader(index, split, codec, batch_size=32, num_workers=0, augment=split == "train")
        b = next(iter(rl))
        print(f"[rec/{split}] batch images {tuple(b['images'].shape)} | widths {b['widths'].min().item()}-"
              f"{b['widths'].max().item()} | targets {tuple(b['targets'].shape)}")
        _save_rec_preview(b, os.path.join(prev, f"rec_{split}.png"))

    dl = make_det_loader(index, "train", batch_size=2, num_workers=0)
    b = next(iter(dl))
    print(f"[det] batch {tuple(b['image'].shape)} | text px {int(b['gt'].sum())} | ignored px {int((1 - b['gt_mask']).sum())}")
    _save_det_preview(b, os.path.join(prev, "det_train.png"))

    if index.page_nlayout.sum():
        ims, tgs = next(iter(make_layout_loader(index, "train", batch_size=2, num_workers=0)))
        print(f"[layout] {len(ims)} images, first {tuple(ims[0].shape)}, labels "
              f"{[LAYOUT_CLASSES[int(l) - 1] for l in tgs[0]['labels']]}")
    v = TableVocab()
    if index.page_ntables.sum():
        tl = make_table_loader(index, "train", v, batch_size=4, num_workers=0)
        tb = next(b for b in tl if b is not None)
        print(f"[table] batch images {tuple(tb['image'].shape)} | tokens {tuple(tb['tokens'].shape)} | "
              f"cells {int(tb['box_mask'].sum())} | skipped {tl.dataset.skipped}")
    print(f"\nPreviews written to {prev}/ - open rec_*.png and check every line is fully visible (matras above and "
          f"below not cut) and matches rec_*.txt; det_train.png shows text in green and ignored regions in red.")
    print("All loaders OK.")


if __name__ == "__main__":
    main()