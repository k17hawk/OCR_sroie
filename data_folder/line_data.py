"""
Line-level dataset for the SVTR recognizer.

Manifest format (one JSON object per line):
    {"image": "pages/page_0001.png",
     "lines": [{"quad": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text": "..."}, ...]}

If your pages are already stored as page JSON (with `image`, `width`, `height`, `lines`), use
`manifest_from_pages(src_dir, out_path)` once to flatten them into this format.

Feasibility: any line whose resized width exceeds `max_width`, or whose label needs more CTC steps than the
model will produce, is dropped at dataset construction and counted in `skipped`. This keeps the training
loss free of silently-zeroed samples (see `ctc_loss_fn`).
"""
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline import crop_quad
from recognizer_svtr import (CTCCodec, collate_lines, check_ctc_feasible, nfc, resize_pad_line)


# --------------------------------------------------------------------------------------
# Manifest utilities
# --------------------------------------------------------------------------------------
def manifest_from_pages(pages_dir: str, out_path: str, image_root: Optional[str] = None) -> int:
    """
    Flatten a directory of page JSONs into a JSONL manifest. Returns the number of pages written.
    `image_root` is prefixed to each page's `image` field so the manifest is portable.
    """
    pages = sorted(Path(pages_dir).glob("*.json"))
    n = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for p in pages:
            rec = json.loads(p.read_text(encoding="utf-8"))
            img = rec.get("image") or rec.get("image_path")
            if not img:
                continue
            if image_root:
                img = str(Path(image_root) / Path(img).name)
            lines = [{"quad": l["quad"], "text": l["text"]} for l in rec.get("lines", [])
                     if l.get("text") and l.get("quad")]
            if not lines:
                continue
            fo.write(json.dumps({"image": img, "lines": lines}, ensure_ascii=False) + "\n")
            n += 1
    return n


def collect_labels(manifest_path: str) -> List[str]:
    """Every label in the manifest, used once to build the charset."""
    out = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for l in rec.get("lines", []):
                if l.get("text"):
                    out.append(l["text"])
    return out


# --------------------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------------------
class Augment:
    """
    Light, line-level augmentation. Applied on the BGR crop before resize_pad_line, so it can use OpenCV.
    Kept conservative on purpose: the detector already produces tightly-cropped lines, and aggressive
    geometric warping would shift the character boundaries the CTC alignment depends on.
    """

    def __init__(self, rotate_deg: float = 1.5, blur_p: float = 0.15, blur_sigma: Tuple[float, float] = (0.3, 1.2),
                 brightness: float = 0.10, contrast: float = 0.10, jpeg_p: float = 0.10,
                 jpeg_quality: Tuple[int, int] = (70, 95), pad_p: float = 0.2, seed: Optional[int] = None):
        self.rotate_deg = rotate_deg
        self.blur_p = blur_p
        self.blur_sigma = blur_sigma
        self.brightness = brightness
        self.contrast = contrast
        self.jpeg_p = jpeg_p
        self.jpeg_quality = jpeg_quality
        self.pad_p = pad_p
        self.rng = random.Random(seed)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        r = self.rng
        h, w = img_bgr.shape[:2]

        # White padding so rotation does not clip the corners.
        if r.random() < self.pad_p:
            py = max(1, int(0.05 * h))
            px = max(1, int(0.02 * w))
            img_bgr = cv2.copyMakeBorder(img_bgr, py, py, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            h, w = img_bgr.shape[:2]

        # Small rotation.
        if self.rotate_deg > 0:
            a = r.uniform(-self.rotate_deg, self.rotate_deg)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
            img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderValue=(255, 255, 255))

        # Blur.
        if r.random() < self.blur_p:
            s = r.uniform(*self.blur_sigma)
            img_bgr = cv2.GaussianBlur(img_bgr, (0, 0), s)

        # Brightness and contrast.
        if self.brightness > 0 or self.contrast > 0:
            alpha = 1.0 + r.uniform(-self.contrast, self.contrast)
            beta = r.uniform(-self.brightness, self.brightness) * 255.0
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

        # JPEG artifacts.
        if r.random() < self.jpeg_p:
            q = r.randint(*self.jpeg_quality)
            ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img_bgr


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class LineDataset(Dataset):
    def __init__(self, manifest: str, codec: CTCCodec, height: int = 48, max_width: int = 2560,
                 margin: float = 1.2, augment: Optional[Augment] = None, cache_size: int = 2):
        self.codec = codec
        self.height = height
        self.max_width = max_width
        self.augment = augment
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size

        self.items: List[Tuple[str, np.ndarray, str]] = []
        self.skipped = {"unknown_chars": 0, "infeasible": 0, "missing_image": 0, "empty_text": 0}
        missing = set()

        with open(manifest, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                path = rec["image"]
                if not Path(path).exists():
                    missing.add(path)
                    continue
                for l in rec.get("lines", []):
                    text = nfc(l.get("text") or "")
                    if not text:
                        self.skipped["empty_text"] += 1
                        continue
                    if codec.unknown_chars(text):
                        self.skipped["unknown_chars"] += 1
                        continue
                    quad = np.asarray(l["quad"], np.float32)
                    # Approximate crop dims from the quad (matches crop_quad within a pixel).
                    w = int(round(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[3] - quad[2]))))
                    h = int(round(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))))
                    if w < 4 or h < 4:
                        self.skipped["infeasible"] += 1
                        continue
                    ok, _ = check_ctc_feasible(text, w, h, height=height, max_width=max_width, margin=margin)
                    if not ok:
                        self.skipped["infeasible"] += 1
                        continue
                    self.items.append((path, quad, text))

        self.skipped["missing_image"] = len(missing)

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: str) -> np.ndarray:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[path] = img
        return img

    def __getitem__(self, idx: int) -> Dict:
        path, quad, text = self.items[idx]
        img = self._load_image(path)
        crop = crop_quad(img, quad)
        if self.augment is not None:
            crop = self.augment(crop)
        # "squeeze" is a safety net: at construction we already checked feasibility, but a 1-px rounding
        # difference in crop_quad can put a line a hair over max_width.
        t, w = resize_pad_line(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), self.height, self.max_width,
                               on_overflow="squeeze")
        return {"tensor": t, "width": w, "text": text, "path": path}


def collate_line_batch(samples: List[Dict], codec: CTCCodec) -> Dict:
    tensors = [s["tensor"] for s in samples]
    widths = [s["width"] for s in samples]
    texts = [s["text"] for s in samples]
    batch, widths_t = collate_lines(tensors, widths)
    targets, target_lengths = codec.encode(texts, strict=True)
    from recognizer_svtr import SVTRRecognizer  # avoid circular import at module load
    input_lengths = SVTRRecognizer.out_lengths(widths_t)
    return {"images": batch, "widths": widths_t,
            "targets": targets, "target_lengths": target_lengths, "input_lengths": input_lengths,
            "texts": texts}