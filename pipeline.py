"""
End-to-end inference: (optional preprocess) -> layout -> line detection -> line recognition -> table structure
-> merge -> output in the ground-truth text format.

    pipe = OCRPipeline(layout=..., detector=..., recognizer=..., codec=..., table=...)
    result = pipe(cv2.imread("page.png"))
    result["text"], result["tables"][0]["html"], result["tables"][0]["grid"], result["lines"][i]["conf"], ...

Every model is optional except detector + recognizer + codec:
  no layout model -> the whole page is one text block; no table model -> tables are read as plain text.

Preprocessing is OFF by default because the models are trained on raw rendered pages. If you turn it on, apply
the SAME preprocess() to your training images. All returned coordinates are always in the ORIGINAL image's
pixels (deskew is undone on the outputs), so they line up for a searchable-PDF text layer.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from detector_dbnet import detect_lines
from layout_detector import detect_layout
from merge import merge_table, order_blocks, quad_to_bbox, serialize_page, block_text
from recognizer_svtr import CTCCodec, SVTRRecognizer, collate_lines, resize_pad_line
from table_slanet import DEFAULT_S, TableStructureModel, boxes_to_crop_pixels, letterbox


# --------------------------------------------------------------------------------------
# Preprocessing (use the same function on training images if enabled)
# --------------------------------------------------------------------------------------
def estimate_skew(img_bgr: np.ndarray, max_angle: float = 5.0, step: float = 0.25) -> float:
    """
    Projection-profile skew estimate: rotate a downscaled ink mask by candidate angles and keep the one whose
    row profile is sharpest. The returned angle is the rotation that straightens the page when passed to
    cv2.getRotationMatrix2D, so the sign is correct by construction.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    s = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = ink.shape
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        rot = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rot.sum(1, dtype=np.float64)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def preprocess(img_bgr: np.ndarray, deskew: bool = True, clahe: bool = True, max_angle: float = 5.0,
               min_angle: float = 0.3) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns (image, M) where M is the 2x3 affine applied (None if not rotated)."""
    out, M = img_bgr, None
    if deskew:
        a = estimate_skew(img_bgr, max_angle)
        if abs(a) >= min_angle:
            h, w = img_bgr.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
            out = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out, M


def _map_points(pts: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    return (np.hstack([p, np.ones((len(p), 1), np.float32)]) @ Minv.T).astype(np.float32)


def _map_bbox(b: List[float], Minv: np.ndarray) -> List[float]:
    c = _map_points(np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]), Minv)
    return [float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())]


# --------------------------------------------------------------------------------------
# Line crops
# --------------------------------------------------------------------------------------
def crop_quad(img: np.ndarray, quad: np.ndarray, rotate_ratio: float = 3.0, rotate_min_h: int = 64) -> np.ndarray:
    """
    Perspective-rectify a quad (tl, tr, br, bl). A crop is treated as vertical text and rotated only if it is
    both very tall relative to its width and taller than rotate_min_h px, so single characters such as a
    serial number '१' (about 6x11 px) are never turned sideways.
    """
    q = np.asarray(quad, np.float32)
    w = int(round(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2]))))
    h = int(round(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))))
    w, h = max(w, 2), max(h, 2)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if h >= rotate_ratio * w and h > rotate_min_h:
        crop = np.ascontiguousarray(np.rot90(crop))
    return crop


def split_wide_crop(crop_bgr: np.ndarray, height: int = 48, max_width: int = 2560,
                    fill: float = 0.8, window: float = 0.10) -> Tuple[List[np.ndarray], List[bool]]:
    """
    Split a crop whose resized width would exceed max_width. Each cut is moved to the lowest-ink column within
    +-window of its ideal position. With fill=0.8 and window=0.10 every piece is at most 0.8 * 1.2 = 0.96 of
    max_width, so it always fits. Returns (pieces, gaps) where gaps[i] is True if cut i fell in whitespace
    (join with a space) and False if it cut through a word (join without a space).
    """
    h, w = crop_bgr.shape[:2]
    scale = height / max(h, 1)
    if w * scale <= max_width:
        return [crop_bgr], []
    n = int(np.ceil(w * scale / (fill * max_width)))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    col = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1].sum(0).astype(np.float32)
    smooth = np.convolve(col, np.ones(5, np.float32) / 5, mode="same")  # prefer wide gaps over 1px ones
    gap_thresh = 0.02 * max(float(col.max()), 1.0)
    cuts, gaps = [0], []
    win = max(int(w / n * window), 3)
    for k in range(1, n):
        ideal = int(w * k / n)
        lo, hi = max(cuts[-1] + 1, ideal - win), min(w - 1, ideal + win)
        c = lo + int(np.argmin(smooth[lo:hi])) if hi > lo else ideal
        cuts.append(c)
        gaps.append(bool(col[max(c - 1, 0):c + 2].max() <= gap_thresh))
    cuts.append(w)
    pieces, kept_gaps = [], []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        if b - a > 2:
            if pieces and i - 1 < len(gaps):
                kept_gaps.append(gaps[i - 1])
            pieces.append(crop_bgr[:, a:b])
    return pieces, kept_gaps


def _join_chunks(texts: List[str], gaps: List[bool]) -> str:
    out = texts[0] if texts else ""
    for t, g in zip(texts[1:], gaps):
        if not t:
            continue
        out = (out + " " + t) if (g and out) else out + t
    return out.strip()


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
class OCRPipeline:
    def __init__(self, detector, recognizer: SVTRRecognizer, codec: CTCCodec, layout=None,
                 table: Optional[TableStructureModel] = None, device="cpu", rec_batch=32,
                 review_thresh=0.8, table_size=DEFAULT_S, do_preprocess=False, det_long_side=1600,
                 rec_height=48, rec_max_width=2560, det_kwargs: Optional[Dict] = None):
        self.layout, self.detector, self.recognizer, self.codec, self.table = layout, detector, recognizer, codec, table
        self.device, self.rec_batch, self.review_thresh = device, rec_batch, review_thresh
        self.table_size, self.do_preprocess, self.det_long_side = table_size, do_preprocess, det_long_side
        self.rec_height, self.rec_max_width = rec_height, rec_max_width
        self.det_kwargs = det_kwargs or {}  # e.g. {"unclip_ratio": 1.5, "box_thresh": 0.6}
        for m in (layout, detector, recognizer, table):
            if m is not None:
                m.to(device).eval()

    # ---- recognition -------------------------------------------------------------------
    @torch.no_grad()
    def recognize_crops(self, crops: List[np.ndarray]) -> List[Dict]:
        """Recognise BGR crops; wide crops are split, chunks recognised in width-sorted batches, and re-joined."""
        pieces, owner = [], []
        gaps_per_crop: List[List[bool]] = []
        for ci, crop in enumerate(crops):
            parts, gaps = split_wide_crop(crop, self.rec_height, self.rec_max_width)
            gaps_per_crop.append(gaps)
            for p in parts:
                t, w = resize_pad_line(cv2.cvtColor(p, cv2.COLOR_BGR2RGB), self.rec_height, self.rec_max_width,
                                       on_overflow="squeeze")
                pieces.append((t, w))
                owner.append(ci)
        decoded: List[Optional[Dict]] = [None] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: pieces[i][1])  # bucket by width
        for s in range(0, len(order), self.rec_batch):
            idx = order[s:s + self.rec_batch]
            batch, widths = collate_lines([pieces[i][0] for i in idx], [pieces[i][1] for i in idx])
            logits = self.recognizer(batch.to(self.device), widths.to(self.device)).float().cpu()
            for i, d in zip(idx, self.codec.decode_greedy(logits, SVTRRecognizer.out_lengths(widths))):
                decoded[i] = d
        results = []
        for ci in range(len(crops)):
            ds = [decoded[i] for i in range(len(pieces)) if owner[i] == ci]  # left-to-right order kept
            nonempty = [d for d in ds if not d["empty"]]
            results.append({
                "text": _join_chunks([d["text"] for d in ds], gaps_per_crop[ci]),
                "conf": min((d["conf_min"] for d in nonempty), default=0.0),
                "conf_mean": float(np.mean([d["conf_mean"] for d in nonempty])) if nonempty else 0.0,
                "char_probs": [p for d in ds for p in d["char_probs"]],
                "chunks": len(ds),
            })
        return results

    def recognize_lines(self, img, quads, det_scores) -> List[Dict]:
        recs = self.recognize_crops([crop_quad(img, q) for q in quads])
        return [{"quad": np.asarray(q, np.float32), "bbox": quad_to_bbox(q), **r, "det_score": float(s)}
                for q, s, r in zip(quads, det_scores, recs)]

    def _reread_box(self, img) :
        def fn(box: List[float]) -> Tuple[str, float]:
            x1, y1, x2, y2 = box
            pad = 2.0
            q = np.array([[x1 + pad, y1 + pad], [x2 - pad, y1 + pad], [x2 - pad, y2 - pad], [x1 + pad, y2 - pad]], np.float32)
            if x2 - x1 <= 2 * pad + 2 or y2 - y1 <= 2 * pad + 2:
                return "", 0.0
            r = self.recognize_crops([crop_quad(img, q)])[0]
            return r["text"], r["conf"]
        return fn

    # ---- tables --------------------------------------------------------------------------
    @torch.no_grad()
    def read_table(self, img, bbox, lines, exclude: Optional[set] = None) -> Optional[Dict]:
        """exclude: ids of lines already used by another table (a line is never read into two tables)."""
        exclude = exclude or set()
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        x = letterbox(crop, self.table_size)[None]
        pred = self.table.predict(x.to(self.device))[0]
        if not pred["cell_boxes"]:
            return None
        boxes = boxes_to_crop_pixels(np.array(pred["cell_boxes"]), w, h, self.table_size) + \
            np.array([x1, y1, x1, y1], np.float32)
        inside = [i for i, l in enumerate(lines) if i not in exclude
                  and x1 <= (l["bbox"][0] + l["bbox"][2]) / 2 <= x2 and y1 <= (l["bbox"][1] + l["bbox"][3]) / 2 <= y2]
        sub = [lines[i] for i in inside]
        merged = merge_table(pred["tokens"], boxes, pred["cell_probs"], sub, review_thresh=self.review_thresh,
                             reread=self._reread_box(img))
        merged["used_line_ids"] = {inside[j] for j in merged["used_line_ids"]}
        merged["straddling_line_ids"] = {inside[j] for j in merged["straddling_line_ids"]}
        merged["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
        merged["structure_conf"] = float(np.min(pred["token_probs"])) if pred["token_probs"] else 0.0
        merged["confidence"] = min([c["confidence"] for c in merged["cells"]] + [merged["structure_conf"]])
        return merged

    # ---- full page -----------------------------------------------------------------------
    def __call__(self, img_bgr: np.ndarray) -> Dict:
        M = None
        img = img_bgr
        if self.do_preprocess:
            img, M = preprocess(img_bgr)
        H, W = img.shape[:2]
        blocks = detect_layout(self.layout, img, self.device) if self.layout is not None else \
            [{"type": "text", "bbox": [0.0, 0.0, float(W), float(H)], "score": 1.0}]
        quads, det_scores = detect_lines(self.detector, img, self.device, long_side=self.det_long_side,
                                         **self.det_kwargs)
        lines = self.recognize_lines(img, quads, det_scores) if quads else []

        # Tables first, highest score first. Each text line is used by at most ONE table: where two table boxes
        # overlap, the shared lines go to the more confident table, and a table left with no lines of its own is
        # dropped (its region is read as plain text instead). Without this, overlapping tables duplicate text.
        consumed, tables = set(), []
        for b in blocks:
            b["lines"] = []
        if self.table is not None:
            for b in sorted([b for b in blocks if b["type"] == "table"], key=lambda b: -b.get("score", 0.0)):
                t = self.read_table(img, b["bbox"], lines, exclude=consumed)
                if t is not None and t["used_line_ids"]:
                    b["table"] = t
                    consumed |= t["used_line_ids"]
                    tables.append(t)
                elif t is not None:
                    b["dropped_table"] = "no lines left after higher-scoring overlapping table"
        tables.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))  # page order for output
        for li, l in enumerate(lines):  # attach remaining lines to the smallest block containing their centre
            if li in consumed:
                continue
            cx, cy = (l["bbox"][0] + l["bbox"][2]) / 2, (l["bbox"][1] + l["bbox"][3]) / 2
            host = [b for b in blocks if "table" not in b and b["bbox"][0] <= cx <= b["bbox"][2]
                    and b["bbox"][1] <= cy <= b["bbox"][3]]
            if host:
                min(host, key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))["lines"].append(l)
            else:
                blocks.append({"type": "text", "bbox": list(l["bbox"]), "score": 0.0, "lines": [l], "orphan": True})
        blocks = [b for b in blocks if b["lines"] or "table" in b]

        ordered = order_blocks(blocks, W)
        for b in ordered:
            if "table" not in b:
                b["text"] = block_text(b)
                b["conf"] = min(l["conf"] for l in b["lines"])
        for l in lines:
            l["needs_review"] = l["conf"] < self.review_thresh
        text = serialize_page(ordered)

        if M is not None:  # report everything in original-image pixels
            Minv = cv2.invertAffineTransform(M)
            for l in lines:
                l["quad"] = _map_points(l["quad"], Minv)
                l["bbox"] = quad_to_bbox(l["quad"])
            for b in ordered:
                b["bbox"] = _map_bbox(b["bbox"], Minv)
            for t in tables:
                t["bbox"] = _map_bbox(t["bbox"], Minv)
                for c in t["cells"]:
                    c["bbox"] = _map_bbox(c["bbox"], Minv)
        return {"text": text, "blocks": ordered, "lines": lines, "tables": tables,
                "deskew_matrix": None if M is None else M.tolist()}