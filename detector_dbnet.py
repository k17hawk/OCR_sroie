"""
Stage 2 - Text line detector: DBNet++ (ResNet-18 backbone + FPN + adaptive scale fusion + DB head).

Training targets come from your line quads (make_db_targets). Inference returns rotated quads with scores
(db_postprocess). Needs: torch, torchvision, opencv-python, numpy, pyclipper.

Notes
- Heads output LOGITS. The loss is computed in fp32 with autocast disabled, so mixed-precision (AMP) training
  works (F.binary_cross_entropy on probabilities raises an error under autocast).
- Resolution: small table text (~13 px on a 1600 px page) needs long_side >= 1600. Train at the same scale.
- Shrink is capped at 30% of the line height per side (make_db_targets max_shrink), so unclip_ratio=1.5 restores
  words and long lines alike. Confirm on held-out pages with tune_unclip_ratio (the overfit test reports it too):
  too small clips matras / shirorekha, too large merges adjacent lines.
"""
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pyclipper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DEFAULT_LONG_SIDE = 1600


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class ASF(nn.Module):
    """DBNet++ adaptive scale fusion: spatial attention, then per-scale attention weights."""

    def __init__(self, inter=64, num_scales=4):
        super().__init__()
        c = inter * num_scales
        self.conv = nn.Conv2d(c, inter, 3, padding=1, bias=False)
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 1, bias=False), nn.Sigmoid())
        self.channel = nn.Sequential(nn.Conv2d(inter, num_scales, 1, bias=False), nn.Sigmoid())
        self.n = num_scales

    def forward(self, feats: List[torch.Tensor]):
        x = self.conv(torch.cat(feats, 1))
        x = x + self.spatial(x.mean(1, keepdim=True))
        w = self.channel(x)  # (B, n, H, W)
        return torch.cat([w[:, i:i + 1] * feats[i] for i in range(self.n)], 1)


class DBNetPP(nn.Module):
    def __init__(self, pretrained_backbone: bool = True, inner=256, k=50.0):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # strides 4,8,16,32
        chs = [64, 128, 256, 512]
        self.lateral = nn.ModuleList([nn.Conv2d(c, inner, 1, bias=False) for c in chs])
        self.smooth = nn.ModuleList([nn.Conv2d(inner, inner // 4, 3, padding=1, bias=False) for _ in chs])
        self.asf = ASF(inner // 4, 4)
        self.k = k
        self.prob_head = self._head(inner)
        self.thresh_head = self._head(inner)

    @staticmethod
    def _head(c):
        """Outputs logits at input resolution (two 2x transposed convs after the stride-4 features)."""
        return nn.Sequential(
            nn.Conv2d(c, c // 4, 3, padding=1, bias=False), nn.BatchNorm2d(c // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c // 4, c // 4, 2, 2), nn.BatchNorm2d(c // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c // 4, 1, 2, 2))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        lat = [l(c) for l, c in zip(self.lateral, (c1, c2, c3, c4))]
        for i in range(3, 0, -1):  # top-down
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        p = [s(f) for s, f in zip(self.smooth, lat)]
        p = [F.interpolate(f, size=p[0].shape[-2:], mode="nearest") for f in p]  # all at stride 4
        fused = self.asf(p)
        prob_logits = self.prob_head(fused)
        out = {"prob_logits": prob_logits, "prob": torch.sigmoid(prob_logits)}
        if self.training:
            thresh = torch.sigmoid(self.thresh_head(fused))
            out["thresh"] = thresh
            out["binary"] = torch.sigmoid(self.k * (out["prob"] - thresh))
        return out


# --------------------------------------------------------------------------------------
# Loss (fp32, AMP-safe)
# --------------------------------------------------------------------------------------
def _balanced_bce(logits, gt, mask, neg_ratio=3.0, eps=1e-6):
    pos, neg = (gt * mask) > 0.5, ((1 - gt) * mask) > 0.5
    n_pos = int(pos.sum())
    # pages with no text still learn from their hardest negatives
    n_neg = min(int(neg.sum()), int(n_pos * neg_ratio) if n_pos > 0 else 100)
    loss = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    pos_loss = (loss * pos).sum()
    neg_loss = torch.topk((loss * neg).view(-1), n_neg).values.sum() if n_neg > 0 else loss.new_zeros(())
    return (pos_loss + neg_loss) / (n_pos + n_neg + eps)


def _dice(pred, gt, mask, eps=1e-6):
    inter = (pred * gt * mask).sum()
    union = (pred * mask).sum() + (gt * mask).sum() + eps
    return 1 - 2 * inter / union


def db_loss(out, batch, alpha=1.0, beta=10.0):
    """batch: gt, gt_mask, thresh_map, thresh_mask - all (B,1,H,W) float tensors at input resolution."""
    device_type = out["prob_logits"].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        o = {k: v.float() for k, v in out.items()}
        b = {k: v.float() for k, v in batch.items()}
        l_prob = _balanced_bce(o["prob_logits"], b["gt"], b["gt_mask"])
        l_bin = _dice(o["binary"], b["gt"], b["gt_mask"])
        tm = b["thresh_mask"]
        l_thr = (torch.abs(o["thresh"] - b["thresh_map"]) * tm).sum() / (tm.sum() + 1e-6)
        total = l_prob + alpha * l_bin + beta * l_thr
    return total, {"prob": l_prob.item(), "binary": l_bin.item(), "thresh": l_thr.item()}


# --------------------------------------------------------------------------------------
# Training targets from polygons / quads
# --------------------------------------------------------------------------------------
def _dist_to_segment(xs, ys, p1, p2):
    a = p2 - p1
    t = ((xs - p1[0]) * a[0] + (ys - p1[1]) * a[1]) / ((a ** 2).sum() + 1e-6)
    t = np.clip(t, 0, 1)
    return np.sqrt((xs - (p1[0] + t * a[0])) ** 2 + (ys - (p1[1] + t * a[1])) ** 2)


def _offset_poly(poly, dist):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(np.round(poly).astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    res = pco.Execute(dist)
    return np.array(res[0]) if res else None


def make_db_targets(polys: Sequence[np.ndarray], ignore: Sequence[bool], h: int, w: int,
                    shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7,
                    max_shrink=0.3) -> Dict[str, np.ndarray]:
    """polys: list of (N,2) float arrays in the (already resized) image frame. ignore[i]=True masks that region."""
    gt = np.zeros((h, w), np.float32)
    gt_mask = np.ones((h, w), np.float32)
    thresh = np.zeros((h, w), np.float32)
    thresh_mask = np.zeros((h, w), np.float32)
    for poly, ign in zip(polys, ignore):
        poly = np.asarray(poly, np.float32)
        area = abs(cv2.contourArea(poly))
        peri = cv2.arcLength(poly, True)
        if area < 4 or peri < 4:
            continue
        if ign:
            cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 0)
            continue
        # DB's shrink distance, capped at max_shrink * short side. Uncapped, a long line (w >> h) loses ~84% of its
        # height, leaving a 1-2 px stripe that postprocessing discards; with the cap the kernel keeps >= 40% of the
        # height and unclip_ratio=1.5 restores both words and long lines to their original size.
        short = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2])) if len(poly) == 4 \
            else min(cv2.minAreaRect(poly)[1])
        d = min(area * (1 - shrink_ratio ** 2) / peri, max_shrink * short)
        shrunk = _offset_poly(poly, -d)
        if shrunk is None or len(shrunk) < 3:
            cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 0)  # too small to shrink: ignore rather than mislabel
            continue
        cv2.fillPoly(gt, [shrunk.astype(np.int32)], 1.0)
        exp = _offset_poly(poly, d)
        if exp is None:
            continue
        x0, y0 = np.maximum(exp.min(0), 0).astype(int)
        x1, y1 = np.minimum(exp.max(0), [w - 1, h - 1]).astype(int)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        n = len(poly)
        dists = np.stack([_dist_to_segment(xs, ys, poly[i], poly[(i + 1) % n]) for i in range(n)]).min(0)
        border = 1.0 - np.clip(dists / d, 0, 1)
        thresh[y0:y1 + 1, x0:x1 + 1] = np.maximum(thresh[y0:y1 + 1, x0:x1 + 1], border)
        cv2.fillPoly(thresh_mask, [exp.astype(np.int32)], 1.0)
    thresh = thresh * (thresh_max - thresh_min) + thresh_min
    return {"gt": gt[None], "gt_mask": gt_mask[None], "thresh_map": thresh[None], "thresh_mask": thresh_mask[None]}


# --------------------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------------------
def resize_for_detector(img_bgr: np.ndarray, long_side=DEFAULT_LONG_SIDE):
    """Resize so both sides are multiples of 32. Use the SAME function when building training samples."""
    h, w = img_bgr.shape[:2]
    s = long_side / max(h, w)
    nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), (w / nw, h / nh)


def normalize_page(img_bgr: np.ndarray) -> torch.Tensor:
    img = (img_bgr[:, :, ::-1].astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)


def preprocess_page(img_bgr: np.ndarray, long_side=DEFAULT_LONG_SIDE):
    """Returns tensor (1,3,H,W) and (scale_x, scale_y) to map back to original pixels."""
    img, scale = resize_for_detector(img_bgr, long_side)
    return normalize_page(img)[None], scale


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Return points as top-left, top-right, bottom-right, bottom-left."""
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _box_score(prob: np.ndarray, pts: np.ndarray) -> float:
    """Mean probability inside the box, computed on the box's bounding window only (fast)."""
    H, W = prob.shape
    x0, y0 = np.clip(np.floor(pts.min(0)).astype(int), 0, [W - 1, H - 1])
    x1, y1 = np.clip(np.ceil(pts.max(0)).astype(int), 0, [W - 1, H - 1])
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [(pts - [x0, y0]).astype(np.int32)], 1)
    return float(cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], mask)[0])


def db_postprocess(prob: np.ndarray, bin_thresh=0.3, box_thresh=0.6, unclip_ratio=1.5,
                   min_size=3, max_candidates=1000) -> Tuple[List[np.ndarray], List[float]]:
    """prob: (H,W) in [0,1]. Returns quads (4,2) in prob-map coordinates and their scores."""
    bitmap = (prob > bin_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads, scores = [], []
    for c in contours[:max_candidates]:
        if len(c) < 4:
            continue
        rect = cv2.minAreaRect(c)
        if min(rect[1]) < min_size:
            continue
        pts = cv2.boxPoints(rect)
        score = _box_score(prob, pts)
        if score < box_thresh:
            continue
        area, peri = cv2.contourArea(pts), cv2.arcLength(pts, True)
        grown = _offset_poly(pts, area * unclip_ratio / max(peri, 1e-6))
        if grown is None:
            continue
        rect2 = cv2.minAreaRect(grown.astype(np.float32))
        if min(rect2[1]) < min_size + 2:
            continue
        quads.append(_order_quad(cv2.boxPoints(rect2)))
        scores.append(score)
    return quads, scores


@torch.no_grad()
def detect_lines(model: DBNetPP, img_bgr: np.ndarray, device="cpu", long_side=DEFAULT_LONG_SIDE, **pp_kwargs):
    """Full-page inference. Returns quads in original-image pixels and scores."""
    model.eval()
    x, (sx, sy) = preprocess_page(img_bgr, long_side)
    prob = model(x.to(device))["prob"][0, 0].float().cpu().numpy()
    quads, scores = db_postprocess(prob, **pp_kwargs)
    quads = [q * np.array([sx, sy], np.float32) for q in quads]
    return quads, scores


# --------------------------------------------------------------------------------------
# Tuning helper
# --------------------------------------------------------------------------------------
def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / union) if union > 0 else 0.0


def tune_unclip_ratio(prob_maps: List[np.ndarray], gt_quads: List[List[np.ndarray]],
                      ratios=(1.4, 1.6, 1.8, 2.0, 2.2, 2.4)) -> Dict[float, float]:
    """
    prob_maps: detector probability maps for held-out pages; gt_quads: ground-truth quads in the SAME frame.
    Returns {ratio: mean best-match IoU}. Pick the ratio with the highest score.
    """
    res = {}
    for r in ratios:
        ious = []
        for prob, gts in zip(prob_maps, gt_quads):
            preds, _ = db_postprocess(prob, unclip_ratio=r)
            for g in gts:
                ious.append(max([_quad_iou(np.asarray(g), p) for p in preds], default=0.0))
        res[r] = float(np.mean(ious)) if ious else 0.0
    return res


if __name__ == "__main__":
    m = DBNetPP(pretrained_backbone=False)
    print("params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
    m.train()
    o = m(torch.randn(2, 3, 256, 256))
    print({k: v.shape for k, v in o.items()})
