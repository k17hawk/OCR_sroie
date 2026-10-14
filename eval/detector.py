"""
Line-detector metrics: quad IoU, greedy matching, H-mean precision/recall at several IoU thresholds
(the ICDAR-2015 style metric for text detection).
"""
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two convex quads (or polygons)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 1e-6 else 0.0


def match_quads(pred_quads: Sequence[np.ndarray], pred_scores: Sequence[float],
                gt_quads: Sequence[np.ndarray], iou_thresh: float = 0.5
                ) -> Tuple[List[int], List[int], List[int], List[float]]:
    """
    Greedy one-to-one matching: predictions are considered in descending score order, each is matched to
    the best unused ground-truth quad with IoU >= iou_thresh.
    Returns (tp_pred_indices, fp_pred_indices, fn_gt_indices, ious_of_tps).
    """
    order = list(np.argsort(-np.asarray(pred_scores, np.float64))) if len(pred_scores) else []
    used_gt = set()
    tp_p, fp_p, ious = [], [], []
    for pi in order:
        best_gt, best_iou = -1, 0.0
        for gi, g in enumerate(gt_quads):
            if gi in used_gt:
                continue
            iou = quad_iou(pred_quads[pi], g)
            if iou > best_iou:
                best_iou, best_gt = iou, gi
        if best_gt >= 0 and best_iou >= iou_thresh:
            used_gt.add(best_gt)
            tp_p.append(pi)
            ious.append(best_iou)
        else:
            fp_p.append(pi)
    fn_g = [gi for gi in range(len(gt_quads)) if gi not in used_gt]
    return tp_p, fp_p, fn_g, ious


def evaluate_detector(pred_pages, gt_pages, iou_threshs=(0.5, 0.7, 0.9)) -> Dict[str, float]:
    assert len(pred_pages) == len(gt_pages), "pages must align"
    out: Dict[str, float] = {}
    for t in iou_threshs:
        tp = fp = fn = 0
        matched_ious: List[float] = []
        for (quads, scores), gts in zip(pred_pages, gt_pages):
            tp_i, fp_i, fn_i, ious = match_quads(quads, scores, gts, iou_thresh=t)
            tp += len(tp_i); fp += len(fp_i); fn += len(fn_i)
            matched_ious.extend(ious)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        h = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        key = f"iou{int(t * 100):02d}"
        out[f"{key}_precision"] = float(prec)
        out[f"{key}_recall"] = float(rec)
        out[f"{key}_hmean"] = float(h)
        out[f"{key}_mean_iou"] = float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0
        out[f"{key}_tp"] = int(tp); out[f"{key}_fp"] = int(fp); out[f"{key}_fn"] = int(fn)
    return out