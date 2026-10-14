"""
Stage 1 - Layout detector.

A trainable PyTorch-native detector (torchvision Faster R-CNN, MobileNetV3-Large FPN) with anchors tuned for
document blocks. For best accuracy you can instead fine-tune PP-DocLayout / RT-DETR weights (Apache 2.0);
DocLayout-YOLO is AGPL, check that licence before shipping it.

Anchors: torchvision's aspect ratio is h / w, so 0.05 means a 20:1 wide box (footers, single-line headings,
short wide tables) and 4.0 a 1:4 tall box (signatures, vertical stamps).

Training: model.train(); losses = model(images, targets); sum(losses.values()).backward()
  images : list of float tensors (3,H,W) in [0,1]
  targets: list of {"boxes": (N,4) float xyxy, "labels": (N,) int64}  -> build with layout_targets_from_json()
"""
from typing import Dict, List

import numpy as np
import torch
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead


LAYOUT_CLASSES = ["text", "heading", "list", "table", "figure", "caption",
                  "header", "footer", "stamp", "signature", "form","handwriting"]  # label id = index + 1 (0 = background)

# Generator type names -> LAYOUT_CLASSES. Unknown names raise, so a new generator type can't be silently dropped.
TYPE_ALIASES = {
    "text": "text", "paragraph": "text", "para": "text", "plain": "text",
    "heading": "heading", "title": "heading", "subheading": "heading",
    "list": "list", "list_item": "list",
    "table": "table",
    "figure": "figure", "chart": "figure", "image": "figure", "logo": "figure",
    "caption": "caption", "signature_caption": "caption",
    "header": "header", "page_header": "header",
    "footer": "footer", "page_footer": "footer", "page_number": "footer",
    "stamp": "stamp", "seal": "stamp",
    "signature": "signature",
    "handwriting": "text",
    "form": "form", "form_field": "form", "key_value": "form",
}
ANCHOR_SIZES = ((32, 64, 128, 256, 512),) * 3
ANCHOR_RATIOS = ((0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),) * 3


def label_of(type_name: str) -> int:
    key = TYPE_ALIASES.get(type_name.lower())
    if key is None:
        raise KeyError(f"unknown layout type '{type_name}': add it to TYPE_ALIASES")
    return LAYOUT_CLASSES.index(key) + 1


def build_layout_detector(num_classes: int = len(LAYOUT_CLASSES), pretrained_backbone: bool = True,
                          min_size: int = 1024, max_size: int = 1600):
    wb = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=wb, num_classes=num_classes + 1,
        min_size=min_size, max_size=max_size, box_detections_per_img=200)
    model.rpn.anchor_generator = AnchorGenerator(ANCHOR_SIZES, ANCHOR_RATIOS)
    model.rpn.head = RPNHead(model.backbone.out_channels, num_anchors=len(ANCHOR_SIZES[0]) * len(ANCHOR_RATIOS[0]))
    return model


def layout_targets_from_json(page: Dict, scale: float = 1.0) -> Dict[str, torch.Tensor]:
    """Build a torchvision target from one page JSON (its `layout` list). `scale` = resized / original."""
    boxes, labels = [], []
    for blk in page.get("layout", []):
        x1, y1, x2, y2 = [float(v) * scale for v in blk["bbox"]]
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(label_of(blk["type"]))
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64)}


@torch.no_grad()
def detect_layout(model, img_bgr: np.ndarray, device="cpu", score_thresh=0.5, nms_iou=0.5) -> List[Dict]:
    """Returns [{type, bbox [x1,y1,x2,y2], score, (alt)}] in original-image pixels, after postprocess_layout."""
    model.eval()
    rgb = torch.from_numpy(img_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    out = model([rgb.to(device)])[0]
    return postprocess_layout(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu(), score_thresh, nms_iou)


# Drawn ON TOP of other content (and often on each other: a signature over a round stamp), so they are never
# suppressed by, and never suppress, boxes of another class.
OVERLAY_CLASSES = {"stamp", "signature", "handwriting"}


def _drop_nested_same_class(idx: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor, names: List[str],
                            min_contained: float) -> torch.Tensor:
    """
    Remove a box that lies >= min_contained inside a higher-scoring box of the SAME class. NMS misses these
    because a half-size duplicate inside a full box has IoU < 0.5 (e.g. a table box covering half of a table).
    Different-class nesting is left alone: a footer overlapping the bottom of a list box is real content.
    """
    order = idx[torch.argsort(scores[idx], descending=True)].tolist()
    kept: List[int] = []
    for i in order:
        bi = boxes[i]
        area_i = float((bi[2] - bi[0]) * (bi[3] - bi[1]))
        nested = False
        for k in kept:
            if names[k] != names[i]:
                continue
            bk = boxes[k]
            ix = float(min(bi[2], bk[2]) - max(bi[0], bk[0]))
            iy = float(min(bi[3], bk[3]) - max(bi[1], bk[1]))
            area_k = float((bk[2] - bk[0]) * (bk[3] - bk[1]))
            if ix > 0 and iy > 0 and ix * iy / max(min(area_i, area_k), 1e-6) >= min_contained:
                nested = True
                break
        if not nested:
            kept.append(i)
    return torch.tensor(kept, dtype=torch.long)


def postprocess_layout(boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, score_thresh: float = 0.5,
                       nms_iou: float = 0.5, min_contained: float = 0.8) -> List[Dict]:
    """
    1. Drop boxes below score_thresh.
    2. Region classes (everything except overlay classes): class-AGNOSTIC NMS, so one region gets one label.
       The best suppressed competitor is kept as "alt" ({type, score}) on the winning box: a small margin means
       the label is uncertain (e.g. list vs table) and worth a second check downstream.
    3. Region classes: drop a box that is >= min_contained inside a higher-scoring box of the same class
       (nested duplicates, e.g. a half-table box inside the full table box).
    4. Overlay classes (stamp, signature, handwriting): drawn on top of other content, so NMS only within the
       same class; never suppressed by, and never suppressing, other classes.
    Returns blocks sorted by score (highest first).
    """
    from torchvision.ops import batched_nms, box_iou, nms
    keep = scores >= score_thresh
    boxes, labels, scores = boxes[keep].float(), labels[keep], scores[keep].float()
    if boxes.numel() == 0:
        return []
    names = [LAYOUT_CLASSES[int(l) - 1] for l in labels]
    overlay = torch.tensor([n in OVERLAY_CLASSES for n in names], dtype=torch.bool)
    reg = torch.nonzero(~overlay).flatten()
    ovl = torch.nonzero(overlay).flatten()
    kept_reg = reg[nms(boxes[reg], scores[reg], nms_iou)] if len(reg) else reg
    if len(kept_reg) > 1:
        kept_reg = _drop_nested_same_class(kept_reg, boxes, scores, names, min_contained)
    kept_ovl = ovl[batched_nms(boxes[ovl], scores[ovl], labels[ovl], nms_iou)] if len(ovl) else ovl
    kept = torch.cat([kept_reg, kept_ovl])
    kept = kept[torch.argsort(scores[kept], descending=True)]

    alt: Dict[int, Dict] = {}
    if len(kept_reg) and len(reg) > len(kept_reg):
        dropped = torch.tensor([i for i in reg.tolist() if i not in set(kept_reg.tolist())])
        ious = box_iou(boxes[dropped], boxes[kept_reg])  # (n_dropped, n_kept)
        for r, d in enumerate(dropped.tolist()):
            k = int(kept_reg[int(ious[r].argmax())])
            if ious[r].max() >= nms_iou and names[d] != names[k] and \
                    (k not in alt or scores[d] > alt[k]["score"]):
                alt[k] = {"type": names[d], "score": round(float(scores[d]), 4)}
    out = []
    for i in kept.tolist():
        b = {"type": names[i], "bbox": [round(float(v), 1) for v in boxes[i]], "score": round(float(scores[i]), 4)}
        if i in alt:
            b["alt"] = alt[i]
        out.append(b)
    return out


if __name__ == "__main__":
    m = build_layout_detector(pretrained_backbone=False, min_size=320, max_size=480)
    print("params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
    m.train()
    imgs = [torch.rand(3, 300, 400)]
    tg = [{"boxes": torch.tensor([[20., 20., 380., 40.], [20., 100., 380., 250.]]),
           "labels": torch.tensor([label_of("footer"), label_of("table")])}]
    losses = m(imgs, tg)
    print({k: round(v.item(), 3) for k, v in losses.items()})