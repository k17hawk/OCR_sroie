"""
Adapter between the pipeline and YOUR trained line recognizer.

The pipeline only needs:
    load_recognizer(weights, device, **kwargs) -> object
    object.recognize(list_of_BGR_uint8_arrays) -> [(text, score), ...]   # same order, same length

Fill in the two places marked  >>> EDIT <<<  so they match your training code:
  1. build_model(): create the network exactly as in training and load the weights
  2. CONFIG: input height, width limits, grayscale, normalization, charset, decoding

Everything else (resize + pad, batching by width, CTC greedy decoding, score) is ready.
If your model is attention/transformer-decoder based instead of CTC, replace decode().
"""
import math

import cv2
import numpy as np
import torch

# >>> EDIT 1: preprocessing and decoding must match what you trained with <<<
CONFIG = {
    "img_height": 48,          # height lines were resized to during training
    "max_width": 1600,         # longer lines are squeezed to this width
    "min_width": 32,
    "width_multiple": 4,       # pad width to a multiple of the model's total stride
    "grayscale": True,         # 1-channel input?
    "mean": 0.5, "std": 0.5,   # x = (pixel/255 - mean) / std
    "pad_value": 1.0,          # pad with white (in 0..1 before normalization)
    "charset_file": None,      # txt file, one character per line (index 1..N); None = read from checkpoint
    "blank_index": 0,          # CTC blank
    "batch_size": 64,
    "fp16": True,              # half precision on GPU
}


def build_model(ckpt, cfg, device):
    """>>> EDIT 2: rebuild your network and load the weights <<<

    Example if your training repo has `models/recognizer.py` with `build_recognizer(cfg)`:

        from models.recognizer import build_recognizer
        model = build_recognizer(ckpt["config"])          # or your own arguments
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        return model

    The model's forward(x) must take a float tensor [B, C, H, W] and return logits
    [B, T, num_classes] (or [T, B, num_classes]; both are handled in decode()).
    """
    raise NotImplementedError(
        "Edit build_model() in my_recognizer.py to construct your network "
        "(same code as in training) and load the checkpoint.")


# ----------------------------------------------------------------------
# Ready-made parts
# ----------------------------------------------------------------------
def load_charset(cfg, ckpt):
    if cfg.get("charset_file"):
        with open(cfg["charset_file"], encoding="utf-8") as f:
            chars = [ln.rstrip("\n") for ln in f if ln.rstrip("\n") != ""]
        return ["<blank>"] + chars
    for key in ("charset", "vocab", "alphabet", "chars", "itos"):
        v = ckpt.get(key) if isinstance(ckpt, dict) else None
        if v is None and isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
            v = ckpt["config"].get(key)
        if v:
            chars = list(v) if isinstance(v, str) else list(v)
            return chars if chars and chars[0] in ("<blank>", "", "[blank]", "<pad>") else ["<blank>"] + chars
    raise ValueError("No charset found: set CONFIG['charset_file'] or store 'charset' in the checkpoint")


def preprocess(img, cfg):
    """BGR uint8 line crop -> float array [C, H, W] at the training height."""
    if cfg["grayscale"]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    H = cfg["img_height"]
    W = int(round(w * H / max(1, h)))
    W = max(cfg["min_width"], min(cfg["max_width"], W))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if h > H else cv2.INTER_CUBIC)
    x = img.astype(np.float32) / 255.0
    if x.ndim == 2:
        x = x[None]
    else:
        x = x.transpose(2, 0, 1)
    return x


def pad_batch(arrs, cfg):
    m = cfg["width_multiple"]
    W = max(a.shape[2] for a in arrs)
    W = int(math.ceil(W / m) * m)
    C, H = arrs[0].shape[0], arrs[0].shape[1]
    out = np.full((len(arrs), C, H, W), cfg["pad_value"], np.float32)
    for i, a in enumerate(arrs):
        out[i, :, :, :a.shape[2]] = a
    return (out - cfg["mean"]) / cfg["std"]


def decode(logits, charset, blank):
    """CTC greedy decoding -> [(text, score)]. score = mean max-probability of kept characters."""
    probs = logits.float().softmax(-1)                 # [B, T, C]
    if probs.shape[-1] != len(charset):
        raise ValueError(f"model outputs {probs.shape[-1]} classes but charset has {len(charset)}")
    conf, idx = probs.max(-1)                          # [B, T]
    results = []
    for b in range(idx.shape[0]):
        seq, sc = idx[b].tolist(), conf[b].tolist()
        chars, scores, prev = [], [], None
        for k, s in zip(seq, sc):
            if k != prev and k != blank:
                chars.append(charset[k])
                scores.append(s)
            prev = k
        results.append(("".join(chars), float(np.mean(scores)) if scores else 0.0))
    return results


class Recognizer:
    def __init__(self, weights, device="cuda", **overrides):
        self.cfg = {**CONFIG, **overrides}
        dev = str(device).replace("gpu", "cuda")
        self.device = torch.device(dev if torch.cuda.is_available() or not dev.startswith("cuda") else "cpu")
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.charset = load_charset(self.cfg, ckpt)
        self.model = build_model(ckpt, self.cfg, self.device).to(self.device).eval()
        self.half = self.cfg["fp16"] and self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.time_major = None   # detected on first batch

    @torch.inference_mode()
    def recognize(self, crops):
        if not crops:
            return []
        arrs = [preprocess(c, self.cfg) for c in crops]
        order = sorted(range(len(arrs)), key=lambda i: arrs[i].shape[2])   # similar widths together
        out = [None] * len(arrs)
        bs = self.cfg["batch_size"]
        for s in range(0, len(order), bs):
            ids = order[s:s + bs]
            x = torch.from_numpy(pad_batch([arrs[i] for i in ids], self.cfg)).to(self.device)
            if self.half:
                x = x.half()
            logits = self.model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if isinstance(logits, dict):
                logits = logits.get("logits", next(iter(logits.values())))
            if logits.shape[0] != len(ids) and logits.shape[1] == len(ids):   # [T, B, C] -> [B, T, C]
                logits = logits.transpose(0, 1)
            for i, r in zip(ids, decode(logits, self.charset, self.cfg["blank_index"])):
                out[i] = r
        return out


def load_recognizer(weights, device="cuda", **kwargs):
    return Recognizer(weights, device, **kwargs)