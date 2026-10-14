"""
Stage 3 - Line recognizer: SVTR-style hybrid (local conv mixing + global attention) with a CTC head.

Input : (B, 3, 48, W)  line crops, height fixed at 48, width variable (right-padded with WHITE, bucket by width)
        Lines are not squeezed: a 957x21 px, ~150-char line becomes ~2190 px wide (T ~ 547 CTC steps).
        Lines wider than max_width must be split (pipeline.split_wide_crop) or skipped in training.
Output: (B, T, num_classes) logits, T = ceil(W / 4). Class 0 is the CTC blank.

Padding: pass `widths` (unpadded widths) to forward(). Attention then ignores padded columns and padded
features are zeroed after every block, so a line's output does not depend on which other lines share its batch.

Charset: labels are NFC-normalised. encode(strict=True) raises on characters outside the charset instead of
silently dropping them (a dropped character means the label no longer matches the image). Decide deliberately
whether ZWJ (U+200D) / ZWNJ (U+200C) are in your charset; build_charset() keeps them by default.

This follows the SVTR/SVTRv2 recipe (local mixing early, global mixing later, height collapsed to a sequence,
CTC head). It is NOT a weight-compatible copy of PaddleOCR's SVTRv2.
"""
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_VALUE = 1.0  # white in the [-1, 1] normalised space


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# --------------------------------------------------------------------------------------
# Charset <-> indices
# --------------------------------------------------------------------------------------
def build_charset(texts: Iterable[str], keep_joiners: bool = True, min_count: int = 1) -> List[str]:
    """Collect the charset from your label corpus (NFC). Sorted for a stable index order across runs."""
    counts: Dict[str, int] = {}
    for t in texts:
        for c in nfc(t):
            counts[c] = counts.get(c, 0) + 1
    chars = [c for c, n in counts.items() if n >= min_count]
    if not keep_joiners:
        chars = [c for c in chars if c not in ("‌", "‍")]
    return sorted(chars)


class CTCCodec:
    """Index 0 = CTC blank."""

    def __init__(self, charset: Sequence[str]):
        self.chars = list(charset)
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("charset has duplicates")
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.num_classes = len(self.chars) + 1

    def unknown_chars(self, text: str) -> List[str]:
        return sorted({c for c in nfc(text) if c not in self.char2idx})

    def filter(self, text: str) -> str:
        """Keep only characters the codec knows (use only at evaluation time, never for training labels)."""
        return "".join(c for c in nfc(text) if c in self.char2idx)

    def encode(self, texts: List[str], strict: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        flat, lens = [], []
        for t in texts:
            t = nfc(t)
            if strict:
                bad = self.unknown_chars(t)
                if bad:
                    raise ValueError(f"label has characters outside the charset: {bad!r} in {t!r}")
            ids = [self.char2idx[c] for c in t if c in self.char2idx]
            flat.extend(ids)
            lens.append(len(ids))
        return torch.tensor(flat, dtype=torch.long), torch.tensor(lens, dtype=torch.long)

    def decode_greedy(self, logits: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        logits: (B, T, C). Returns list of dicts with text, per-char probs, line confidence.
        conf_min is the weakest character (best signal for 'is any char wrong'); conf_mean is the average.
        A line with no decoded characters gets conf 0.0 and empty=True.
        """
        probs = logits.float().softmax(-1)
        best_p, best_i = probs.max(-1)  # (B, T)
        out = []
        for b in range(logits.size(0)):
            T = int(lengths[b]) if lengths is not None else logits.size(1)
            idx, p = best_i[b, :T].tolist(), best_p[b, :T].tolist()
            chars, cps, prev = [], [], 0
            for i, pi in zip(idx, p):
                if i != 0 and i != prev:
                    chars.append(self.chars[i - 1])
                    cps.append(pi)
                elif i != 0 and i == prev and cps:
                    cps[-1] = max(cps[-1], pi)  # repeated frame of the same char
                prev = i
            out.append({"text": "".join(chars), "char_probs": cps, "empty": not cps,
                        "conf_min": min(cps) if cps else 0.0,
                        "conf_mean": sum(cps) / len(cps) if cps else 0.0})
        return out


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):  # (B, C, H, W) -> normalise over C
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.size(0),) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class LocalMixing(nn.Module):
    """Depthwise conv token mixer + conv MLP. Cheap, keeps positional information."""

    def __init__(self, dim, kernel=(3, 7), mlp_ratio=2.0, drop_path=0.0):
        super().__init__()
        self.n1 = LayerNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, kernel, padding=(kernel[0] // 2, kernel[1] // 2), groups=dim)
        self.n2 = LayerNorm2d(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hid, 1), nn.GELU(), nn.Conv2d(hid, dim, 1))
        self.dp = DropPath(drop_path)

    def forward(self, x, col_mask=None):
        x = x + self.dp(self.dw(self.n1(x)))
        x = x + self.dp(self.mlp(self.n2(x)))
        return x


class GlobalMixing(nn.Module):
    """Multi-head self-attention over all H*W positions + MLP. Padded columns are excluded as keys."""

    def __init__(self, dim, heads=4, mlp_ratio=2.0, drop_path=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.dp = DropPath(drop_path)

    def forward(self, x, col_mask=None):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, N, C), N = H*W, index = h*W + w
        qkv = self.qkv(self.n1(t)).reshape(B, -1, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_mask = None
        if col_mask is not None:  # (B, W) True = real column
            attn_mask = col_mask[:, None, :].expand(B, H, W).reshape(B, 1, 1, H * W)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, heads, N, d)
        a = a.transpose(1, 2).reshape(B, -1, C)
        t = t + self.dp(self.proj(a))
        t = t + self.dp(self.mlp(self.n2(t)))
        return t.transpose(1, 2).reshape(B, C, H, W)


class Downsample(nn.Module):
    """Halve the height only, keep width resolution so each character gets enough time steps."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=(2, 1), padding=1)
        self.norm = LayerNorm2d(cout)

    def forward(self, x, col_mask=None):
        return self.norm(self.conv(x))


# --------------------------------------------------------------------------------------
# Recognizer
# --------------------------------------------------------------------------------------
class SVTRRecognizer(nn.Module):
    """
    Default config: dims (128, 256, 384). Stage plan (L = local, G = global): s1 LLL | s2 LGLG | s3 LGG.
    Feature heights (in_height=48): 48 -> 12 (stem) -> 6 -> 3, then collapsed to a width-T sequence.
    For faster CPU inference on long lines use plan=("LLL", "LLLL", "LGG") (global attention only at height 3)
    and measure accuracy before/after.
    """

    def __init__(self, num_classes: int, dims=(128, 256, 384),
                 plan=("LLL", "LGLG", "LGG"), heads=(4, 8, 8),
                 mlp_ratio=2.0, drop_path=0.1, in_height=48, dropout=0.1):
        super().__init__()
        assert in_height % (4 * 2 ** (len(dims) - 1)) == 0, "in_height must divide by 4 * 2^(stages-1)"
        self.in_height = in_height
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, 3, stride=2, padding=1), LayerNorm2d(dims[0] // 2), nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1), LayerNorm2d(dims[0]), nn.GELU(),
        )
        total = sum(len(p) for p in plan)
        dpr = [drop_path * i / max(total - 1, 1) for i in range(total)]
        blocks, k = [], 0
        for si, (dim, pat) in enumerate(zip(dims, plan)):
            if si > 0:
                blocks.append(Downsample(dims[si - 1], dim))
            for ch in pat:
                blocks.append(LocalMixing(dim, mlp_ratio=mlp_ratio, drop_path=dpr[k]) if ch == "L"
                              else GlobalMixing(dim, heads[si], mlp_ratio, drop_path=dpr[k]))
                k += 1
        self.blocks = nn.ModuleList(blocks)
        self.final_h = in_height // 4 // (2 ** (len(dims) - 1))
        self.collapse = nn.Sequential(nn.Linear(dims[-1] * self.final_h, dims[-1]), nn.GELU(), nn.LayerNorm(dims[-1]))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, widths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x (B,3,H,W); widths (B,) unpadded pixel widths. Returns logits (B, T, num_classes)."""
        x = self.stem(x)  # (B, C, h, T)
        col_mask = None
        if widths is not None:
            T = x.shape[-1]
            valid = self.out_lengths(widths.to(x.device)).clamp(max=T)
            col_mask = torch.arange(T, device=x.device)[None, :] < valid[:, None]  # (B, T)
            x = x * col_mask[:, None, None, :].to(x.dtype)
        for blk in self.blocks:
            x = blk(x, col_mask)
            if col_mask is not None:
                x = x * col_mask[:, None, None, :].to(x.dtype)  # keep padded columns at exactly zero
        B, C, h, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * h)
        return self.head(self.drop(self.collapse(x)))

    @staticmethod
    def out_lengths(widths: torch.Tensor) -> torch.Tensor:
        """Number of CTC time steps for each (unpadded) image width: ceil(w / 4)."""
        return torch.clamp(torch.div(widths + 3, 4, rounding_mode="floor"), min=1)


# --------------------------------------------------------------------------------------
# CTC helpers
# --------------------------------------------------------------------------------------
def ctc_min_steps(text: str) -> int:
    """Minimum CTC time steps a label needs: one per char plus one blank between each adjacent repeat."""
    return len(text) + sum(1 for a, b in zip(text, text[1:]) if a == b)


def ctc_time_steps(img_w: int, img_h: int, height: int = 48) -> int:
    """T the model will produce for a crop of size img_w x img_h resized to `height` (no width cap)."""
    new_w = max(8, int(round(img_w * height / max(img_h, 1))))
    return max(1, (new_w + 3) // 4)


def check_ctc_feasible(text: str, img_w: int, img_h: int, height: int = 48, max_width: int = 2560,
                       margin: float = 1.2):
    """
    Call in your dataloader on the NFC label and skip or split samples that fail.
    ok is False when the line is wider than max_width after resizing, or when T < margin * ctc_min_steps(text).
    """
    text = nfc(text)
    new_w = int(round(img_w * height / max(img_h, 1)))
    T, need = ctc_time_steps(img_w, img_h, height), ctc_min_steps(text)
    ok = new_w <= max_width and T >= margin * need
    return ok, {"resized_width": new_w, "T": T, "needed": need, "too_wide": new_w > max_width}


def ctc_loss_fn(logits, targets, target_lengths, input_lengths):
    """
    logits (B,T,C), blank=0. Infeasible samples (T < label length + repeats) are detected exactly and excluded
    from the average instead of being silently zeroed. Returns (loss, stats); log stats["skipped"] every step:
    it should stay at 0 if your dataloader runs check_ctc_feasible.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lp = logits.float().log_softmax(-1).transpose(0, 1)  # (T,B,C)
        need, off = [], 0
        tcpu = targets.cpu()
        for L in target_lengths.tolist():
            seq = tcpu[off:off + L]
            off += L
            need.append(L + int((seq[1:] == seq[:-1]).sum()))
        bad = input_lengths.cpu() < torch.tensor(need, dtype=input_lengths.dtype)
        per = F.ctc_loss(lp, targets, input_lengths, target_lengths, blank=0, reduction="none", zero_infinity=True)
        good = (~bad).to(per.device)
        if bool(good.any()):
            loss = (per[good] / target_lengths.to(per.device).clamp(min=1)[good]).mean()
        else:
            loss = per.sum() * 0.0
    return loss, {"skipped": int(bad.sum()), "batch": len(need)}


# --------------------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------------------
def resize_line_uint8(img_rgb: np.ndarray, height=48, max_width=2560, on_overflow="raise") -> np.ndarray:
    """Aspect-preserving resize to `height` (uint8 HxWx3). Shared by training loaders and inference."""
    import cv2
    h, w = img_rgb.shape[:2]
    natural = int(round(w * height / max(h, 1)))
    if natural > max_width and on_overflow == "raise":
        raise ValueError(f"line is {natural}px wide at height {height} (> max_width={max_width}); split it first")
    new_w = max(8, min(max_width, natural))
    return cv2.resize(img_rgb, (new_w, height), interpolation=cv2.INTER_AREA if h > height else cv2.INTER_LINEAR)


def line_to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB -> float (3,H,W) in [-1, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1).float() / 127.5 - 1.0


def resize_pad_line(img_rgb: np.ndarray, height=48, max_width=2560, pad_to=None, on_overflow="raise"):
    """
    numpy HxWx3 uint8 -> float tensor (3, height, W'), normalised to [-1, 1], right-padded with white.
    Keeps aspect ratio. If the natural width exceeds max_width: "raise" (training: split or skip the line) or
    "squeeze" (inference safety net: squeezes width slightly rather than failing the page).
    """
    img = resize_line_uint8(img_rgb, height, max_width, on_overflow)
    new_w = img.shape[1]
    t = line_to_tensor(img)
    if pad_to is not None and pad_to > new_w:
        t = F.pad(t, (0, pad_to - new_w), value=PAD_VALUE)
    return t, new_w


def collate_lines(tensors: List[torch.Tensor], widths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of (3,H,w) tensors with white to a common width (multiple of 4). Returns (batch, widths)."""
    W = max(widths)
    W += (-W) % 4
    batch = torch.stack([F.pad(t, (0, W - t.shape[-1]), value=PAD_VALUE) for t in tensors])
    return batch, torch.tensor(widths, dtype=torch.long)


if __name__ == "__main__":
    codec = CTCCodec(build_charset(["abc"]))
    m = SVTRRecognizer(codec.num_classes)
    print("params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
    print(m(torch.randn(2, 3, 48, 320), torch.tensor([320, 200])).shape)
