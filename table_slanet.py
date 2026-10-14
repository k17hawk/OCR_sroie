"""
Stage 4 - Table structure recognizer, SLANet-style:
  CNN backbone + FPN neck  ->  GRU decoder with attention  ->  (a) HTML structure tokens, (b) one box per cell.

It reads the table *layout* only (rows, columns, spans, cell boxes). Cell *text* comes from the line recognizer
and is merged in later (merge.py), so this model is script-agnostic.

Token scheme (cells carry no text):
    <thead> </thead> <tbody> </tbody> <tr> </tr>
    <td></td>                              simple cell   (box predicted here)
    <td  rowspan="k" colspan="k"  >  </td> spanned cell  (box predicted at '>')

Header rows: your generator writes headers as <th> in the first row(s) with no <thead>. html_to_tokens()
wraps the leading all-<th> rows in <thead> and the rest in <tbody>, so the model learns headers and
tokens_to_html() writes them back as <th>.

Dataset: table_sample_from_json() turns one entry of page["tables"] + the page image into a training sample.

Size: S=640 by default. Wide, short tables (e.g. 977x144) lose row resolution when letterboxed into a square;
use S=768 or split very wide tables if row boxes come out poorly.
"""
import re
from html import escape
from html.parser import HTMLParser
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import FrozenBatchNorm2d

DEFAULT_S = 640
_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# --------------------------------------------------------------------------------------
# Vocabulary and HTML <-> tokens
# --------------------------------------------------------------------------------------
class TableVocab:
    def __init__(self, max_span: int = 20):
        self.max_span = max_span
        base = ["<pad>", "<sos>", "<eos>", "<thead>", "</thead>", "<tbody>", "</tbody>", "<tr>", "</tr>",
                "<td></td>", "<td", ">", "</td>"]
        spans = [f' rowspan="{k}"' for k in range(2, max_span + 1)] + \
                [f' colspan="{k}"' for k in range(2, max_span + 1)]
        self.tokens = base + spans
        self.t2i = {t: i for i, t in enumerate(self.tokens)}
        self.pad, self.sos, self.eos = self.t2i["<pad>"], self.t2i["<sos>"], self.t2i["<eos>"]

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def is_cell_token(tok: str) -> bool:
        return tok in ("<td></td>", ">")


class _HTMLToTokens(HTMLParser):
    """Collects rows; remembers whether each row is all <th> and whether explicit thead/tbody were used."""

    def __init__(self, max_span):
        super().__init__()
        self.max_span = max_span
        self.rows: List[Dict] = []  # {"section": "thead"/"tbody"/None, "tokens": [...], "all_th": bool}
        self.section, self.cur_row, self.cur_cell = None, None, None
        self.explicit = False

    def handle_starttag(self, tag, attrs):
        if tag in ("thead", "tbody"):
            self.section, self.explicit = tag, True
        elif tag == "tr":
            self.cur_row = {"section": self.section, "tokens": [], "all_th": True, "n": 0}
        elif tag in ("td", "th") and self.cur_row is not None:
            a = dict(attrs)
            self.cur_cell = (int(a.get("rowspan") or 1), int(a.get("colspan") or 1))
            self.cur_row["all_th"] &= tag == "th"
            self.cur_row["n"] += 1

    def handle_endtag(self, tag):
        if tag in ("thead", "tbody"):
            self.section = None
        elif tag == "tr" and self.cur_row is not None:
            self.rows.append(self.cur_row)
            self.cur_row = None
        elif tag in ("td", "th") and self.cur_cell is not None and self.cur_row is not None:
            rs, cs = min(self.cur_cell[0], self.max_span), min(self.cur_cell[1], self.max_span)
            t = self.cur_row["tokens"]
            if rs == 1 and cs == 1:
                t.append("<td></td>")
            else:
                t.append("<td")
                if rs > 1:
                    t.append(f' rowspan="{rs}"')
                if cs > 1:
                    t.append(f' colspan="{cs}"')
                t += [">", "</td>"]
            self.cur_cell = None


def html_to_tokens(html: str, vocab: TableVocab) -> List[str]:
    p = _HTMLToTokens(vocab.max_span)
    p.feed(html)
    rows = p.rows
    if p.explicit:
        sections = [r["section"] or "tbody" for r in rows]
    else:  # leading rows made only of <th> are the header
        n_head = 0
        while n_head < len(rows) and rows[n_head]["all_th"] and rows[n_head]["n"] > 0:
            n_head += 1
        if n_head == len(rows):  # a table of only <th> cells: treat as body
            n_head = 0
        sections = ["thead"] * n_head + ["tbody"] * (len(rows) - n_head)
    out, open_sec = [], None
    for r, sec in zip(rows, sections):
        if sec != open_sec:
            if open_sec:
                out.append(f"</{open_sec}>")
            out.append(f"<{sec}>")
            open_sec = sec
        out += ["<tr>"] + r["tokens"] + ["</tr>"]
    if open_sec:
        out.append(f"</{open_sec}>")
    return out


def tokens_to_html(tokens: Sequence[str], cell_texts: Optional[Sequence[str]] = None,
                   keep_sections: bool = False) -> str:
    """
    Rebuild HTML. Cells inside <thead> become <th>. cell_texts are in cell (HTML) order.
    keep_sections=False drops <thead>/<tbody> wrappers so output matches your ground-truth format exactly.
    """
    out, in_head, i, ci = ["<table>"], False, 0, 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("<thead>", "</thead>", "<tbody>", "</tbody>"):
            in_head = t == "<thead>" or (in_head and t not in ("</thead>", "<tbody>"))
            if keep_sections:
                out.append(t)
        elif t in ("<tr>", "</tr>"):
            out.append(t)
        elif t in ("<td></td>", "<td"):
            attrs = ""
            if t == "<td":
                j = i + 1
                while j < len(tokens) and tokens[j] != ">":
                    attrs += tokens[j]
                    j += 1
                i = j
                if i + 1 < len(tokens) and tokens[i + 1] == "</td>":
                    i += 1
            text = escape(cell_texts[ci], quote=False) if cell_texts is not None and ci < len(cell_texts) else ""
            tag = "th" if in_head else "td"
            out.append(f"<{tag}{attrs}>{text}</{tag}>")
            ci += 1
        i += 1
    out.append("</table>")
    return "".join(out)


def tokens_to_cells(tokens: Sequence[str]) -> List[Dict]:
    """Recover (row, col, rowspan, colspan, header) for each cell, in HTML order."""
    cells, occ, r, c, in_head, i = [], set(), -1, 0, False, 0
    while i < len(tokens):
        t = tokens[i]
        if t == "<thead>":
            in_head = True
        elif t in ("</thead>", "<tbody>"):
            in_head = False
        elif t == "<tr>":
            r, c = r + 1, 0
        elif t in ("<td></td>", "<td"):
            rs = cs = 1
            if t == "<td":
                j = i + 1
                while j < len(tokens) and tokens[j] != ">":
                    m = re.search(r'(row|col)span="(\d+)"', tokens[j])
                    if m:
                        rs, cs = (int(m.group(2)), cs) if m.group(1) == "row" else (rs, int(m.group(2)))
                    j += 1
                i = j
                if i + 1 < len(tokens) and tokens[i + 1] == "</td>":
                    i += 1
            r = max(r, 0)
            while (r, c) in occ:
                c += 1
            cells.append({"row": r, "col": c, "rowspan": rs, "colspan": cs, "header": in_head})
            occ.update((r + dr, c + dc) for dr in range(rs) for dc in range(cs))
            c += cs
        i += 1
    return cells


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------
def cells_in_html_order(cells: List[Dict]) -> List[Dict]:
    return sorted(cells, key=lambda c: (c["row"], c["col"]))


def letterbox(crop_bgr: np.ndarray, S: int = DEFAULT_S) -> torch.Tensor:
    """Keep aspect, place top-left on a white SxS canvas, ImageNet-normalise. Same for training and inference."""
    h, w = crop_bgr.shape[:2]
    s = S / max(h, w)
    img = cv2.resize(crop_bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((S, S, 3), 255, np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    x = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return (x - _MEAN) / _STD


def prepare_table_sample(crop_bgr: np.ndarray, html: str, cells: List[Dict], vocab: TableVocab,
                         S: int = DEFAULT_S, max_len: int = 600) -> Dict[str, torch.Tensor]:
    """
    cells[i]["bbox"] = [x1, y1, x2, y2] in crop pixels. Raises ValueError for samples to skip.
    """
    toks = html_to_tokens(html, vocab)
    ordered = cells_in_html_order(cells)
    n_cell_tok = sum(vocab.is_cell_token(t) for t in toks)
    if n_cell_tok != len(ordered):
        raise ValueError(f"html has {n_cell_tok} cells but cells[] has {len(ordered)}")
    if len(toks) + 2 > max_len:
        raise ValueError(f"table too long: {len(toks) + 2} tokens > {max_len}")
    h, w = crop_bgr.shape[:2]
    s = S / max(h, w)
    ids = [vocab.sos] + [vocab.t2i[t] for t in toks] + [vocab.eos]
    boxes = torch.zeros(len(ids), 4)
    bmask = torch.zeros(len(ids), dtype=torch.bool)
    ci = 0
    for p, t in enumerate(["<sos>"] + toks + ["<eos>"]):
        if vocab.is_cell_token(t):
            x1, y1, x2, y2 = [v * s / S for v in ordered[ci]["bbox"]]
            boxes[p] = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]).clamp(0, 1)
            bmask[p] = True
            ci += 1
    return {"image": letterbox(crop_bgr, S), "tokens": torch.tensor(ids), "boxes": boxes, "box_mask": bmask}


def table_sample_from_json(page_bgr: np.ndarray, table: Dict, vocab: TableVocab, S: int = DEFAULT_S,
                           pad: int = 4) -> Dict[str, torch.Tensor]:
    """One entry of page["tables"] -> training sample. Cell boxes are moved from page to crop coordinates."""
    H, W = page_bgr.shape[:2]
    x1, y1, x2, y2 = table["bbox"]
    x1, y1 = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
    x2, y2 = min(int(np.ceil(x2)) + pad, W), min(int(np.ceil(y2)) + pad, H)
    crop = page_bgr[y1:y2, x1:x2]
    cells = [{**c, "bbox": [c["bbox"][0] - x1, c["bbox"][1] - y1, c["bbox"][2] - x1, c["bbox"][3] - y1]}
             for c in table["cells"]]
    return prepare_table_sample(crop, table["html"], cells, vocab, S)


def collate_tables(samples: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    L = max(len(s["tokens"]) for s in samples)
    B = len(samples)
    tokens = torch.full((B, L), pad_id, dtype=torch.long)
    boxes = torch.zeros(B, L, 4)
    bmask = torch.zeros(B, L, dtype=torch.bool)
    for i, s in enumerate(samples):
        n = len(s["tokens"])
        tokens[i, :n], boxes[i, :n], bmask[i, :n] = s["tokens"], s["boxes"], s["box_mask"]
    return {"image": torch.stack([s["image"] for s in samples]), "tokens": tokens, "boxes": boxes, "box_mask": bmask}


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def sincos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device).float() / d))
    ey = torch.arange(h, device=device).float()[:, None] * omega[None]
    ex = torch.arange(w, device=device).float()[:, None] * omega[None]
    ey, ex = torch.cat([ey.sin(), ey.cos()], 1), torch.cat([ex.sin(), ex.cos()], 1)
    pos = torch.cat([ey[:, None].expand(h, w, 2 * d), ex[None].expand(h, w, 2 * d)], -1)
    return pos.reshape(h * w, dim)


class TableStructureModel(nn.Module):
    def __init__(self, vocab: TableVocab, feat_dim=128, hidden=256, att_dim=256, emb_dim=64,
                 pretrained_backbone=True):
        super().__init__()
        self.vocab = vocab
        # No BatchNorm anywhere: train-mode BN uses per-batch statistics, and letterboxed tables are mostly white
        # padding in varying amounts, so batch statistics swing from batch to batch and never match the running
        # averages used in eval mode. An autoregressive decoder then derails on the first wrong token. Pretrained:
        # BN frozen at its ImageNet statistics (same in train and eval). From scratch: GroupNorm.
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        norm = FrozenBatchNorm2d if pretrained_backbone else (lambda c: nn.GroupNorm(32, c))
        r = torchvision.models.resnet18(weights=weights, norm_layer=norm)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat3, self.lat4, self.lat5 = (nn.Conv2d(c, feat_dim, 1) for c in (128, 256, 512))
        self.out = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.GroupNorm(32, feat_dim), nn.ReLU(True))
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.init_h = nn.Linear(feat_dim, hidden)
        self.att_f, self.att_h, self.att_v = nn.Linear(feat_dim, att_dim), nn.Linear(hidden, att_dim), nn.Linear(att_dim, 1)
        self.gru = nn.GRUCell(emb_dim + feat_dim, hidden)
        self.struct_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(True), nn.Linear(hidden, len(vocab)))
        self.loc_head = nn.Sequential(nn.Linear(hidden + feat_dim, hidden), nn.ReLU(True), nn.Linear(hidden, 4))

    def encode(self, x):
        c2 = self.l1(self.stem(x)); c3 = self.l2(c2); c4 = self.l3(c3); c5 = self.l4(c4)
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        f = self.out(p4 + F.adaptive_avg_pool2d(p3, p4.shape[-2:]))  # stride 16
        B, C, H, W = f.shape
        return f.flatten(2).transpose(1, 2) + sincos_2d(H, W, C, f.device)[None]  # (B, N, C)

    def step(self, tok, h, feats, pf):
        a = self.att_v(torch.tanh(pf + self.att_h(h)[:, None])).squeeze(-1).softmax(-1)
        ctx = torch.bmm(a[:, None], feats).squeeze(1)
        h = self.gru(torch.cat([self.embed(tok), ctx], -1), h)
        return h, ctx

    def forward(self, images, tokens):
        """Teacher forcing. Returns logits (B, L-1, V) and boxes (B, L-1, 4) for targets tokens[:, 1:]."""
        feats = self.encode(images)
        pf = self.att_f(feats)
        h = torch.tanh(self.init_h(feats.mean(1)))
        logits, boxes = [], []
        for t in range(tokens.size(1) - 1):
            h, ctx = self.step(tokens[:, t], h, feats, pf)
            logits.append(self.struct_head(h))
            boxes.append(self.loc_head(torch.cat([h, ctx], -1)).sigmoid())
        return torch.stack(logits, 1), torch.stack(boxes, 1)

    @torch.no_grad()
    def predict(self, images, max_len=600):
        """
        Greedy decoding. Returns one dict per image:
          tokens, token_probs, cell_boxes (n,4) normalised xyxy in the SxS letterboxed frame, cell_probs (n,)
        """
        self.eval()
        V = self.vocab
        feats = self.encode(images)
        pf = self.att_f(feats)
        B = images.size(0)
        h = torch.tanh(self.init_h(feats.mean(1)))
        tok = torch.full((B,), V.sos, dtype=torch.long, device=images.device)
        res = [{"tokens": [], "token_probs": [], "cell_boxes": [], "cell_probs": []} for _ in range(B)]
        done = [False] * B
        for _ in range(max_len):
            h, ctx = self.step(tok, h, feats, pf)
            lg = self.struct_head(h).float()
            lg[:, V.pad] = float("-inf")
            lg[:, V.sos] = float("-inf")
            p = lg.softmax(-1)
            tok = p.argmax(-1)
            box = self.loc_head(torch.cat([h, ctx], -1)).sigmoid()
            for b in range(B):
                if done[b]:
                    continue
                t = int(tok[b])
                if t == V.eos:
                    done[b] = True
                    continue
                name, pr = V.tokens[t], float(p[b, t])
                res[b]["tokens"].append(name)
                res[b]["token_probs"].append(pr)
                if V.is_cell_token(name):
                    cx, cy, w, hh = box[b].tolist()
                    res[b]["cell_boxes"].append([cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2])
                    res[b]["cell_probs"].append(pr)
            if all(done):
                break
        return res


def table_loss(logits, boxes, tokens, tok_boxes, box_mask, pad_id, box_weight=2.0):
    """logits/boxes come from forward(); targets are shifted by one. Computed in fp32 (AMP-safe)."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, boxes = logits.float(), boxes.float()
        V = logits.size(-1)
        ce = F.cross_entropy(logits.reshape(-1, V), tokens[:, 1:].reshape(-1), ignore_index=pad_id)
        m = box_mask[:, 1:].float()
        l1 = (F.smooth_l1_loss(boxes, tok_boxes[:, 1:].float(), reduction="none", beta=0.02).sum(-1) * m).sum() / (m.sum() + 1e-6)
    return ce + box_weight * l1, {"struct_ce": ce.item(), "box_l1": l1.item()}


def boxes_to_crop_pixels(norm_xyxy: np.ndarray, crop_w: int, crop_h: int, S: int = DEFAULT_S) -> np.ndarray:
    """Undo the letterbox: normalised boxes in the SxS frame -> pixel boxes in the original table crop."""
    s = S / max(crop_w, crop_h)
    b = np.asarray(norm_xyxy, np.float32).reshape(-1, 4) * S / s
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, crop_w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, crop_h)
    return b


if __name__ == "__main__":
    v = TableVocab()
    html = ('<table><tr><th>क्र.सं.</th><th>नाम</th><th>बजेट</th></tr>'
            '<tr><td>१</td><td rowspan="2">क</td><td>१०</td></tr><tr><td>२</td><td>२०</td></tr>'
            '<tr><td colspan="2">जम्मा</td><td>३०</td></tr></table>')
    toks = html_to_tokens(html, v)
    print(toks)
    print(tokens_to_html(toks, ["क्र.सं.", "नाम", "बजेट", "१", "क", "१०", "२", "२०", "जम्मा", "३०"]))
    print(tokens_to_cells(toks))