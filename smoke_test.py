"""
Run: python smoke_test.py
CPU is fine. Models are randomly initialised (no weight downloads), so recognised text is junk by design;
this checks shapes, losses, wiring and every fix from the code review.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

from detector_dbnet import DBNetPP, db_loss, db_postprocess, make_db_targets
from layout_detector import build_layout_detector, label_of, layout_targets_from_json
from merge import block_text, cells_to_grid, grid_to_csv, merge_table, order_blocks
from pipeline import OCRPipeline, crop_quad, estimate_skew, split_wide_crop, _join_chunks
from recognizer_svtr import (CTCCodec, SVTRRecognizer, build_charset, check_ctc_feasible, collate_lines,
                             ctc_loss_fn, resize_pad_line)
from table_slanet import (TableStructureModel, TableVocab, collate_tables, html_to_tokens, prepare_table_sample,
                          table_loss, table_sample_from_json, tokens_to_cells, tokens_to_html)

torch.manual_seed(0)
NOPRE = dict(pretrained_backbone=False)  # no downloads in the smoke test; use True for real training


def ok(name, cond=True):
    if not cond:
        print(f"[FAIL] {name}")
        sys.exit(1)
    print(f"[ok] {name}")


# ---------------- recognizer ----------------
codec = CTCCodec(build_charset(["abc०१२३", "पुर्‍याइदिने"]))
ok("ZWJ kept in charset", "‍" in codec.chars)
try:
    codec.encode(["abcx"])
    ok("strict encode rejects unknown chars", False)
except ValueError:
    ok("strict encode rejects unknown chars")

rec = SVTRRecognizer(codec.num_classes)
x = torch.randn(4, 3, 48, 256)
tg, tl = codec.encode(["abc", "०१२", "ab", "c३"])
il = SVTRRecognizer.out_lengths(torch.full((4,), 256))
loss, stats = ctc_loss_fn(rec(x, torch.full((4,), 256)), tg, tl, il)
loss.backward()
ok(f"CTC step, loss={loss.item():.3f}, skipped={stats['skipped']}", stats["skipped"] == 0)

# infeasible sample is detected (not silently zeroed)
tg2, tl2 = codec.encode(["abcabcabcabc"])
_, st2 = ctc_loss_fn(rec(torch.randn(1, 3, 48, 16)), tg2, tl2, SVTRRecognizer.out_lengths(torch.tensor([16])))
ok("infeasible CTC sample counted as skipped", st2["skipped"] == 1)

# long line from your sample: 957x21 px, ~150 chars -> feasible at height 48
fe, info = check_ctc_feasible("क" * 150, 957, 21)
ok(f"957x21 line feasible (T={info['T']}, need={info['needed']})", fe)

# padding no longer changes a line's output
rec.eval()
img = (np.random.rand(48, 120, 3) * 255).astype(np.uint8)
t, w = resize_pad_line(img)
with torch.no_grad():
    alone = rec(*collate_lines([t], [w]))[0, :30]
    other = resize_pad_line((np.random.rand(48, 600, 3) * 255).astype(np.uint8))
    batched = rec(*collate_lines([t, other[0]], [w, other[1]]))[0, :30]
ok(f"batch-independent output (max diff {float((alone - batched).abs().max()):.2e})",
   torch.allclose(alone, batched, atol=1e-4))

# ---------------- detector ----------------
H, W = 256, 384
polys = [np.array([[30, 40], [300, 40], [300, 80], [30, 80]], np.float32)]
tgt = make_db_targets(polys, [False], H, W)
det = DBNetPP(**NOPRE).train()
out = det(torch.randn(1, 3, H, W))
l, parts = db_loss(out, {k: torch.from_numpy(v)[None] for k, v in tgt.items()})
l.backward()
ok(f"DB loss {parts}")
if torch.cuda.is_available():
    with torch.autocast("cuda"):
        o = det.cuda()(torch.randn(1, 3, H, W).cuda())
        db_loss(o, {k: torch.from_numpy(v)[None].cuda() for k, v in tgt.items()})
    det.cpu()
    ok("DB loss under AMP")
prob = np.zeros((H, W), np.float32); prob[40:80, 30:300] = 0.9
ok("DB postprocess finds the box", len(db_postprocess(prob)[0]) == 1)

# ---------------- layout ----------------
lay = build_layout_detector(**NOPRE, min_size=320, max_size=480).train()
ls = lay([torch.rand(3, 300, 400)], [{"boxes": torch.tensor([[20., 20., 380., 35.]]), "labels": torch.tensor([label_of("footer")])}])
ok(f"layout losses {{{', '.join(f'{k}: {v.item():.3f}' for k, v in ls.items())}}}")
try:
    label_of("mystery_block")
    ok("unknown layout type raises", False)
except KeyError:
    ok("unknown layout type raises")

# ---------------- table ----------------
v = TableVocab()
gt_html = ('<table><tr><th>क्र.सं.</th><th>नाम</th><th>बजेट</th></tr>'
           '<tr><td>१</td><td rowspan="2">क</td><td>१०</td></tr><tr><td>२</td><td>२०</td></tr>'
           '<tr><td colspan="2">जम्मा</td><td>३०</td></tr></table>')
toks = html_to_tokens(gt_html, v)
ok("leading <th> row becomes <thead>", toks[0] == "<thead>" and "</thead>" in toks)
texts = ["क्र.सं.", "नाम", "बजेट", "१", "क", "१०", "२", "२०", "जम्मा", "३०"]
ok("tokens_to_html round-trips your GT html exactly", tokens_to_html(toks, texts) == gt_html)
cells_l = tokens_to_cells(toks)
ok("header flags + spans recovered", cells_l[0]["header"] and not cells_l[3]["header"]
   and cells_l[4]["rowspan"] == 2 and cells_l[8]["colspan"] == 2)

html2 = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
cells = [dict(row=0, col=0, rowspan=1, colspan=1, bbox=[0, 0, 100, 30]), dict(row=0, col=1, rowspan=1, colspan=1, bbox=[100, 0, 200, 30]),
         dict(row=1, col=0, rowspan=1, colspan=1, bbox=[0, 30, 100, 60]), dict(row=1, col=1, rowspan=1, colspan=1, bbox=[100, 30, 200, 60])]
smp = prepare_table_sample(np.full((60, 200, 3), 255, np.uint8), html2, cells, v, S=256)
b = collate_tables([smp, smp], v.pad)
tm = TableStructureModel(v, **NOPRE).train()
lg, bx = tm(b["image"], b["tokens"])
tl_, tp = table_loss(lg, bx, b["tokens"], b["boxes"], b["box_mask"], v.pad)
tl_.backward()
ok(f"table step {tp}")

# ---------------- merge ----------------
toks2 = html_to_tokens(html2, v)
boxes = np.array([c["bbox"] for c in cells], np.float32)
lines = [{"bbox": [10, 5, 60, 25], "text": "नाम", "conf": 0.97}, {"bbox": [110, 5, 160, 25], "text": "Age", "conf": 0.62},
         {"bbox": [10, 35, 60, 55], "text": "राम", "conf": 0.91}, {"bbox": [110, 35, 160, 55], "text": "२५", "conf": 0.88}]
m = merge_table(toks2, boxes, [0.99] * 4, lines)
ok("merge html " + m["html"], m["html"].startswith("<table><tr><th>नाम</th>"))
ok("review flags " + str([c["needs_review"] for c in m["cells"]]), [c["needs_review"] for c in m["cells"]] == [False, True, False, False])
# a line that straddles two cells is re-read per cell
lines_s = [{"bbox": [10, 35, 190, 55], "text": "राम२५", "conf": 0.9}]
m2 = merge_table(toks2, boxes, [0.99] * 4, lines_s, reread=lambda box: (f"cell@{int(box[0])}", 0.95))
ok("straddling line re-read per cell " + str(cells_to_grid(m2["cells"])), m2["cells"][2]["text"] == "cell@0" and m2["cells"][3]["text"] == "cell@100")

blocks = [{"bbox": [0, 0, 600, 50]}, {"bbox": [0, 60, 280, 300]}, {"bbox": [320, 60, 600, 300]},
          {"bbox": [240, 310, 360, 330]}, {"bbox": [0, 340, 280, 400]}, {"bbox": [320, 340, 600, 400]}]
ok("reading order with centred heading", [blocks.index(x) for x in order_blocks(blocks, 600)] == [0, 1, 2, 3, 4, 5])
lst = {"type": "list", "lines": [{"bbox": [0, 0, 500, 20], "text": "१. पहिलो"}, {"bbox": [0, 25, 300, 45], "text": "लाइन"},
                                 {"bbox": [0, 50, 500, 70], "text": "२. दोस्रो"}]}
ok("list serialised like GT", block_text(lst) == "१. पहिलो लाइन\n२. दोस्रो")

# ---------------- line crops ----------------
digit = np.array([[121.7, 158.4], [126.6, 158.4], [126.6, 168.7], [121.7, 168.6]], np.float32)  # '१' from your page
page = np.full((300, 300, 3), 255, np.uint8)
ok("single digit crop is not rotated", crop_quad(page, digit).shape[0] > crop_quad(page, digit).shape[1])

wide = np.full((40, 8000, 3), 0, np.uint8)  # ~3.75x max_width at height 48 and NO gaps at all
pieces, gaps = split_wide_crop(wide, 48, 2560)
widths = [int(round(p.shape[1] * 48 / 40)) for p in pieces]
ok(f"gapless wide line splits into {len(pieces)} pieces, max {max(widths)} px <= 2560", max(widths) <= 2560 and not any(gaps))
for p in pieces:
    resize_pad_line(cv2.cvtColor(p, cv2.COLOR_BGR2RGB), 48, 2560)  # raises if too wide
spaced = np.full((40, 8000, 3), 255, np.uint8)
for x0 in range(0, 8000, 120):
    spaced[8:32, x0:x0 + 100] = 0
_, gaps2 = split_wide_crop(spaced, 48, 2560)
ok("cuts in whitespace are flagged as gaps", all(gaps2))
ok("chunk joining", _join_chunks(["रा", "म"], [False]) == "राम" and _join_chunks(["राम", "श्याम"], [True]) == "राम श्याम")

# ---------------- deskew sign ----------------
lines_img = np.full((800, 600, 3), 255, np.uint8)
for y in range(60, 760, 30):
    cv2.line(lines_img, (40, y), (560, y), (0, 0, 0), 4)
Mrot = cv2.getRotationMatrix2D((300, 400), 2.0, 1.0)
tilted = cv2.warpAffine(lines_img, Mrot, (600, 800), borderValue=(255, 255, 255))
a = estimate_skew(tilted)
ok(f"deskew estimates {a:+.2f} deg for a +2.00 deg tilt (undoes it)", abs(a + 2.0) <= 0.25)

# ---------------- full pipeline ----------------
pipe = OCRPipeline(detector=DBNetPP(**NOPRE), recognizer=SVTRRecognizer(codec.num_classes), codec=codec,
                   layout=build_layout_detector(**NOPRE, min_size=320, max_size=480), table=TableStructureModel(v, **NOPRE),
                   det_long_side=512)
res = pipe(np.full((300, 400, 3), 255, np.uint8))
ok(f"pipeline runs, keys {sorted(res.keys())}")
img = np.full((200, 3200, 3), 255, np.uint8)
cv2.putText(img, "abc " * 40, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
quads = [np.array([[10, 30], [200, 30], [200, 100], [10, 100]], np.float32),
         np.array([[5, 30], [3195, 30], [3195, 100], [5, 100]], np.float32)]  # resizes to ~2190 px wide: fits
r = pipe.recognize_lines(img, quads, [1.0, 1.0])
ok(f"recognize_lines: chunks per line {[x['chunks'] for x in r]}")
huge = np.full((40, 12000, 3), 255, np.uint8)
r2 = pipe.recognize_crops([huge])
ok(f"12000 px line handled without crash ({r2[0]['chunks']} chunks)", r2[0]["chunks"] >= 5)
pipe.do_preprocess = True
res2 = pipe(tilted)
ok("pipeline with deskew returns its matrix", res2["deskew_matrix"] is not None)

# ---------------- your sample page (if present) ----------------
for path in ("attachment.txt", "page_0017574.json"):
    if os.path.exists(path):
        pj = json.load(open(path, encoding="utf-8"))
        tg = layout_targets_from_json(pj)
        ok(f"layout targets from {path}: {tg['labels'].tolist()}")
        canvas = np.full((pj["height"], pj["width"], 3), 255, np.uint8)
        for t in pj["tables"]:
            s = table_sample_from_json(canvas, t, v)
            back = tokens_to_html([v.tokens[i] for i in s["tokens"][1:-1].tolist()], [c["text"] for c in sorted(t["cells"], key=lambda c: (c["row"], c["col"]))])
            ok(f"table {t['html'][:30]}... round-trips", back == t["html"])
        break

print("ALL OK")
