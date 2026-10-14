"""
Table metrics.

  structure_exact_match : the token sequence produced by the model exactly matches the ground-truth tokens.
  cell_text_em          : exact-match of the text inside each cell, in row/col order, when the cell counts
                          line up. If they do not, it is 0.0 — you must fix structure first.
  evaluate_table        : aggregates the above over a list of predictions and references.

The reference is expected to be the same shape that comes out of `merge_table`:
    {"tokens": [...], "cells": [{"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "..."}, ...]}
"""
from typing import Dict, List, Sequence

from eval.recognizer import cer


def _strip_special(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if t not in ("<sos>", "<eos>", "<pad>")]


def structure_exact_match(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return 1.0 if _strip_special(pred_tokens) == _strip_special(gt_tokens) else 0.0


def _cells_in_order(cells: Sequence[Dict]) -> List[Dict]:
    return sorted(cells, key=lambda c: (c.get("row", 0), c.get("col", 0)))


def cell_text_em(pred_cells: Sequence[Dict], gt_cells: Sequence[Dict]) -> Dict[str, float]:
    """
    When the cell counts and (row, col) keys line up, compare texts; otherwise return zeros with `aligned=0`.
    Also returns per-cell CER and a "no empty mismatch" flag (empty in ref but non-empty in pred is usually
    a merge or reread bug worth surfacing separately).
    """
    p = _cells_in_order(pred_cells)
    g = _cells_in_order(gt_cells)
    if len(p) != len(g):
        return {"aligned": 0.0, "em": 0.0, "cer": 1.0, "empty_mismatch": 1.0}
    # Align by (row, col) when possible; fall back to order.
    p_by_rc = {(c["row"], c["col"]): c for c in p}
    g_by_rc = {(c["row"], c["col"]): c for c in g}
    if set(p_by_rc) == set(g_by_rc):
        pairs = [(p_by_rc[k], g_by_rc[k]) for k in g_by_rc]
    else:
        pairs = list(zip(p, g))
    n = len(pairs)
    em = sum(1 for pc, gc in pairs if (pc.get("text") or "") == (gc.get("text") or "")) / n
    cers = [cer(pc.get("text") or "", gc.get("text") or "") for pc, gc in pairs]
    empty_mismatch = sum(1 for pc, gc in pairs
                         if bool((pc.get("text") or "").strip()) != bool((gc.get("text") or "").strip())) / n
    return {"aligned": 1.0, "em": float(em), "cer": float(sum(cers) / n), "empty_mismatch": float(empty_mismatch)}


def evaluate_table(preds: Sequence[Dict], refs: Sequence[Dict]) -> Dict[str, float]:
    """
    preds/refs: one dict per table, each with `tokens` (list of structure tokens) and `cells` (list of cell
    dicts with at least `row`, `col`, `text`). Returns aggregate metrics.
    """
    assert len(preds) == len(refs), "predictions and references must align"
    n = len(preds)
    if n == 0:
        return {"n": 0}
    struct = [structure_exact_match(p.get("tokens", []), r.get("tokens", [])) for p, r in zip(preds, refs)]
    cell_ems, cell_cers, empty_mm = [], [], []
    for p, r in zip(preds, refs):
        m = cell_text_em(p.get("cells", []), r.get("cells", []))
        cell_ems.append(m["em"])
        cell_cers.append(m["cer"])
        empty_mm.append(m["empty_mismatch"])
    return {
        "n": n,
        "structure_em": float(sum(struct) / n),
        "cell_text_em": float(sum(cell_ems) / n),
        "cell_text_cer": float(sum(cell_cers) / n),
        "cell_empty_mismatch": float(sum(empty_mm) / n),
    }