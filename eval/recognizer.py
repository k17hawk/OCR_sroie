"""
Recognition metrics: CER, WER, confidence-vs-error AUC, per-character confusion.

All functions are pure and independent of the training loop. `evaluate_recognizer` takes a list of
prediction records (dicts with `text`, `ref`, and optionally `conf_min`/`conf_mean`) and returns a summary.
"""
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance on any sequences (str or list). Two-row DP, O(min(|a|,|b|)) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """Character error rate. Empty reference -> 0.0 if pred empty, else 1.0."""
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    """Word error rate (whitespace-tokenised)."""
    p, r = pred.split(), ref.split()
    if not r:
        return 0.0 if not p else 1.0
    return levenshtein(p, r) / len(r)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    AUC where higher score means more likely positive. Ties contribute 0.5. Returns NaN if one class is
    empty. O(n_pos * n_neg) — fine for eval sizes (a few thousand lines).
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def conf_error_auc(confs: Sequence[float], errors: Sequence[int]) -> float:
    """
    AUC of "1 - conf" as a detector of errored lines. Well-calibrated recognizer -> near 1.0; a
    confidence that carries no information -> 0.5.
    """
    scores = [1.0 - c for c in confs]
    return _auc(scores, errors)


def confusion_report(pairs: Iterable[Tuple[str, str]], top: int = 30) -> List[Tuple[str, str, int]]:
    """
    Top (ref_char, pred_char) substitutions at matched positions using a simple per-character alignment.
    Returns a list of (ref, pred, count), highest first. Useful for spotting systematic Devanagari errors
    (matras, shirorekha variants, ZWJ/ZWNJ).
    """
    edits: Counter = Counter()
    for ref, pred in pairs:
        # Simple alignment: use DP with backtracking on a small alphabet to extract substitutions.
        m, n = len(ref), len(pred)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if ref[i - 1] == pred[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        i, j = m, n
        while i > 0 and j > 0:
            cost = 0 if ref[i - 1] == pred[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    edits[(ref[i - 1], pred[j - 1])] += 1
                i, j = i - 1, j - 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                i -= 1
            else:
                j -= 1
    return [(r, p, c) for (r, p), c in edits.most_common(top)]


def evaluate_recognizer(records: Sequence[Dict]) -> Dict[str, float]:
    """
    records: list of {"text": str, "ref": str, "conf_min": float (optional), "conf_mean": float (optional)}.
    Returns overall CER, WER, mean confidences, and conf-error AUCs if confidences are present.
    Also returns a breakdown for short vs. long lines: CTC tends to fail more on very long lines, and it is
    useful to see that separately.
    """
    if not records:
        return {"n": 0, "cer": float("nan"), "wer": float("nan")}

    cers = [cer(r["text"], r["ref"]) for r in records]
    wers = [wer(r["text"], r["ref"]) for r in records]
    errors = [1 if r["text"] != r["ref"] else 0 for r in records]
    out = {
        "n": len(records),
        "cer": float(sum(cers) / len(cers)),
        "wer": float(sum(wers) / len(wers)),
        "line_em": float(sum(1 - e for e in errors) / len(errors)),
        "total_edits": int(sum(levenshtein(r["text"], r["ref"]) for r in records)),
        "total_chars": int(sum(len(r["ref"]) for r in records)),
    }
    confs = [r.get("conf_min") for r in records if r.get("conf_min") is not None]
    if len(confs) == len(records):
        out["conf_min_mean"] = float(sum(confs) / len(confs))
        out["auc_conf_min"] = float(conf_error_auc(confs, errors))
        # Quartile error rates are more interpretable than AUC for a first look.
        order = sorted(range(len(confs)), key=lambda i: confs[i])
        q = max(1, len(order) // 4)
        out["err_rate_low_conf_quartile"] = float(sum(errors[i] for i in order[:q]) / q)
        out["err_rate_high_conf_quartile"] = float(sum(errors[i] for i in order[-q:]) / q)
    # Short / long line split (a common failure mode for CTC on long Devanagari lines).
    for lo, hi, key in [(0, 60, "short"), (60, 150, "medium"), (150, 10**9, "long")]:
        sel = [(r, c) for r, c in zip(records, cers) if lo <= len(r["ref"]) < hi]
        if sel:
            out[f"cer_{key}"] = float(sum(c for _, c in sel) / len(sel))
            out[f"n_{key}"] = len(sel)
    return out