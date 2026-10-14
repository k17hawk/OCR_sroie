#!/usr/bin/env python3
"""
corpus_tools.py — diagnose and clean a text corpus for Nepali/English OCR training.

Two subcommands:
  diagnose   report every category of noise in a corpus (read-only)
  clean      drop noise, normalize the rest, write a new file

Both stream and both work on files bigger than RAM.

Usage:
  python corpus_tools.py diagnose --input nepali_clean.txt --sample 1000000
  python corpus_tools.py diagnose --input nepali_clean.txt english_clean.txt --all
  python corpus_tools.py clean    --input nepali_clean.txt --output nepali_final.txt --workers 8
  python corpus_tools.py clean    --input nepali.txt english.txt --output all_final.txt --workers 8
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from multiprocessing import Pool

# --------------------------------------------------------------------------
# Patterns. These are the same checks generate.py / prepare_lines.py already
# use, extended with a few the pipeline did not previously cover.
# --------------------------------------------------------------------------
TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][A-Za-z0-9]*[^<>]{0,300}?>")
ENTITY_RE = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[A-Za-z]{2,8});")
URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|np|gov|edu|io|co|info)\b", re.I)
SLUG_RE = re.compile(r"\S*-\S*-\S*")            # 2+ hyphens in a token
DEV_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ZW_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
REP_RE = re.compile(r"(.)\1{5,}")               # 6+ identical chars
LATIN_SHORT_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]{1,3}(?![A-Za-z])")
DIGITS_ONLY_RE = re.compile(r"^[\s\d\u0966-\u096f.,%+\-/()]+$")
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE00-\uFE0F]")

# Devanagari-validity checks (same as ocr_lines.devanagari_problem)
_MATRA = set(chr(c) for c in list(range(0x093E, 0x094D)) +
             [0x094E, 0x094F, 0x0955, 0x0956, 0x0957, 0x0962, 0x0963])
_VIRAMA, _NUKTA = "\u094d", "\u093c"
_SIGNS = set("\u0901\u0902\u0903")


def _is_consonant(c):
    o = ord(c)
    return 0x0915 <= o <= 0x0939 or 0x0958 <= o <= 0x095F or 0x0978 <= o <= 0x097F


def broken_devanagari(text):
    prev = prev2 = " "
    for c in text:
        if c in _MATRA or c == _VIRAMA:
            ok = _is_consonant(prev) or (prev == _NUKTA and _is_consonant(prev2))
            if not ok:
                return "matra_or_virama_without_consonant"
        elif c == _NUKTA:
            if not _is_consonant(prev):
                return "nukta_without_consonant"
        elif c in _SIGNS:
            o = ord(prev)
            if not (0x0900 <= o <= 0x097F) or prev in _SIGNS or prev == _VIRAMA:
                return "sign_without_letter"
        elif c == "\u25cc":
            return "dotted_circle"
        prev2, prev = prev, c
    return None


def latin_fragments(text):
    """Broken Preeti-style: Devanagari text with 3+ isolated short Latin tokens."""
    if not DEV_RE.search(text):
        return False
    return len(LATIN_SHORT_RE.findall(text)) >= 3


# --------------------------------------------------------------------------
# Per-line issue detection. Returns a set of issue tags (empty if clean).
# --------------------------------------------------------------------------
def issues_of(raw, min_len=30, max_len=2000):
    tags = set()
    line = raw.rstrip("\r\n")

    if not line.strip():
        return {"empty"}
    if len(line) < min_len:
        tags.add("too_short")
    if len(line) > max_len:
        tags.add("too_long")

    if TAG_RE.search(line):
        tags.add("html_tag")
    if ENTITY_RE.search(line):
        tags.add("html_entity")
    if URL_RE.search(line):
        tags.add("url")
    # slug: token with 2+ hyphens, only flag when Devanagari is present
    if DEV_RE.search(line):
        for tok in line.split():
            if len(tok) > 8 and SLUG_RE.search(tok):
                tags.add("url_slug")
                break
    if CTRL_RE.search(line):
        tags.add("control_char")
    if ZW_RE.search(line):
        tags.add("zero_width")
    if unicodedata.normalize("NFC", line) != line:
        tags.add("non_nfc")
    if EMOJI_RE.search(line):
        tags.add("emoji")
    if REP_RE.search(line):
        tags.add("repeated_char")
    if DIGITS_ONLY_RE.match(line):
        tags.add("digits_only")

    if DEV_RE.search(line) and broken_devanagari(line):
        tags.add("broken_dev")
    if latin_fragments(line):
        tags.add("latin_fragments")

    return tags


# --------------------------------------------------------------------------
# Line-level cleanup (applied to lines that survive the drop rules).
# --------------------------------------------------------------------------
def normalize_line(line):
    line = line.strip()
    line = unicodedata.normalize("NFC", line)
    line = ZW_RE.sub("", line)
    # collapse runs of whitespace
    line = re.sub(r"[ \t\u00a0]+", " ", line)
    # if this is Devanagari text, collapse 3+ dots to a danda
    if DEV_RE.search(line):
        line = re.sub(r"\.{3,}", "\u0964", line)
    return line.strip()


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_lines(path):
    with open(path, "rb") as f:
        for i, raw in enumerate(f, 1):
            try:
                yield i, raw.decode("utf-8")
            except UnicodeDecodeError:
                yield i, None      # signal: not UTF-8


# --------------------------------------------------------------------------
# diagnose
# --------------------------------------------------------------------------
def cmd_diagnose(args):
    inputs = args.input
    total_bytes = sum(len_path(p) for p in inputs)
    total_lines = 0
    counts = Counter()
    samples = {}
    lang = Counter()

    for path in inputs:
        print(f"Reading {path} ...", file=sys.stderr, flush=True)
        for _, line in read_lines(path):
            total_lines += 1
            if line is None:
                counts["utf8_bad"] += 1
                continue
            tags = issues_of(line, args.min_len, args.max_len)
            for t in tags:
                counts[t] += 1
                samples.setdefault(t, line.strip()[:120])

            # language bucket (only for lines with no other issue)
            if not tags:
                has_dev = bool(DEV_RE.search(line))
                has_lat = bool(LATIN_RE.search(line))
                n_dev = len(DEV_RE.findall(line))
                n_lat = len(LATIN_RE.findall(line))
                if has_dev and has_lat:
                    lang["mixed"] += 1
                elif has_dev:
                    lang["ne"] += 1
                elif has_lat:
                    lang["en"] += 1
                else:
                    lang["other"] += 1

            if args.sample and total_lines >= args.sample:
                break
        if args.sample and total_lines >= args.sample:
            break

    # ---- report ----
    print()
    print("=" * 78)
    print(f"Corpus diagnosis: {total_lines:,} lines "
          f"({total_bytes / 1e9:.2f} GB on disk)")
    if args.sample:
        print(f"(sampled: first {total_lines:,} lines)")
    print("=" * 78)
    print()
    print(f"{'issue':18} {'count':>12} {'rate':>10}   example")
    print("-" * 78)
    for tag in sorted(counts, key=lambda t: -counts[t]):
        c = counts[tag]
        rate = c / max(total_lines, 1)
        ex = samples.get(tag, "")
        print(f"{tag:18} {c:>12,} {rate:>9.4%}   {ex[:40]}")
    if not counts:
        print("(no issues found)")
    print()
    print("Language mix (of clean lines only):")
    for k in ("ne", "en", "mixed", "other"):
        if lang[k]:
            print(f"  {k:6} {lang[k]:>12,}  {lang[k] / max(total_lines, 1):>9.2%}")
    print()
    print("Guidance:")
    print("  - lines with any of the above tags will be dropped by `clean`")
    print("  - too_short / too_long thresholds: see --min-len / --max-len")
    print("  - non_nfc is normalized, not dropped")
    print()


def len_path(p):
    import os
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


# --------------------------------------------------------------------------
# clean
# --------------------------------------------------------------------------
CHUNK = 20000           # lines per work unit


def _worker(payload):
    """Clean a chunk of lines. Returns (kept, drops) where drops is a Counter."""
    lines, min_len, max_len = payload
    kept = []
    drops = Counter()
    for line in lines:
        if line is None:
            drops["utf8_bad"] += 1
            continue
        tags = issues_of(line, min_len, max_len)
        # normalize first so non_nfc is not counted as a drop
        tags.discard("non_nfc")
        tags.discard("zero_width")          # stripped by normalize_line
        if tags:
            for t in tags:
                drops[t] += 1
            continue
        out = normalize_line(line)
        if len(out) < min_len:
            drops["too_short_after_clean"] += 1
            continue
        kept.append(out)
    return kept, drops


def chunks(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def cmd_clean(args):
    lines_iter = _read_all(args.input)
    tasks = ((chunk, args.min_len, args.max_len)
             for chunk in chunks((ln for _, ln in lines_iter), CHUNK))

    kept_total = 0
    dropped = Counter()
    seen_total = 0

    def counter_of(buf):
        return sum(1 for _ in buf)

    with open(args.output, "w", encoding="utf-8") as fo, \
            Pool(args.workers) as pool:
        for i, (kept, drops) in enumerate(
                pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            for line in kept:
                fo.write(line + "\n")
            kept_total += len(kept)
            for k, v in drops.items():
                dropped[k] += v
            seen_total = kept_total + sum(dropped.values())
            if i % 50 == 0:
                print(f"  ... {seen_total:>12,} lines in, "
                      f"{kept_total:>12,} kept", file=sys.stderr, flush=True)

    total = kept_total + sum(dropped.values())
    print()
    print("=" * 78)
    print(f"Cleaning complete: {total:,} lines in, {kept_total:,} kept "
          f"({kept_total / max(total, 1):.2%})")
    print(f"Written: {args.output}")
    print("=" * 78)
    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {k:24} {v:>12,}  ({v / max(total, 1):.3%})")
    print()


def _read_all(paths):
    for p in paths:
        print(f"Reading {p} ...", file=sys.stderr, flush=True)
        yield from read_lines(p)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="report noise without modifying anything")
    d.add_argument("--input", nargs="+", required=True)
    d.add_argument("--sample", type=int, default=1_000_000,
                   help="read only the first N lines (default 1M); use --all for everything")
    d.add_argument("--all", action="store_true", help="read the whole file")
    d.add_argument("--min-len", type=int, default=30)
    d.add_argument("--max-len", type=int, default=2000)

    c = sub.add_parser("clean", help="drop noise and write a new file")
    c.add_argument("--input", nargs="+", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--min-len", type=int, default=30)
    c.add_argument("--max-len", type=int, default=2000)

    args = ap.parse_args()
    if args.cmd == "diagnose":
        if args.all:
            args.sample = 0
        cmd_diagnose(args)
    else:
        cmd_clean(args)


if __name__ == "__main__":
    main()