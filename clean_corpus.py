#!/usr/bin/env python3
"""
Clean a raw text corpus before it is used to render synthetic OCR pages.

The generator renders a corpus line and uses the same string as the label, so anything
that renders differently from what the string says (invisible characters, broken matra
order, glyphs a font lacks, Preeti leftovers) becomes a wrong training pair. Every line
this script keeps is safe to render.

Language:
  --script ne (default)   Devanagari + ASCII + common punctuation
  --script en             Latin + Latin-1 + Latin Extended-A/B + Latin punctuation

  When --script is given, sensible defaults follow:
      ne: --min-chars 20, --danda-space tight, --spell drop
      en: --min-chars 15, --danda-space keep,  --spell off

Rules for Nepali (--script ne), from auditing a real corpus:
  1. Normalise: NFC, remove invisible characters (except the ZWJ in र्‍ "eyelash ra",
     as in गर्‍यो), plain spaces, Windows-1252 quote bytes -> proper quotes.
  2. Repair Preeti-conversion leftovers (¥ -> र्‍, ÷ -> /, Œ -> त्त्, ˆ -> फ्, ...) and
     common sequence mistakes (ि before ्र, matra before ्र, ँ/ं before matra, doubled
     signs, visarga after a digit).
  3. Danda: | -> ।, ।। -> ॥, no space before it by default.
  4. Long lines (whole articles) are split into chunks of whole sentences (~700 chars).
  5. A chunk is dropped if it has a character outside the allowed set, an invalid
     Devanagari sequence, Preeti text, a mixed-script word, mostly Latin, Hindi,
     URL/e-mail/HTML, boilerplate, character runs, or mostly symbols.
  6. Lines shorter than --min-chars that pass go to <output>.short.txt.
  7. Exact duplicates are removed.
  8. Spelling: chunks containing a very rare word one edit from a very common word are
     dropped (default) or fixed (--spell fix). --dict adds known-good words.

Rules for English (--script en):
  - Normalise: NFC, plain spaces, Windows-1252 -> proper quotes.
  - Split long lines the same way (into whole sentences packed to ~700 chars).
  - Drop: bad UTF-8, out-of-set characters, URL/e-mail/HTML, boilerplate,
    character runs, non-Latin-heavy lines, symbol-heavy lines.
  - Keep: everything else, including accented Latin (é ñ ü ø ā č š …).

Usage:
  python clean_corpus.py nepali_raw.txt nepali_clean.txt --workers 16 --fonts fonts
  python clean_corpus.py english_raw.txt english_clean.txt --script en --workers 16
  python clean_corpus.py nepali_raw.txt trial.txt --limit 1000000
"""
import argparse
import hashlib
import os
import random
import re
import resource
import threading
import time
import unicodedata
from array import array
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

# --------------------------------------------------------------------------
# Character policy
# --------------------------------------------------------------------------
ZWJ = "\u200d"
EYELASH_RA = "\u0930\u094d\u200d"  # र्‍
INVISIBLE = dict.fromkeys(map(ord, "\ufeff\u00ad\u200b\u200c\u200d\u2060\u200e\u200f"
                                   "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u180e\ufffc"), None)
SPACES = re.compile(r"[\t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
C1_QUOTES = str.maketrans({"\x91": "\u2018", "\x92": "\u2019", "\x93": "\u201c", "\x94": "\u201d",
                           "\x96": "\u2013", "\x97": "\u2014", "\x85": "\u2026"})
PUNCT = str.maketrans({
    "\u201e": "\u201c", "\u201a": "\u2018", "\u2033": '"', "\u2032": "'", "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2212": "-", "\u2015": "\u2014",
    "\uff0c": ",", "\uff1a": ":", "\uff08": "(", "\uff09": ")", "\uff01": "!", "\uff1f": "?",
})

PREETI_SYMBOLS = [
    ("\u00a5", EYELASH_RA, True),          # ¥ -> र्‍
    ("\u00f7", "/", False),                # ÷ -> /
    ("\u0152", "\u0924\u094d\u0924\u094d", True),  # Œ -> त्त्
    ("\u00ec", "\u0924\u094d\u0924\u094d", True),  # ì -> त्त्
    ("\u02c6", "\u092b\u094d", True),      # ˆ -> फ्
]

FIXES = [
    ("\u093e\u0947", "\u094b"),
    ("\u093e\u0948", "\u094c"),
    ("\u0947\u093e", "\u094b"),
    ("\u0948\u093e", "\u094c"),
    ("\u094b\u0947", "\u094b"),
    ("\u0905\u094b", "\u0913"),
    ("\u0905\u094c", "\u0914"),
    ("\u0906\u0947", "\u0913"),
    ("\u0906\u0948", "\u0914"),
    ("\u0905\u093e", "\u0906"),
    ("\u090f\u0947", "\u0910"),
    ("\u0902\u0901", "\u0901"),
    ("\u0901\u0902", "\u0901"),
]

C, V, NUK, HAL = "[\u0915-\u0939]", "[\u0905-\u0914]", "\u093c", "\u094d"
M = "[\u093e-\u094c]"
B = "[\u0901\u0902]?\u0903?"

REGEX_FIXES = [
    ("same sign typed twice", re.compile("([\u0901-\u0903\u093c\u093e-\u094d])\\1+"), r"\1"),
    ("ि before half letter (टि्र -> ट्रि)", re.compile(f"({C})\u093f\u094d({C})"), "\\1\u094d\\2\u093f"),
    ("matra before ्र (गे्र -> ग्रे)", re.compile(f"({C})([\u0941\u0942\u0947\u0948\u094b\u094c])\u094d\u0930"),
     "\\1\u094d\u0930\\2"),
    ("ँ/ं before matra (ँु -> ुँ)", re.compile("([\u0901\u0902])([\u093e-\u094c])"), r"\2\1"),
    ("same sign typed twice", re.compile("([\u0901-\u0903\u093c\u093e-\u094d])\\1+"), r"\1"),
    ("visarga after digit (१ः -> १:)", re.compile("([0-9\u0966-\u096f])\u0903"), r"\1:"),
]


def _ranges(*pairs):
    return {chr(c) for a, b in pairs for c in range(a, b + 1)}


# --- Devanagari + ASCII (Nepali) ----------------------------------------
ALLOWED_NE = (
    _ranges((0x20, 0x7E))
    | _ranges((0x0901, 0x0903))
    | _ranges((0x0905, 0x090B)) | {"\u090f", "\u0910", "\u0911", "\u0913", "\u0914"}
    | _ranges((0x0915, 0x0939))
    | {"\u093c", "\u093d"}
    | _ranges((0x093E, 0x0943)) | {"\u0945", "\u0947", "\u0948", "\u0949", "\u094b", "\u094c"}
    | {"\u094d", "\u0950", "\u0964", "\u0965", "\u0970"}
    | _ranges((0x0966, 0x096F))
    | set("\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u00b0")
    | {ZWJ}
)

# --- Latin + accented Latin (English) -----------------------------------
ALLOWED_EN = (
    _ranges((0x20, 0x7E))                     # Basic Latin printable
    | _ranges((0xA0, 0x024F))                 # Latin-1 Supplement + Latin Extended-A + B
    | set("\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u00b0\u2022\u2020\u2021\u2039\u203a\u2030\u00a9\u00ae\u2122\u00a3\u00a5\u00a2\u20ac\u00a7\u00b6\u00b7")
)

SYL = re.compile(f"(?:{C}{NUK}?{HAL}\u200d?)*{C}{NUK}?(?:{M}|{HAL})?{B}|{V}{B}")
MARK = re.compile("[\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962\u0963]")

DEV_LETTER = re.compile("[\u0905-\u0939]")
DEV_ANY = re.compile("[\u0900-\u0963\u0970-\u097f]")
LAT_LETTER = re.compile("[A-Za-z]")
WORD = re.compile("[\u0900-\u0963\u0970-\u097f]+")
TOKEN_MIXED = re.compile("(?:[A-Za-z][\u0900-\u0963]|[\u0900-\u0963][A-Za-z])")
PREETI_CH = set("][{}\\;`~^")

URL = re.compile(r"https?://|www\.|\b[\w.-]+\.(?:com|org|net|np|gov|edu|info|io)\b", re.I)
EMAIL = re.compile(r"\S+@\S+\.\S+")
HTML = re.compile(r"<[^<>]{1,60}>|&[a-zA-Z]{2,8};|&#\d+;")
REPEAT = re.compile(r"([^\d\u0966-\u096f\s])\1{4,}")
BOILERPLATE = ["थप पढ्नुहोस्", "यो पनि पढ्नुहोस्", "प्रतिक्रिया दिनुहोस्", "सेयर गर्नुहोस्", "शेयर गर्नुहोस्",
               "कमेन्ट गर्नुहोस", "सम्बन्धित खबर", "सम्बन्धित समाचार", "फेसबुक", "ट्विटर", "facebook",
               "twitter", "youtube", "copyright", "सर्वाधिकार", "advertisement", "click here", "subscribe"]

HINDI = {"है", "हैं", "में", "और", "नहीं", "था", "थी", "गया", "गई", "किया", "रहा", "रही", "होता",
         "होती", "यह", "भी", "वह", "इस", "हुआ", "हुई", "कहा", "सकता", "सकते", "जाता", "जाती", "अपने",
         "उनके", "उनकी", "करना", "करें", "समझते", "लोग"}
NEPALI = {"छ", "छन्", "हो", "छैन", "गरेको", "भएको", "पनि", "लागि", "गर्न", "थियो", "थिए", "हुने",
          "गरी", "गरेका", "भने", "यो", "त्यो", "यस", "उनले", "गर्ने", "रहेको", "र"}

SPACE_BEFORE = re.compile(r" +([,;:?!)\]])")
SPACE_AFTER = re.compile(r"([(\[]) +")
SENT_END = re.compile("(?<=[\u0964\u0965?!.])\\s+")

_OPT = {}


def _init(opt):
    _OPT.update(opt)


def acquire_lock(output):
    import atexit
    lock = output + ".lock"
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            raise SystemExit(f"Another run (PID {pid}) is already writing {output}. "
                             f"Wait for it, or stop it with: kill {pid}")
        except (ValueError, ProcessLookupError):
            pass
        except PermissionError:
            raise SystemExit(f"{lock} belongs to a running process of another user.")
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))


def bounded(it, slots):
    for x in it:
        slots.acquire()
        yield x


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def hash64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


# --------------------------------------------------------------------------
# Normalise + repair (never rejects)
# --------------------------------------------------------------------------
def normalize(s, fixes):
    script = _OPT.get("script", "ne")
    s = unicodedata.normalize("NFC", s)
    s = s.translate(C1_QUOTES)
    s = s.replace(EYELASH_RA, "\ue000")
    s = s.translate(INVISIBLE).replace("\ue000", EYELASH_RA)
    s = CONTROL.sub(" ", s)
    s = SPACES.sub(" ", s)
    s = s.translate(PUNCT)
    s = " ".join(s.split())
    if not s:
        return s

    if script == "ne":
        for sym, rep, after_dev in PREETI_SYMBOLS:
            if sym in s:
                pat = f"(?<=[\u0900-\u097f]){re.escape(sym)}" if after_dev else re.escape(sym)
                s, n = re.subn(pat, rep, s)
                if n:
                    fixes[f"{sym} -> {rep}"] += n
        for bad, good in FIXES:
            if bad in s:
                fixes[f"{bad} -> {good}"] += s.count(bad)
                s = s.replace(bad, good)
        for name, rx, rep in REGEX_FIXES:
            s, n = rx.subn(rep, s)
            if n:
                fixes[name] += n

        s = s.replace("|", "\u0964")
        s = re.sub("\u0964(?: ?\u0964)+", "\u0965", s)
        s = re.sub("\u0965+", "\u0965", s)

    s = SPACE_BEFORE.sub(r"\1", s)
    s = SPACE_AFTER.sub(r"\1", s)
    pol = _OPT.get("danda_space", "tight")
    if script == "ne" and pol == "tight":
        s = re.sub(" +([\u0964\u0965])", r"\1", s)
    elif script == "ne" and pol == "spaced":
        s = re.sub("(?<=[^\\s\u0966-\u096f\\d])([\u0964\u0965])", r" \1", s)
    return s.strip()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def _shared_problem(s):
    """Problems that apply to any language. Returns a reason or None."""
    low = s.lower()
    if URL.search(s) or EMAIL.search(s):
        return "url_email"
    if HTML.search(s):
        return "html"
    if len(s) < 200 and any(b in low for b in BOILERPLATE):
        return "boilerplate"
    if REPEAT.search(s):
        return "char_run"
    return None


def _problem_ne(s, badchars, min_dev=8):
    bad = [ch for ch in set(s) if ch not in ALLOWED_NE]
    if ZWJ in s.replace(EYELASH_RA, ""):
        bad.append(ZWJ)
    if bad:
        badchars.update(bad)
        return "bad_char"

    toks = s.split()
    dev = len(DEV_LETTER.findall(s))
    dev_all = len(DEV_ANY.findall(s))
    lat = len(LAT_LETTER.findall(s))
    if sum(1 for t in toks if any(c in PREETI_CH for c in t) and LAT_LETTER.search(t)) >= 2:
        return "preeti"
    if dev < min_dev or dev_all / max(dev_all + lat, 1) < 0.7:
        return "not_devanagari"
    nonspace = len(s) - s.count(" ")
    if (dev_all + lat) / max(nonspace, 1) < 0.5:
        return "symbol_heavy"
    if any(TOKEN_MIXED.search(t) for t in toks):
        return "mixed_script_word"
    if MARK.search(SYL.sub("", s)):
        return "invalid_sequence"
    words = WORD.findall(s)
    hi = sum(w in HINDI for w in words)
    ne = sum(w in NEPALI for w in words)
    if hi >= 2 and hi > ne:
        return "hindi"
    return None


def _problem_en(s, badchars, min_dev=8):
    bad = [ch for ch in set(s) if ch not in ALLOWED_EN]
    if bad:
        badchars.update(bad)
        return "bad_char"

    lat = len(LAT_LETTER.findall(s))
    nonspace = len(s) - s.count(" ")
    if lat < min_dev:
        return "not_latin"
    if lat / max(nonspace, 1) < 0.5:
        return "symbol_heavy"
    # must contain at least one vowel, otherwise it's consonant soup or an acronym
    if not re.search(r"[aeiouAEIOU]", s):
        return "no_vowel"
    return None


def problem(s, badchars, min_dev=8):
    """Dispatch to the script-specific validator. Also runs the shared checks first."""
    shared = _shared_problem(s)
    if shared:
        return shared
    if _OPT.get("script", "ne") == "en":
        return _problem_en(s, badchars, min_dev)
    return _problem_ne(s, badchars, min_dev)


def chunks_of(s, target):
    if len(s) <= target:
        return [s]
    out, cur = [], ""
    for sent in SENT_END.split(s):
        if cur and len(cur) + 1 + len(sent) > target:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}" if cur else sent
    if cur:
        if out and len(cur) < _OPT.get("min_chars", 20):
            out[-1] += " " + cur
        else:
            out.append(cur)
    return out


def work(lines):
    kept, hashes, short = [], [], []
    reasons, words, chars, fixes, badchars = Counter(), Counter(), Counter(), Counter(), Counter()
    samples = defaultdict(list)
    n_chunks = 0
    min_chars, max_chars, target = _OPT["min_chars"], _OPT["max_chars"], _OPT["chunk"]

    def reject(why, text):
        reasons[why] += 1
        if len(samples[why]) < 4:
            samples[why].append(text[:160])

    def keep(c):
        kept.append(c)
        hashes.append(hash64(c))
        words.update(w for w in WORD.findall(c) if len(w) >= 3)
        chars.update(c)

    def flush(sents):
        text = " ".join(sents)
        if len(text) >= min_chars and not problem(text, Counter()):
            keep(text)
            fixes["sentences salvaged from a bad chunk"] += len(sents)

    for raw in lines:
        s = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if "\ufffd" in s:
            reject("bad_utf8", s)
            continue
        s = normalize(s, fixes)
        if not s:
            reasons["empty"] += 1
            continue
        if len(s) < min_chars:
            why = problem(s, badchars, min_dev=2)
            if why:
                reject("short:" + why, s)
            else:
                short.append(s)
            continue
        for c in chunks_of(s, target):
            n_chunks += 1
            if len(c) <= max_chars and not problem(c, Counter()):
                keep(c)
                continue
            good = []
            for sent in SENT_END.split(c):
                why = "sentence_too_long" if len(sent) > max_chars else problem(sent, badchars, min_dev=2)
                if why:
                    reject(why, sent)
                    flush(good)
                    good = []
                else:
                    good.append(sent)
            flush(good)
    return kept, hashes, short, n_chunks, len(lines), reasons, words, chars, fixes, badchars, samples


def read_chunks(path, n, limit):
    buf, k = [], 0
    with open(path, "rb") as f:
        for ln in f:
            buf.append(ln)
            k += 1
            if len(buf) >= n:
                yield buf
                buf = []
            if limit and k >= limit:
                break
    if buf:
        yield buf


# --------------------------------------------------------------------------
# Spelling (Nepali workflow only by default)
# --------------------------------------------------------------------------
def deletes1(w):
    return {w[:i] + w[i + 1:] for i in range(len(w))}


def find_suspects(freq, rare_max, common_min, dictionary):
    common = {w: c for w, c in freq.items() if c >= common_min}
    index = {}
    for w, c in common.items():
        for k in deletes1(w) | {w}:
            if k not in index or common[index[k]] < c:
                index[k] = w
    suspects = {}
    for w, c in freq.items():
        if c > rare_max or w in common or w in dictionary or len(w) < 4:
            continue
        cands = []
        if w in index:
            cands.append(index[w])
        for k in deletes1(w):
            if k in common:
                cands.append(k)
            if k in index:
                cands.append(index[k])
        cands = [x for x in set(cands) if x != w]
        if cands:
            suspects[w] = max(cands, key=common.get)
    return suspects


def load_dict(path):
    words = set()
    if path:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                w = ln.split("/")[0].strip()
                if w and not w.isdigit():
                    words.add(unicodedata.normalize("NFC", w))
    return words


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def font_coverage(font_dir, chars):
    from fontTools.ttLib import TTFont
    cmaps = []
    for root, _, files in os.walk(font_dir):
        for f in sorted(files):
            if f.lower().endswith((".ttf", ".otf")):
                try:
                    cmaps.append((f, set((TTFont(os.path.join(root, f), lazy=True).getBestCmap() or {}).keys())))
                except Exception:
                    pass
    return {ch: [f for f, cm in cmaps if ord(ch) not in cm] for ch in chars}, len(cmaps)


def write_report(path, st, reasons, samples, fixes, badchars, suspects, freq, chars, kept_samples, args):
    pc = lambda n: f"{100 * n / max(st['chunks'], 1):.1f}%"
    label = "Devanagari" if args.script == "ne" else "Latin"
    L = [f"# Corpus cleaning report ({args.script})", "",
         f"- Input lines: **{st['lines']:,}** → split into {st['chunks']:,} chunks of up to ~{args.chunk} chars",
         f"- Kept chunks: **{st['kept']:,}** ({pc(st['kept'])}) → `{args.output}`",
         f"- Short lines kept for headings/cells: {st['short']:,} → `{args.output}.short.txt`",
         f"- Duplicates removed: {st['dups']:,} chunks, {st['short_dups']:,} short lines",
         f"- Spelling ({args.spell}): {st['spell']:,} {'sentences dropped' if args.spell == 'drop' else 'chunks changed'}",
         f"- Time: {st['time'] / 60:.1f} min", "",
         "## Repairs made", "", "| Repair | Count |", "|---|---:|"]
    L += [f"| {k} | {n:,} |" for k, n in fixes.most_common()]
    L += ["", "## Rejected (chunks, or single sentences inside a chunk)", "",
          "| Reason | Count | Share of chunks | Examples |", "|---|---:|---:|---|"]
    for r, n in reasons.most_common():
        ex = " ⏐ ".join(x.replace("|", "\\|") for x in samples.get(r, [])[:3])
        L.append(f"| {r} | {n:,} | {pc(n)} | {ex} |")
    L += ["", "## Characters still causing rejections (top 30)", "", "| Char | Code | Name | Count |", "|---|---|---|---:|"]
    for ch, n in badchars.most_common(30):
        L.append(f"| {ch if ch.isprintable() else ''} | U+{ord(ch):04X} | {unicodedata.name(ch, '?')} | {n:,} |")
    if args.script == "ne":
        L += ["", f"## Suspected typos ({len(suspects):,} words, top 60)", "",
              f"Seen <= {args.rare_max}x, one edit from a word seen >= {args.common_min}x.", "",
              "| Rare word | Count | Likely intended | Count |", "|---|---:|---|---:|"]
        for w in sorted(suspects, key=lambda w: -freq[suspects[w]])[:60]:
            L.append(f"| {w} | {freq[w]} | {suspects[w]} | {freq[suspects[w]]:,} |")
    if args.script == "ne" and args.fonts:
        cov, nf = font_coverage(args.fonts, chars)
        if nf:
            missing = [(ch, m) for ch, m in cov.items() if m]
            L += ["", f"## Characters some of your {nf} fonts lack", ""]
            L += [f"- `{ch}` U+{ord(ch):04X} {unicodedata.name(ch, '?')}: {chars[ch]:,} uses; missing in "
                  f"{len(m)} fonts: {', '.join(m[:6])}{' …' if len(m) > 6 else ''}"
                  for ch, m in sorted(missing, key=lambda x: -chars[x[0]])] or ["- none"]
    L += ["", "## Random kept chunks", ""] + [f"- {s[:250]}" for s in kept_samples]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--script", choices=["ne", "en"], default="ne",
                    help="ne: Devanagari + ASCII; en: Latin + accented Latin")
    ap.add_argument("--report", help="Markdown report path (default <output>.report.md)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="only read the first N lines (trial run)")
    ap.add_argument("--chunk", type=int, default=700, help="target chunk length in characters")
    ap.add_argument("--min-chars", type=int, default=None,
                    help="shorter lines go to <output>.short.txt (default: 20 for --script ne, 15 for en)")
    ap.add_argument("--max-chars", type=int, default=1500, help="drop a single sentence longer than this")
    ap.add_argument("--danda-space", choices=["tight", "spaced", "keep"], default=None,
                    help="tight: छ।  spaced: छ ।  keep: as written (default: tight for ne, keep for en)")
    ap.add_argument("--spell", choices=["drop", "fix", "off"], default=None,
                    help="(default: drop for ne, off for en)")
    ap.add_argument("--rare-max", type=int, default=2)
    ap.add_argument("--common-min", type=int, default=500)
    ap.add_argument("--dict", help="word list, one word per line (Hunspell .dic works)")
    ap.add_argument("--fonts", help="font folder: report characters some fonts lack (Nepali only)")
    ap.add_argument("--max-vocab", type=int, default=4_000_000,
                    help="word-frequency table size limit; words seen once are forgotten when it is reached")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # script-dependent defaults
    if args.min_chars is None:
        args.min_chars = 15 if args.script == "en" else 20
    if args.danda_space is None:
        args.danda_space = "keep" if args.script == "en" else "tight"
    if args.spell is None:
        args.spell = "off" if args.script == "en" else "drop"

    report = args.report or args.output + ".report.md"
    tmp = f"{args.output}.pass1.{os.getpid()}.tmp"
    acquire_lock(args.output)
    t0 = time.time()
    opt = {"script": args.script, "danda_space": args.danda_space,
           "min_chars": args.min_chars, "max_chars": args.max_chars, "chunk": args.chunk}

    reasons, freq, chars, fixes, badchars = Counter(), Counter(), Counter(), Counter(), Counter()
    samples = defaultdict(list)
    hashes = array("Q")
    n_lines = n_chunks = n_short = short_dups = 0
    short_seen = set()
    pruned = 0
    print(f"Pass 1: {args.input} ({os.path.getsize(args.input) / 1e9:.2f} GB), "
          f"script={args.script}, {args.workers} workers", flush=True)
    slots = threading.BoundedSemaphore(args.workers * 3)
    with Pool(args.workers, initializer=_init, initargs=(opt,)) as pool, \
            open(tmp, "w", encoding="utf-8") as fo, open(args.output + ".short.txt", "w", encoding="utf-8") as fs:
        for k, (kept, hs, short, nc, nl, rs, ws, cs, fx, bc, sm) in enumerate(
                pool.imap(work, bounded(read_chunks(args.input, 5000, args.limit), slots)), 1):
            slots.release()
            n_lines += nl
            n_chunks += nc
            if kept:
                fo.write("\n".join(kept) + "\n")
            hashes.extend(hs)
            for s in short:
                h = hash64(s)
                if h in short_seen:
                    short_dups += 1
                else:
                    short_seen.add(h)
                    fs.write(s + "\n")
                    n_short += 1
            reasons.update(rs); freq.update(ws); chars.update(cs); fixes.update(fx); badchars.update(bc)
            if len(freq) > args.max_vocab:
                before = len(freq)
                freq = Counter({w: c for w, c in freq.items() if c > 1})
                pruned += before - len(freq)
            for r, xs in sm.items():
                samples[r].extend(xs[: max(0, 6 - len(samples[r]))])
            if k % 100 == 0:
                print(f"  {n_lines:,} lines, {n_chunks:,} chunks, kept {len(hashes):,}, "
                      f"{n_lines / (time.time() - t0):,.0f} lines/s, words {len(freq):,}, peak RAM {peak_gb():.1f} GB",
                      flush=True)
    print(f"Pass 1 done: {n_chunks:,} chunks, {len(hashes):,} kept before dedup, peak RAM {peak_gb():.1f} GB"
          + (f", {pruned:,} once-seen words forgotten" if pruned else ""), flush=True)

    h = np.frombuffer(hashes, dtype=np.uint64) if len(hashes) else np.zeros(0, np.uint64)
    keep = np.zeros(len(h), bool)
    if len(h):
        keep[np.unique(h, return_index=True)[1]] = True
    del h, hashes

    suspects = {}
    if args.spell != "off":
        suspects = find_suspects(freq, args.rare_max, args.common_min, load_dict(args.dict))
        print(f"Spelling: {len(suspects):,} suspected typo words", flush=True)

    rng = random.Random(args.seed)
    reservoir, dups, spell_hits, kept_n = [], 0, 0, 0
    out_chars = Counter()
    with open(tmp, encoding="utf-8") as fi, open(args.output, "w", encoding="utf-8") as fo:
        for i, s in enumerate(fi):
            if not keep[i]:
                dups += 1
                continue
            s = s.rstrip("\n")
            if suspects and any(w in suspects for w in WORD.findall(s)):
                if args.spell == "drop":
                    sents = SENT_END.split(s)
                    good = [x for x in sents if not any(w in suspects for w in WORD.findall(x))]
                    spell_hits += len(sents) - len(good)
                    s = " ".join(good)
                    if len(s) < args.min_chars:
                        continue
                else:
                    spell_hits += 1
                    s = WORD.sub(lambda m: suspects.get(m.group(0), m.group(0)), s)
            fo.write(s + "\n")
            kept_n += 1
            out_chars.update(s)
            if len(reservoir) < 40:
                reservoir.append(s)
            elif rng.random() < 40 / kept_n:
                reservoir[rng.randrange(40)] = s
    if os.path.exists(tmp):
        os.remove(tmp)

    st = {"lines": n_lines, "chunks": n_chunks, "kept": kept_n, "short": n_short, "short_dups": short_dups,
          "dups": dups, "spell": spell_hits, "time": time.time() - t0}
    write_report(report, st, reasons, samples, fixes, badchars, suspects, freq, out_chars, reservoir, args)
    print(f"Done: {kept_n:,} chunks -> {args.output}, {n_short:,} short lines -> {args.output}.short.txt\n"
          f"Report: {report}")


if __name__ == "__main__":
    main()