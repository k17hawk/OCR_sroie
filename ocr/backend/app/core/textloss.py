"""Text-loss guard (forms prompt section 7).

Saving must never drop printed text. ``find_text_loss(before, after)``
compares the printed characters of every region in the page the editor
sent (``before``) with the page that would be written (``after``) and
reports, per region, the characters that disappeared.

What counts as printed: every character of a region's text, its lines,
its table cells and its list items, EXCEPT fill-in marks and layout noise:
dot / underscore leaders, tabs (also a typed "\\t" or "\\n"), ☐ ☑ ☒ ✓ ✔, and
whitespace. Markup that the old v2 format derived (table HTML tags, the
"## " heading prefix, the "- " list bullet, "<figure/>") is not printed
text either.

A character is lost only when NO representation of the region keeps it
(text, lines, cells and items are pooled per region), so moving text from
the region text into its lines is not a loss.
"""
from __future__ import annotations

import difflib
import html
import re
from collections import Counter
from typing import Any

# leaders: any ellipsis run, 3+ dots (spaced or not), 2+ underscores, long dash runs
_LEADER = re.compile(r"…+|\.(?:\s*[.…]){2,}|_{2,}|-{3,}|–{2,}|—{2,}")
_MARKS = re.compile(r"[☐☑☒✓✔]|\\[tn]")  # a typed "\t" / "\n" is not printed text
_SPACE = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
_HEADING_MARK = re.compile(r"^\s*#+\s*")
_LIST_DASH = re.compile(r"^\s*-\s+", re.MULTILINE)


def printed_chars(text: str | None, *, kind: str = "") -> Counter[str]:
    """Multiset of the printed characters in ``text``.

    ``kind`` names where the text comes from ("heading", "list", "table",
    "figure") so that v2-derived markup can be ignored.
    """
    if not text:
        return Counter()
    t = text
    if kind == "table" or "<table" in t:
        t = html.unescape(_TAG.sub(" ", t.replace("<br>", "\n")))
    if kind == "heading":
        t = _HEADING_MARK.sub("", t)
    if kind in ("list", "sublist"):
        t = _LIST_DASH.sub("", t)
    if kind == "figure":
        t = t.replace("<figure/>", "")
    t = _LEADER.sub(" ", t)
    t = _MARKS.sub(" ", t)
    t = _SPACE.sub("", t)
    return Counter(t)


def _missing_passages(old: str, new: str) -> str:
    """The parts of ``old`` that are not in ``new`` (for the message only;
    the decision is made on printed-character counts), joined by " … "."""
    a, b = old.replace("\n", " "), new.replace("\n", " ")
    parts = []
    for op, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            seg = a[i1:i2].strip()
            if printed_chars(seg):
                parts.append(seg)
    return " … ".join(parts)


def _distinct_rows(text: Any) -> str:
    """The region text with each row counted once. A region's stored text is
    built from its lines; when two lines held the same words (old pages: a
    cell's OCR line plus an automatic whole-cell line) the text has them twice.
    Merging those lines is not a loss - the lines and cells still carry the
    real count of every repeated word."""
    if not isinstance(text, str):
        return ""
    seen, out = set(), []
    for row in text.split("\n"):
        key = row.strip()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(row)
    return "\n".join(out)


def _union(*counters: Counter[str]) -> Counter[str]:
    out: Counter[str] = Counter()
    for c in counters:
        out |= c
    return out


def _region_chars(doc: dict[str, Any], el: dict[str, Any]) -> Counter[str]:
    kind = el.get("type", "")
    rid = el.get("id")
    lines = [ln for ln in doc.get("lines") or []
             if isinstance(ln, dict) and ln.get("block_id") == rid]
    parts = [
        printed_chars(_distinct_rows(el.get("text")), kind=kind),
        printed_chars("\n".join(str(ln.get("text") or "") for ln in lines)),
    ]
    if kind == "table":
        ti = el.get("table_index")
        tables = doc.get("tables") or []
        if isinstance(ti, int) and 0 <= ti < len(tables) and isinstance(tables[ti], dict):
            cells = tables[ti].get("cells") or []
            parts.append(printed_chars("\n".join(str(c.get("text") or "") for c in cells
                                                 if isinstance(c, dict))))
    if kind in ("list", "sublist"):
        items = el.get("items") or []
        parts.append(printed_chars("\n".join(str(i.get("text") or "") for i in items
                                             if isinstance(i, dict))))
    if kind == "stamp":
        parts.append(printed_chars(el.get("stamp_text")))
    return _union(*parts)


def find_text_loss(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Regions whose printed characters would be lost, as readable issues:
    ``{"where", "region_id", "type", "lost", "before", "after"}``."""
    after_by_id = {el.get("id"): el for el in after.get("layout") or [] if isinstance(el, dict)}
    issues: list[dict[str, Any]] = []
    for i, el in enumerate(before.get("layout") or []):
        if not isinstance(el, dict):
            continue
        had = _region_chars(before, el)
        if not had:
            continue
        new_el = after_by_id.get(el.get("id"))
        has = _region_chars(after, new_el) if new_el is not None else Counter()
        lost = had - has
        if not lost:
            continue
        old_text = str(el.get("text") or "")
        new_text = str((new_el or {}).get("text") or "")
        shown = _missing_passages(old_text, new_text) or "".join(sorted(lost.elements()))
        issues.append({
            "where": f"layout[{i}] (#{el.get('id')} {el.get('type')})",
            "region_id": el.get("id"),
            "type": el.get("type"),
            "lost": shown,
            "before": old_text,
            "after": str((new_el or {}).get("text") or ""),
        })
    return issues
