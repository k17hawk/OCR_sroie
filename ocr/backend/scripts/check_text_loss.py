"""Dry-run a save on page files: show what the text of each region becomes
and whether the save would be blocked for losing printed text.

    python -m scripts.check_text_loss PATH [PATH ...] [--grep TEXT]

PATH is a page .jsonl / .json (or a folder: every *.jsonl under it). Nothing
is written. Runs the same steps as a save: migrate -> validate -> finalize
-> text-loss check (app/core/textloss.py), comparing against the page as the
editor receives it.

Exit code 1 if any page would be blocked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.finalize import finalize
from app.core.textloss import find_text_loss
from app.storage.doc_store import doc_from_record


def _record(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(next(ln for ln in text.splitlines() if ln.strip()))


def check(path: Path, grep: str | None) -> bool:
    raw = _record(path)
    loaded, warns = doc_from_record(raw)
    before = loaded.model_dump(mode="json")          # what the editor gets and sends back
    after = finalize(loaded).model_dump(mode="json")  # what a save would write
    lost = find_text_loss(before, after) + [
        dict(x, where=x["where"] + " [vs file on disk]") for x in find_text_loss(raw, after)
        if x["region_id"] not in {y["region_id"] for y in find_text_loss(before, after)}
    ]

    print(f"== {path}  ({len(warns)} migration note(s))")
    by_id = {el.get("id"): el for el in after["layout"]}
    raw_by_id = {el.get("id"): el for el in raw.get("layout") or []}
    for rid, el in by_id.items():
        old = (raw_by_id.get(rid) or {}).get("text")
        new = el.get("text")
        if grep and grep not in (old or "") + (new or ""):
            continue
        if old != new or grep:
            print(f"  #{rid} {el['type']}")
            print(f"     file : {old!r}")
            print(f"     save : {new!r}")
    for x in lost:
        print(f"  BLOCKED {x['where']}: would lose {x['lost']!r}")
    if not lost:
        print("  OK: a save keeps all printed text")
    return not lost


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--grep", help="only show regions whose text contains this")
    args = ap.parse_args(argv)
    files: list[Path] = []
    for p in args.paths:
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    ok = all([check(f, args.grep) for f in files])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
