"""Collect every annotated page under manual_annotation/ into labels.jsonl
for training (OCR_sroie data.py / train.py).

    python -m scripts.build_labels                       # all pages
    python -m scripts.build_labels --status submitted reviewed
    python -m scripts.build_labels --out /some/folder/labels.jsonl

Each page is run through finalize() (the same step the exporter uses) and
its "image" is rewritten relative to the labels.jsonl folder, so
    python data.py --data <that folder>
finds every image. Pages that fail to parse are listed and skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from app.config import get_settings
from app.core.finalize import finalize
from app.storage import doc_store


def main() -> int:
    s = get_settings()
    root = Path(s.storage_root).resolve() / s.annotation_dir.strip("/")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(root / "labels.jsonl"))
    ap.add_argument("--status", nargs="*", help="only these annotation.status values")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    def page_no(p: Path) -> int:
        tail = p.stem.rsplit("_page", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    files = sorted(root.glob("*/*_page*.jsonl"), key=lambda p: (p.parent.name, page_no(p)))
    n_ok = n_skip = 0
    with open(out.with_suffix(".tmp"), "w", encoding="utf-8") as g:
        for f in files:
            try:
                doc = finalize(doc_store.read_doc_file(f))
            except ValueError as e:
                print(f"[skip] {f}: {e}", file=sys.stderr)
                n_skip += 1
                continue
            if a.status and doc.annotation.status not in a.status:
                continue
            image = Path(s.storage_root).resolve() / doc.image
            if not image.is_file():
                print(f"[skip] {f}: image {image} missing", file=sys.stderr)
                n_skip += 1
                continue
            doc = doc.model_copy(update={"image": os.path.relpath(image, out.parent)})
            g.write(doc.model_dump_json() + "\n")
            n_ok += 1
    os.replace(out.with_suffix(".tmp"), out)
    print(f"{n_ok} page(s) -> {out}  ({n_skip} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
