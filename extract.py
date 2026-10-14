#!/usr/bin/env python3
"""One-pass extractor: keep lines with no Devanagari and at least one Latin letter."""
import re, sys

DEV = re.compile(r"[\u0900-\u097F]")
LAT = re.compile(r"[A-Za-z]")

src_path, dst_path = sys.argv[1], sys.argv[2]
read = kept = skipped_dev = skipped_other = 0

with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
    for raw in src:
        read += 1
        line = raw.decode("utf-8", "ignore").strip()
        if not line:
            continue
        if DEV.search(line):
            skipped_dev += 1
            continue
        if not LAT.search(line):
            skipped_other += 1
            continue
        dst.write((line + "\n").encode("utf-8"))
        kept += 1
        if read % 2_000_000 == 0:
            print(f"  read {read:,}  kept {kept:,}", file=sys.stderr, flush=True)

print(f"Done. read {read:,}  kept {kept:,}  "
      f"(dropped: {skipped_dev:,} with Devanagari, {skipped_other:,} no Latin)",
      file=sys.stderr)