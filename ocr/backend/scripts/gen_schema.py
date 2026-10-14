"""Generate ``schema/doc_v2.schema.json`` from the Pydantic models.

Run with::

    python -m scripts.gen_schema

A CI check compares the checked-in file with this generator's output, so
please re-run after any schema change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.schemas.doc_v2 import Doc


def main() -> int:
    schema = Doc.model_json_schema()
    repo_root = Path(__file__).resolve().parents[2]
    out = repo_root / "schema" / "doc_v2.schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
