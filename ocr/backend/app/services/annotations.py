"""PUT / PATCH handling with optimistic locking and project-level validation.

Why saves were failing with 422
-------------------------------
The editor sends its in-memory page as-is. Several things in that JSON
could make the save fail before it was stored:

  * image / width / height / document_id differing even slightly from the
    stored row (e.g. "/images/x.png" vs "x.png", 1000.0 vs 1000)
  * doc_v2 forbids extra keys on every model, and the editor leaves old keys
    behind when you change a box's Type (`level` after heading, `table_index`
    after table, `text` on a new signature/stamp, ...) -> every such save failed
  * float bbox values, missing required fields
  * the 422 detail was never logged, so none of this was visible

This version:
  1. takes document_id / image / width / height from the server row (the
     client can't change them anyway), so they never cause a 422;
  2. trims every object to exactly the fields its doc_v2 model declares
     (read from the schema, so new types are picked up automatically) and
     fills required fields the editor didn't set;
  3. on a real schema error, logs and returns a short readable list
     ("layout[3] (#12 title).type: Input should be ...") instead of a raw dump.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from typing import get_args

import jsonpatch
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from app.core.finalize import finalize
from app.schemas.doc_v2 import (
    Annotation, Block, Doc, ExtraRegion, FieldOption,
    FormFieldEntry, FormFieldLabel, LayoutElement, Line, ListItem,
    NormalFieldValue, Quality, RawLengths, RulingLine, Table, TableCell,
)
from app.core.migrate import migrate
from app.services.geometry import ensure_quads
from app.services.projects import Project
from app.storage import doc_store

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# cleaning the editor payload, driven by the Doc schema itself
#
# Every model in doc_v2 has extra="forbid", so ONE stray key anywhere fails
# the whole save. The editor leaves such keys behind all the time, e.g.
#   heading -> paragraph  keeps `level`       table  -> paragraph keeps `table_index`
#   figure  -> paragraph  keeps `subtype`     list   -> paragraph keeps `items`
#   new signature / stamp gets `text: ""`     (not allowed on those types)
# So every object is trimmed to exactly the fields its model declares, and
# required fields the editor didn't set are filled with safe defaults.
# ---------------------------------------------------------------------------

def _union_members(annotated: Any) -> list[type]:
    """Classes inside Annotated[Union[...], Field(discriminator=...)]."""
    inner = get_args(annotated)[0] if get_args(annotated) else annotated
    return [m for m in get_args(inner) if isinstance(m, type)]


# "title" -> TitleLayout, "checkbox" -> CheckboxExtraRegion, ...  (read from doc_v2)
LAYOUT_MODELS: dict[str, type[BaseModel]] = {
    m.model_fields["type"].default: m for m in _union_members(LayoutElement)
}
EXTRA_MODELS: dict[str, type[BaseModel]] = {
    m.model_fields["type"].default: m for m in _union_members(ExtraRegion)
}

# safe values for required fields the editor may not have set
LAYOUT_DEFAULTS: dict[str, Any] = {
    "text": "", "level": 2, "subtype": "photo", "fields": [], "table_index": -1,
    "list_style": "bullet", "items": [], "signer": "other", "stamp_text": "",
    "reading_order": None,
}
LOCKED_TYPES = {"logo", "signature", "stamp"}


def _keep(d: Any, model: type[BaseModel]) -> Any:
    """Only the keys `model` declares (drops editor leftovers)."""
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if k in model.model_fields}


def _int_bbox(b: Any) -> Any:
    """[x0, y0, x1, y1] as ints (floor the start, ceil the end), like normalizeDoc.ts."""
    if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
        return [math.floor(b[0]), math.floor(b[1]), math.ceil(b[2]), math.ceil(b[3])]
    return b


def _fix_bboxes(obj: Any) -> None:
    """Make every ``bbox`` integer-valued, at any depth (in place)."""
    if isinstance(obj, dict):
        if "bbox" in obj:
            obj["bbox"] = _int_bbox(obj["bbox"])
        for v in obj.values():
            _fix_bboxes(v)
    elif isinstance(obj, list):
        for v in obj:
            _fix_bboxes(v)


def _to_int(v: Any, default: int | None) -> int | None:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _clean_form_field(f: Any) -> Any:
    f = _keep(f, FormFieldEntry)
    if not isinstance(f, dict):
        return f
    lab = f.get("label")
    f["label"] = _keep(lab, FormFieldLabel) if isinstance(lab, dict) else None
    f.setdefault("field_type", "text")
    v = f.get("value")
    if isinstance(v, dict):
        v = _keep(v, NormalFieldValue)
        v.setdefault("text", "")
        v["handwritten"] = bool(v.get("handwritten", False))
        f["value"] = v
    if isinstance(f.get("options"), list):
        f["options"] = [_keep(o, FieldOption) for o in f["options"] if isinstance(o, dict)]
    return f


def _clean_layout_element(el: dict[str, Any]) -> dict[str, Any]:
    t = el.get("type")
    model = LAYOUT_MODELS.get(t)
    if model is None:
        return el  # unknown type: leave it, validation will name it
    out = _keep(el, model)
    for name, field in model.model_fields.items():
        if name not in out and field.is_required() and name in LAYOUT_DEFAULTS:
            out[name] = LAYOUT_DEFAULTS[name]
        elif name in out and out[name] is None and name in LAYOUT_DEFAULTS and field.is_required():
            out[name] = LAYOUT_DEFAULTS[name]
    if t in LOCKED_TYPES:
        out["reading_order"] = None
    else:
        out["reading_order"] = _to_int(out.get("reading_order"), None)
    if t == "heading":
        out["level"] = min(6, max(1, _to_int(out.get("level"), 2) or 2))
    if t == "list":
        out["items"] = [
            {**_keep(i, ListItem), "level": _to_int((i or {}).get("level"), 0) or 0}
            for i in out.get("items") or [] if isinstance(i, dict)
        ]
        for i in out["items"]:
            i.setdefault("marker", ""); i.setdefault("text", "")
    if t == "form_field" or (t in ("table", "list") and isinstance(out.get("fields"), list)):
        out["fields"] = [_clean_form_field(f) for f in out.get("fields") or []]
    return out


def _clean_line(ln: Any) -> Any:
    if not isinstance(ln, dict) or not isinstance(ln.get("block_id"), int):
        return None  # a line not attached to any region can't be stored
    out = _keep(ln, Line)
    fs = out.get("font_size")
    if not isinstance(fs, (int, float)) or fs != fs:  # missing or NaN
        h = out["bbox"][3] - out["bbox"][1] if isinstance(out.get("bbox"), list) else 12
        out["font_size"] = float(max(8, h * 0.7))
    out.setdefault("text", "")
    out.setdefault("line_in_block", 0)
    out["is_heading"] = bool(out.get("is_heading", False))
    if out.get("role") not in (None, "label", "value", "option"):
        out.pop("role", None)
    return out


def _clean_table(t: Any) -> Any:
    out = _keep(t, Table)
    if not isinstance(out, dict):
        return out
    cells = []
    for c in out.get("cells") or []:
        c = _keep(c, TableCell)
        if not isinstance(c, dict):
            continue
        c.setdefault("rowspan", 1); c.setdefault("colspan", 1)
        c["text"] = c.get("text") or ""
        c["is_header"] = bool(c.get("is_header", False))
        c["handwritten"] = bool(c.get("handwritten", False))
        c["empty"] = c["text"].strip() == ""
        cells.append(c)
    out["cells"] = cells
    out.setdefault("num_rows", max([c["row"] + c["rowspan"] for c in cells if "row" in c] or [0]))
    out.setdefault("num_cols", max([c["col"] + c["colspan"] for c in cells if "col" in c] or [0]))
    out.setdefault("header_rows", 0)
    if out.get("ruling_style") not in ("full", "horizontal", "none", "partial"):
        out["ruling_style"] = "full"
    out.setdefault("html", "")
    if not isinstance(out.get("reading_order"), list):
        out["reading_order"] = [c["id"] for c in cells if "id" in c]
    return out


# ---------------------------------------------------------------------------
# lists drawn as a box with text_line boxes inside
#
# The List tool creates an empty list (items: [], text: ""); the text_line
# boxes drawn inside it never became its items, so an "empty" list was
# stored - and dropped as empty. Fill it from the boxes it contains.
# ---------------------------------------------------------------------------

# (text_line / list_item boxes are loaded as paragraphs now; see core/migrate.py)
LIST_CHILD_TYPES = {"paragraph"}
_MARKER_RE = re.compile(
    r"^\s*((?:[०-९]+|[0-9]+)[.)]|\(?[क-ह]\)|\([०-९0-9]+\)|[a-zA-Z][.)]"
    r"|(?:[ivxlcdm]+|[IVXLCDM]+)[.)]|[-•*●▪◦–])\s+(.*)$",
    re.S,
)


def _inside_frac(inner: list[int], outer: list[int]) -> float:
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return (ix * iy) / a if a else 0.0


def _split_marker(text: str) -> tuple[str, str]:
    m = _MARKER_RE.match(text or "")
    if not m:
        return "", (text or "").strip()
    marker = "•" if m.group(1) in ("-", "*", "●") else m.group(1)
    return marker, m.group(2).strip()


def _list_style(items: list[dict[str, Any]]) -> str:
    m = next((i["marker"] for i in items if i["marker"]), "")
    if re.match(r"^\(?[०-९क-ह]", m):
        return "dev"
    if re.match(r"^[ivxIVX]+[.)]$", m):
        return "roman"
    if re.match(r"^[0-9a-zA-Z]", m):
        return "latin"
    return "bullet"


def _fill_lists_from_children(layout: list[dict[str, Any]]) -> None:
    for el in layout:
        if el.get("type") != "list" or el.get("items"):
            continue
        if (el.get("text") or "").strip():
            continue  # typed text: finalize() turns it into items
        box = el.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        kids = [
            c for c in layout
            if c is not el and c.get("type") in LIST_CHILD_TYPES
            and isinstance(c.get("bbox"), list) and _inside_frac(c["bbox"], box) >= 0.8
            and (c.get("text") or "").strip()
        ]
        if kids:
            kids.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
            left = min(c["bbox"][0] for c in kids)
            step = max(12, min(c["bbox"][3] - c["bbox"][1] for c in kids))
            items = []
            for c in kids:
                marker, text = _split_marker(c["text"])
                items.append({"marker": marker, "level": int((c["bbox"][0] - left) // (step * 1.5)), "text": text})
        else:
            continue
        el["items"] = items
        if el.get("list_style") in (None, "", "bullet"):
            el["list_style"] = _list_style(items)
        el["text"] = "\n".join(
            "  " * i["level"] + (("-" if i["marker"] == "•" else i["marker"]) + " " if i["marker"] else "") + i["text"]
            for i in items
        )
        log.info("list #%s: %d item(s) filled from the box(es) inside it", el.get("id"), len(items))


def _clean_payload(payload: dict[str, Any], row: doc_store.DocumentRow) -> dict[str, Any]:
    """Return a copy of the editor payload that the Doc schema accepts."""
    # an editor that loaded an old page may still send old shapes
    p, _ = migrate(json.loads(json.dumps(payload)))
    server = row.doc.model_dump(mode="json")

    # Fields the client may not change: always the server's values.
    p["document_id"] = row.document_id
    p["image"] = row.image_key
    p["width"] = row.doc.width
    p["height"] = row.doc.height

    # Required page-level fields the editor may have dropped: keep the server's.
    for k in ("page_number", "page_count", "language", "digit_style", "doc_type",
              "text", "quality", "annotation", "next_line_id"):
        if p.get(k) is None:
            p[k] = server.get(k)

    p = _keep(p, Doc)
    p["layout"] = [_clean_layout_element(e) for e in p.get("layout") or [] if isinstance(e, dict)]
    _fill_lists_from_children(p["layout"])
    p["lines"] = [x for x in (_clean_line(ln) for ln in p.get("lines") or []) if x is not None]
    p["tables"] = [_clean_table(t) for t in p.get("tables") or []]
    p["blocks"] = [_keep(b, Block) for b in p.get("blocks") or []]
    p["ruling_lines"] = [_keep(r, RulingLine) for r in p.get("ruling_lines") or []]
    p["extra_regions"] = [
        _keep(er, EXTRA_MODELS[er.get("type")]) if isinstance(er, dict) and er.get("type") in EXTRA_MODELS else er
        for er in p.get("extra_regions") or []
    ]
    for er in p["extra_regions"]:
        if isinstance(er, dict) and er.get("type") == "fingerprint" and er.get("hand") not in ("left", "right"):
            er["hand"] = "right"
    if isinstance(p.get("quality"), dict):
        q = _keep(p["quality"], Quality)
        if isinstance(q.get("raw_lengths"), dict):
            q["raw_lengths"] = _keep(q["raw_lengths"], RawLengths)
        p["quality"] = q
    if isinstance(p.get("annotation"), dict):
        p["annotation"] = _keep(p["annotation"], Annotation)

    _fix_bboxes(p)
    # After the bboxes are integers: rebuild missing / stale quads from them.
    ensure_quads(p)
    return p


# ---------------------------------------------------------------------------
# readable validation errors
# ---------------------------------------------------------------------------

def _describe_loc(loc: tuple[Any, ...], payload: dict[str, Any]) -> str:
    """("layout", 3, "type") -> 'layout[3] (#12 title).type'."""
    out = ""
    node: Any = payload
    for i, part in enumerate(loc):
        if isinstance(part, int):
            out += f"[{part}]"
            node = node[part] if isinstance(node, list) and part < len(node) else None
            if i == 1 and loc[0] == "layout" and isinstance(node, dict):
                out += f" (#{node.get('id')} {node.get('type')})"
        else:
            out += ("." if out else "") + str(part)
            node = node.get(part) if isinstance(node, dict) else None
    return out or "(document)"


def _allowed_values_hint() -> str:
    """Layout types the Doc schema accepts (read from doc_v2)."""
    return ", ".join(sorted(LAYOUT_MODELS))


def _readable_errors(e: ValidationError, payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for err in e.errors()[:20]:
        loc = tuple(err.get("loc", ()))
        item: dict[str, Any] = {
            "loc": list(loc),
            "where": _describe_loc(loc, payload),
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        inp = err.get("input")
        if isinstance(inp, (str, int, float, bool)) or inp is None:
            item["got"] = inp
        errors.append(item)
    return errors


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _validate_against_project(doc: Doc, project: Project) -> None:
    """Minimal validation: doc_type is recorded but never blocks.

    Layout types and field types are not checked at all — the schema
    accepts whatever the client sends. Submission gating lives in
    status.py.
    """
    if doc.doc_type not in project.doc_types:
        log.warning(
            "doc_type %r not in project %s allowed types %r",
            doc.doc_type, project.id, project.doc_types,
        )


def _assert_immutable(row: doc_store.DocumentRow, doc: Doc) -> None:
    # _clean_payload already copies these from the row; this is a safety net.
    if doc.document_id != row.document_id:
        raise HTTPException(422, "document_id is immutable")
    if doc.image != row.image_key:
        raise HTTPException(422, "image is immutable")
    if doc.width != row.doc.width or doc.height != row.doc.height:
        raise HTTPException(422, "image dimensions are immutable")


# ---------------------------------------------------------------------------
# PUT / PATCH
# ---------------------------------------------------------------------------

def replace(
    *,
    row: doc_store.DocumentRow,
    project: Project,
    payload: dict[str, Any],
    revision: int,
    user_id: int | None,
) -> doc_store.DocumentRow:
    if revision != row.revision:
        raise HTTPException(
            409, f"revision mismatch: client={revision} server={row.revision}"
        )

    cleaned = _clean_payload(payload, row)

    try:
        doc = Doc.model_validate(cleaned)
    except ValidationError as e:
        errors = _readable_errors(e, cleaned)
        log.warning(
            "save rejected for %s (%d error(s)):\n  %s",
            row.document_id, len(e.errors()),
            "\n  ".join(f"{x['where']}: {x['msg']} (got {x.get('got')!r})" for x in errors),
        )
        if any(x["where"].endswith(".type") or x["type"] in ("union_tag_invalid", "literal_error")
               for x in errors):
            log.warning("layout/region types accepted by the Doc schema: %s", _allowed_values_hint())
        raise HTTPException(422, detail=errors) from e

    _assert_immutable(row, doc)
    _validate_against_project(doc, project)

    try:
        finalized = finalize(doc)
    except Exception as e:  # noqa: BLE001 — surface it instead of a bare 500
        log.exception("finalize() failed for %s", row.document_id)
        raise HTTPException(422, detail=[{
            "loc": [], "where": "(finalize)", "msg": f"{type(e).__name__}: {e}", "type": "finalize_error",
        }]) from e

    row.doc = finalized
    row.status = finalized.annotation.status
    row.revision += 1

    doc_store.save(row)
    doc_store.append_audit({
        "user_id": user_id,
        "document_id": row.document_id,
        "action": "save",
        "diff_size": len(json.dumps(payload)),
    })
    log.info("saved %s -> revision %d", row.document_id, row.revision)
    return row


def patch(
    *,
    row: doc_store.DocumentRow,
    project: Project,
    ops: list[dict[str, Any]],
    revision: int,
    user_id: int | None,
) -> doc_store.DocumentRow:
    if revision != row.revision:
        raise HTTPException(
            409, f"revision mismatch: client={revision} server={row.revision}"
        )
    try:
        new_payload = jsonpatch.JsonPatch(ops).apply(row.doc.model_dump(mode="json"))
    except jsonpatch.JsonPatchException as e:
        raise HTTPException(400, f"invalid JSON patch: {e}") from e
    return replace(
        row=row, project=project, payload=new_payload,
        revision=row.revision, user_id=user_id,
    )