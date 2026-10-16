"""Bring an old v2 page record up to the current schema (section 1 of the spec).

``migrate(rec) -> (rec, warnings)`` works on the raw JSON dict, before the
Pydantic models see it, so files written by older versions of the tool still
load. It is idempotent: a migrated record comes back unchanged with no
warnings.

What it does
------------
1. Region types that are no longer in the enum are mapped (``LEGACY_TYPES``);
   ``list_item`` becomes ``form_field`` when it has fields or option words,
   else ``paragraph``. Every mapping is reported.
2. Lines without an ``id`` get one, in reading order; ``next_line_id`` is
   kept above every id in use.
3. Old checkbox values ``value: {text, options: [{text, checked, box_quad,
   text_quad}]}`` become ``field.options[]``; each option word becomes (or
   reuses) a line with role "option".
4. Line-level ``field_index`` / ``checked`` / ``box`` / ``tick_cell_id``
   move into ``field.options[]`` and are dropped from the line.
5. ``choice_grid`` (on a table region or a ``tables[]`` entry) becomes a
   checkbox field with ``options[].tick_cell_id``, then is dropped.
6. Checkbox fields get ``single_choice`` (true when the question contains
   "कुनै एक") and ``value: null``.
7. A label drawn over a label line gets that line's id (``label.line_id``).

Nothing here invents geometry: an option word whose box isn't known is left
for validation to report rather than given a placeholder box.
"""
from __future__ import annotations

import copy
import math
from typing import Any

# old type -> (new type, handwriting subtype or None)
LEGACY_TYPES: dict[str, tuple[str, str | None]] = {
    "text_line": ("paragraph", None),
    "mixed_text": ("paragraph", None),
    "options": ("form_field", None),
    "form_label": ("form_field", None),
    "form_value": ("form_field", None),
    "label": ("form_field", None),
    "page_number": ("footer", None),
    "signature_caption": ("caption", None),
    "table_footer": ("caption", None),
    "handwritten_annotation": ("handwriting", "annotation"),
    "handwritten_correction": ("handwriting", "correction"),
    "handwritten_number": ("handwriting", "number"),
}

CHECK_GLYPHS = ("☐", "☑", "✓", "✔")
SINGLE_CHOICE_HINT = "कुनै एक"
_LINE_LEGACY_KEYS = ("field_index", "checked", "box", "tick_cell_id")

Warning_ = dict[str, Any]


def _warn(out: list[Warning_], path: str, message: str, level: str = "warning") -> None:
    out.append({"level": level, "path": path, "message": message})


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _bbox_of_quad(q: Any) -> list[int] | None:
    try:
        xs = [float(p[0]) for p in q]
        ys = [float(p[1]) for p in q]
    except (TypeError, IndexError, ValueError):
        return None
    if len(xs) != 4:
        return None
    return [math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))]


def _is_box(b: Any) -> bool:
    return (
        isinstance(b, (list, tuple)) and len(b) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)
    )


def _iou(a: Any, b: Any) -> float:
    if not (_is_box(a) and _is_box(b)):
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _quad(b: list[int]) -> list[list[int]]:
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


# ---------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------

class _Lines:
    """Line list + id allocation for one record."""

    def __init__(self, rec: dict[str, Any]):
        self.rec = rec
        self.lines: list[dict[str, Any]] = [ln for ln in rec.get("lines") or [] if isinstance(ln, dict)]
        rec["lines"] = self.lines
        used = [ln["id"] for ln in self.lines if _is_int(ln.get("id"))]
        nxt = rec.get("next_line_id")
        self.next_id = max([*used, (nxt - 1) if _is_int(nxt) else -1]) + 1 if (used or _is_int(nxt)) else 0

    def by_id(self) -> dict[int, dict[str, Any]]:
        return {ln["id"]: ln for ln in self.lines if _is_int(ln.get("id"))}

    def take_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def of_block(self, block_id: Any) -> list[dict[str, Any]]:
        return sorted(
            (ln for ln in self.lines if ln.get("block_id") == block_id),
            key=lambda ln: ln.get("line_in_block") if _is_int(ln.get("line_in_block")) else 0,
        )

    def add(self, *, text: str, bbox: list[int], block_id: int, role: str,
            table_index: int | None = None, cell_id: int | None = None) -> dict[str, Any]:
        mine = self.of_block(block_id)
        lib = max((ln.get("line_in_block") or 0 for ln in mine), default=-1) + 1
        ln: dict[str, Any] = {
            "id": self.take_id(), "text": text, "bbox": list(bbox), "block_id": block_id,
            "line_in_block": lib, "is_heading": False,
            "font_size": round(0.8 * (bbox[3] - bbox[1]), 1), "font_id": None, "role": role,
        }
        if table_index is not None:
            ln["table_index"] = table_index
        if cell_id is not None:
            ln["cell_id"] = cell_id
        ln["quad"] = _quad(bbox)
        self.lines.append(ln)
        return ln


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def _map_region_types(rec: dict[str, Any], warns: list[Warning_]) -> None:
    lines = [ln for ln in rec.get("lines") or [] if isinstance(ln, dict)]
    mapped: list[str] = []
    for i, el in enumerate(rec.get("layout") or []):
        if not isinstance(el, dict):
            continue
        t = el.get("type")
        new: str | None = None
        subtype: str | None = None
        if t == "list_item":
            has_fields = bool(el.get("fields"))
            text = el.get("text") or ""
            has_option_words = any(g in text for g in CHECK_GLYPHS) or any(
                ln.get("block_id") == el.get("id") and ln.get("role") == "option" for ln in lines
            )
            new = "form_field" if (has_fields or has_option_words) else "paragraph"
        elif t in LEGACY_TYPES:
            new, subtype = LEGACY_TYPES[t]
        if new is None:
            continue
        el["type"] = new
        if new == "form_field":
            el.setdefault("fields", [])
            if not isinstance(el["fields"], list):
                el["fields"] = []
        else:
            el.pop("fields", None)  # only form_field / table / list carry fields
        if subtype is not None and not el.get("subtype"):
            el["subtype"] = subtype
        mapped.append(f"#{el.get('id')} {t} → {new}")
        _warn(warns, f"layout[{i}]", f"region #{el.get('id')}: type {t!r} is no longer a region type — loaded as {new!r}")
    if mapped:
        _warn(warns, "layout", f"{len(mapped)} region(s) mapped to current types: " + ", ".join(mapped))


def _reading_key(rec: dict[str, Any]):
    pos: dict[Any, tuple[float, int]] = {}
    for i, el in enumerate(rec.get("layout") or []):
        if isinstance(el, dict):
            ro = el.get("reading_order")
            pos[el.get("id")] = (ro if _is_int(ro) else math.inf, i)

    def key(item: tuple[int, dict[str, Any]]):
        idx, ln = item
        ro, i = pos.get(ln.get("block_id"), (math.inf, 1 << 30))
        lib = ln.get("line_in_block") if _is_int(ln.get("line_in_block")) else 0
        return (ro, i, lib, idx)
    return key


def _assign_line_ids(L: _Lines, warns: list[Warning_]) -> None:
    missing = [(i, ln) for i, ln in enumerate(L.lines) if not _is_int(ln.get("id"))]
    if not missing:
        return
    for _, ln in sorted(missing, key=_reading_key(L.rec)):
        ln["id"] = L.take_id()
    _warn(warns, "lines", f"{len(missing)} line(s) had no id — ids given in reading order", level="info")


def _regions_by_id(rec: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    return {el.get("id"): el for el in rec.get("layout") or [] if isinstance(el, dict)}


def _convert_checkbox_values(L: _Lines, warns: list[Warning_]) -> None:
    """value: {text, options: [{text, checked, box_quad, text_quad}]} -> options[]."""
    for i, el in enumerate(L.rec.get("layout") or []):
        if not isinstance(el, dict):
            continue
        for j, f in enumerate(el.get("fields") or []):
            v = f.get("value") if isinstance(f, dict) else None
            if not (isinstance(v, dict) and "options" in v):
                continue
            opts_out: list[dict[str, Any]] = list(f.get("options") or [])
            for k, o in enumerate(v.get("options") or []):
                if not isinstance(o, dict):
                    continue
                text = (o.get("text") or "").strip()
                tb = _bbox_of_quad(o.get("text_quad"))
                bb = _bbox_of_quad(o.get("box_quad"))
                # reuse a line that is already there for this word
                line = None
                for ln in L.of_block(el.get("id")):
                    if ln.get("role") in (None, "option") and (ln.get("text") or "").strip() == text \
                            and (tb is None or _iou(ln.get("bbox"), tb) >= 0.5):
                        line = ln
                        break
                if line is None:
                    if tb is None:
                        _warn(warns, f"layout[{i}].fields[{j}].value.options[{k}]",
                              f"region #{el.get('id')}: option {text!r} has no text box — not converted; draw it with the Option tool")
                        continue
                    line = L.add(text=text, bbox=tb, block_id=el.get("id"), role="option")
                else:
                    line["role"] = "option"
                opt: dict[str, Any] = {"line_id": line["id"],
                                       "checked": o.get("checked") if isinstance(o.get("checked"), bool) else None}
                if bb is not None:
                    opt["box"] = bb
                opts_out.append(opt)
            f["options"] = opts_out
            f["field_type"] = "checkbox"
            f["value"] = None
            _warn(warns, f"layout[{i}].fields[{j}]",
                  f"region #{el.get('id')}: old checkbox value converted to {len(opts_out)} option(s)", level="info")


def _convert_line_level_options(L: _Lines, warns: list[Warning_]) -> None:
    regions = _regions_by_id(L.rec)
    order = sorted(enumerate(L.lines), key=_reading_key(L.rec))
    for idx, ln in order:
        if not any(k in ln for k in _LINE_LEGACY_KEYS):
            continue
        fi = ln.pop("field_index", None)
        checked = ln.pop("checked", None)
        box = ln.pop("box", None)
        tick = ln.pop("tick_cell_id", None)
        if ln.get("role") != "option":
            continue  # a stray key on a non-option line: just dropped
        el = regions.get(ln.get("block_id"))
        fields = el.get("fields") if isinstance(el, dict) else None
        if not (_is_int(fi) and isinstance(fields, list) and 0 <= fi < len(fields)):
            _warn(warns, f"lines[{idx}]",
                  f"option line {ln.get('text')!r} (id {ln.get('id')}) names no existing field — attach it to a field")
            continue
        f = fields[fi]
        if not isinstance(f.get("options"), list):
            f["options"] = []
        opts = f["options"]
        if any(o.get("line_id") == ln["id"] for o in opts if isinstance(o, dict)):
            continue
        opt: dict[str, Any] = {"line_id": ln["id"], "checked": checked if isinstance(checked, bool) else None}
        if _is_box(box):
            opt["box"] = [int(round(v)) for v in box]
        if _is_int(tick):
            opt["tick_cell_id"] = tick
        opts.append(opt)
        f["field_type"] = "checkbox"
        f["value"] = None


def _option_line_for_cell(L: _Lines, ti: int, el: dict[str, Any], cell: dict[str, Any],
                          warns: list[Warning_], path: str) -> dict[str, Any] | None:
    cid = cell.get("id")
    for ln in L.lines:
        if ln.get("table_index") == ti and ln.get("cell_id") == cid and (ln.get("text") or "").strip():
            ln["role"] = "option"
            return ln
    text = (cell.get("text") or "").strip()
    if not text or not _is_box(cell.get("bbox")):
        _warn(warns, path, f"table {ti} cell {cid}: option cell has no text — option skipped")
        return None
    _warn(warns, path, f"table {ti} cell {cid}: option line {text!r} made from the cell box — check it",
          level="info")
    return L.add(text=text, bbox=[int(v) for v in cell["bbox"]], block_id=el.get("id"), role="option",
                 table_index=ti, cell_id=cid)


def _grid_pairs(cg: Any, cells: dict[Any, dict[str, Any]]) -> list[tuple[Any, Any, Any]] | None:
    """(option_cell_id, tick_cell_id, checked) from the shapes choice_grid had."""
    entries = cg.get("options") if isinstance(cg, dict) else cg
    if isinstance(entries, list):
        out = []
        for e in entries:
            if not isinstance(e, dict):
                return None
            oc = next((e[k] for k in ("cell_id", "option_cell_id", "option_cell") if k in e), None)
            tc = next((e[k] for k in ("tick_cell_id", "tick_cell") if k in e), None)
            out.append((oc, tc, e.get("checked")))
        return out
    if isinstance(cg, dict):
        grid = {(c.get("row"), c.get("col")): c.get("id") for c in cells.values()}
        for a, b, axis in (("option_row", "tick_row", "row"), ("option_col", "tick_col", "col")):
            if _is_int(cg.get(a)) and _is_int(cg.get(b)):
                out = []
                for c in sorted(cells.values(), key=lambda c: (c.get("row"), c.get("col"))):
                    if c.get(axis) != cg[a]:
                        continue
                    other = (cg[b], c.get("col")) if axis == "row" else (c.get("row"), cg[b])
                    out.append((c.get("id"), grid.get(other), None))
                return out
    return None


def _convert_choice_grids(L: _Lines, warns: list[Warning_]) -> None:
    tables = L.rec.get("tables") or []
    table_regions = {el.get("table_index"): el for el in L.rec.get("layout") or []
                     if isinstance(el, dict) and el.get("type") == "table"}
    sources: list[tuple[str, dict[str, Any], int | None]] = []
    for i, el in enumerate(L.rec.get("layout") or []):
        if isinstance(el, dict) and "choice_grid" in el:
            sources.append((f"layout[{i}]", el, el.get("table_index")))
    for ti, t in enumerate(tables):
        if isinstance(t, dict) and "choice_grid" in t:
            sources.append((f"tables[{ti}]", t, ti))
    for path, holder, ti in sources:
        cg = holder.pop("choice_grid")
        if cg in (None, [], {}):
            continue
        el = table_regions.get(ti)
        t = tables[ti] if _is_int(ti) and 0 <= ti < len(tables) and isinstance(tables[ti], dict) else None
        cells = {c.get("id"): c for c in (t or {}).get("cells") or [] if isinstance(c, dict)}
        pairs = _grid_pairs(cg, cells) if el is not None and t is not None else None
        if not pairs:
            _warn(warns, path, f"choice_grid could not be read and was dropped: {cg!r}", level="error")
            continue
        options: list[dict[str, Any]] = []
        for oc, tc, checked in pairs:
            cell = cells.get(oc)
            if cell is None:
                _warn(warns, path, f"choice_grid names cell {oc!r}, which table {ti} doesn't have")
                continue
            ln = _option_line_for_cell(L, ti, el, cell, warns, path)
            if ln is None:
                continue
            opt: dict[str, Any] = {"line_id": ln["id"], "checked": checked if isinstance(checked, bool) else None}
            if tc in cells:
                opt["tick_cell_id"] = tc
            elif tc is not None:
                _warn(warns, path, f"choice_grid tick cell {tc!r} doesn't exist in table {ti}")
            options.append(opt)
        label: dict[str, Any] | None = None
        rid = next((cg.get(k) for k in ("label_region_id", "region_id", "question_region_id")
                    if isinstance(cg, dict) and _is_int(cg.get(k))), None)
        if rid is not None:
            label = {"text": "", "region_id": rid}
        field: dict[str, Any] = {"label": label, "field_type": "checkbox", "value": None, "options": options}
        if isinstance(cg, dict) and isinstance(cg.get("single_choice"), bool):
            field["single_choice"] = cg["single_choice"]
        if not isinstance(el.get("fields"), list):
            el["fields"] = []
        el["fields"].append(field)
        _warn(warns, path, f"choice_grid of table {ti} converted to a checkbox field with {len(options)} option(s)",
              level="info")


def _finish_fields(L: _Lines) -> None:
    regions = _regions_by_id(L.rec)
    by_id = L.by_id()
    for el in L.rec.get("layout") or []:
        if not isinstance(el, dict):
            continue
        for f in el.get("fields") or []:
            if not isinstance(f, dict):
                continue
            lab = f.get("label")
            # label.line_id for a label drawn over a label line
            if isinstance(lab, dict) and not _is_int(lab.get("line_id")) and _is_box(lab.get("bbox")):
                best = max(
                    (ln for ln in L.of_block(el.get("id")) if ln.get("role") == "label"),
                    key=lambda ln: _iou(ln.get("bbox"), lab["bbox"]), default=None,
                )
                if best is not None and _iou(best.get("bbox"), lab["bbox"]) >= 0.5 and _is_int(best.get("id")):
                    lab["line_id"] = best["id"]
            if f.get("field_type") == "checkbox":
                f["value"] = None
                if not isinstance(f.get("single_choice"), bool):
                    q = ""
                    if isinstance(lab, dict):
                        q = lab.get("text") or ""
                        if _is_int(lab.get("line_id")) and lab["line_id"] in by_id:
                            q += " " + (by_id[lab["line_id"]].get("text") or "")
                        if lab.get("region_id") in regions:
                            q += " " + (regions[lab["region_id"]].get("text") or "")
                    f["single_choice"] = SINGLE_CHOICE_HINT in q
                if f.get("options") is None:
                    f["options"] = []


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def migrate(rec: dict[str, Any]) -> tuple[dict[str, Any], list[Warning_]]:
    """Return an upgraded deep copy of ``rec`` and what was changed."""
    rec = copy.deepcopy(rec)
    warns: list[Warning_] = []
    if not isinstance(rec, dict):
        return rec, warns

    _map_region_types(rec, warns)
    L = _Lines(rec)
    _assign_line_ids(L, warns)
    _convert_checkbox_values(L, warns)
    _convert_line_level_options(L, warns)
    _convert_choice_grids(L, warns)
    _finish_fields(L)

    if L.lines or _is_int(rec.get("next_line_id")):
        rec["next_line_id"] = max(L.next_id, rec.get("next_line_id") or 0)
    return rec, warns
