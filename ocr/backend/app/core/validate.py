from __future__ import annotations
import unicodedata

from app.schemas.doc_v2 import Doc, FormFieldLayout, TableLayout
from app.schemas.api import ValidateIssue


def validate(doc: Doc) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []

    # Line -> block coverage
    block_ids = {el.id for el in doc.layout}
    for i, ln in enumerate(doc.lines):
        if ln.block_id not in block_ids:
            issues.append(ValidateIssue(level="warning", path=f"lines[{i}]",
                                        message=f"block_id {ln.block_id} does not exist"))

    # Reading-order duplicates
    seen: dict[int, int] = {}
    for i, el in enumerate(doc.layout):
        ro = el.reading_order
        if ro is None:
            continue
        if ro in seen:
            issues.append(ValidateIssue(level="warning", path=f"layout[{i}]",
                                        message=f"duplicate reading_order {ro}"))
        seen[ro] = i

    # Table tiling
    for ti, t in enumerate(doc.tables):
        covered = {(c.row, c.col) for c in t.cells}
        for c in t.cells:
            for dr in range(c.rowspan):
                for dc in range(c.colspan):
                    if dr == 0 and dc == 0:
                        continue
                    covered.add((c.row + dr, c.col + dc))
        expected = {(r, c) for r in range(t.num_rows) for c in range(t.num_cols)}
        if covered != expected:
            issues.append(ValidateIssue(level="warning", path=f"tables[{ti}]",
                                        message="cells do not tile the table"))

    # Form-field shape
    for i, el in enumerate(doc.layout):
        if not isinstance(el, FormFieldLayout):
            continue
        for j, f in enumerate(el.fields):
            if f.value is not None and f.value.__class__.__name__ == "NormalFieldValue":
                if f.value.text == "" and not f.value.handwritten:
                    issues.append(ValidateIssue(level="warning",
                                                path=f"layout[{i}].fields[{j}]",
                                                message="empty non-handwritten value"))

    # NFC
    for i, ln in enumerate(doc.lines):
        if unicodedata.normalize("NFC", ln.text) != ln.text:
            issues.append(ValidateIssue(level="warning", path=f"lines[{i}]",
                                        message="text is not NFC-normalized"))

    # Quad self-intersection
    for i, el in enumerate(doc.layout):
        if _is_self_intersecting(el.quad):
            issues.append(ValidateIssue(level="warning", path=f"layout[{i}]",
                                        message="quad is self-intersecting"))

    # Out-of-bounds
    for i, el in enumerate(doc.layout):
        x0, y0, x1, y1 = el.bbox
        if x0 < 0 or y0 < 0 or x1 > doc.width or y1 > doc.height:
            issues.append(ValidateIssue(level="warning", path=f"layout[{i}]",
                                        message="bbox outside image bounds"))

    return issues


def _is_self_intersecting(quad) -> bool:
    # signed area test; TL, TR, BR, BL must be a simple polygon
    n = len(quad)
    for i in range(n):
        a, b = quad[i], quad[(i + 1) % n]
        c, d = quad[(i + 2) % n], quad[(i + 3) % n]
        if _segments_intersect(a, b, c, d):
            return True
    return False


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
