/**
 * Nested-region helpers.
 *
 * "A is inside B" uses the same rule as the exporter (normalizeDoc.ts):
 * at least 80 % of A's area lies inside B, and B is at least as big.
 * Two identical boxes are ordered by id so they never contain each other.
 */
import type { BBox, Doc, LayoutElement } from "../types";
import type { Selection } from "../store";
import { parentLayoutOf } from "./commands";

export const NEST_THRESHOLD = 0.8;

export function bboxArea(b: BBox): number {
  return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
}

/** Fraction of `inner` that lies inside `outer` (0..1). */
export function insideFrac(inner: BBox, outer: BBox): number {
  const ix = Math.max(0, Math.min(inner[2], outer[2]) - Math.max(inner[0], outer[0]));
  const iy = Math.max(0, Math.min(inner[3], outer[3]) - Math.max(inner[1], outer[1]));
  const a = bboxArea(inner);
  return a > 0 ? (ix * iy) / a : 0;
}

/** Does region `p` contain region `c`? */
export function contains(p: LayoutElement, c: LayoutElement): boolean {
  if (p.id === c.id) return false;
  if (insideFrac(c.bbox, p.bbox) < NEST_THRESHOLD) return false;
  const ap = bboxArea(p.bbox);
  const ac = bboxArea(c.bbox);
  return ap > ac || (ap === ac && p.id < c.id);
}

/** Smallest region that contains `el` (its direct parent), if any. */
export function containerOf(layout: LayoutElement[], el: LayoutElement): LayoutElement | undefined {
  return layout
    .filter((p) => contains(p, el))
    .sort((a, b) => bboxArea(a.bbox) - bboxArea(b.bbox))[0];
}

/** For every region: id of its direct container and how deep it is nested. */
export function nestingMap(layout: LayoutElement[]): Map<number, { parent: number | null; depth: number }> {
  const parent = new Map<number, number | null>();
  for (const el of layout) parent.set(el.id, containerOf(layout, el)?.id ?? null);

  const depthOf = (id: number, seen: Set<number>): number => {
    const p = parent.get(id);
    if (p == null || seen.has(id)) return 0;
    seen.add(id);
    return 1 + depthOf(p, seen);
  };

  const out = new Map<number, { parent: number | null; depth: number }>();
  for (const el of layout) {
    out.set(el.id, { parent: parent.get(el.id) ?? null, depth: depthOf(el.id, new Set()) });
  }
  return out;
}

/** Alt+↑ — the region around the current selection. */
export function outerSelection(doc: Doc, sel: Selection): Selection {
  if (!sel) return null;
  if (sel.kind !== "layout") {
    const p = parentLayoutOf(doc, sel);
    return p != null ? { kind: "layout", id: p } : null;
  }
  const el = doc.layout.find((e: LayoutElement) => e.id === sel.id);
  if (!el) return null;
  const outer = containerOf(doc.layout, el);
  return outer ? { kind: "layout", id: outer.id } : null;
}

/** Alt+↓ — the biggest region inside the selected one, else its first line/cell/field. */
export function innerSelection(doc: Doc, sel: Selection): Selection {
  if (!sel || sel.kind !== "layout") return null;
  const el = doc.layout.find((e: LayoutElement) => e.id === sel.id);
  if (!el) return null;

  const inner = doc.layout
    .filter((c: LayoutElement) => contains(el, c))
    .sort((a: LayoutElement, b: LayoutElement) => bboxArea(b.bbox) - bboxArea(a.bbox))[0];
  if (inner) return { kind: "layout", id: inner.id };

  const li = doc.lines.findIndex((l: { block_id: number | null }) => l.block_id === el.id);
  if (li >= 0) return { kind: "line", index: li };
  if (el.type === "form_field" && el.fields.length) {
    return { kind: "field", elementId: el.id, fieldIndex: 0 };
  }
  if (el.type === "table") {
    const ti = (el as any).table_index as number;
    const c = doc.tables[ti]?.cells[0];
    if (c) return { kind: "cell", tableIndex: ti, cellId: c.id };
  }
  return null;
}
