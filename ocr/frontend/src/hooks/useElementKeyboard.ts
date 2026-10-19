import { useEffect } from "react";
import { useStore, type Selection } from "../store";
import type { BBox, Doc, Quad } from "../types";
import { translateLayoutElement } from "../editor/commands";
import { outerSelection, innerSelection } from "../editor/nesting";

type Flat = { sel: Exclude<Selection, null>; bbox: [number, number, number, number] };

/**
 * Global annotator shortcuts (installed once, ignored while typing):
 *   Tab / Shift+Tab, [ / ]     cycle selection (visual order: y then x)
 *   Alt+↑                      select the region AROUND the selection (nested boxes)
 *   Alt+↓                      select the biggest region / first line INSIDE it
 *   Arrows                     nudge selected bbox by 1 px
 *   Shift + Arrows             nudge by 10 px
 *   Delete / Backspace         delete selected layout element
 *   Esc                        clear selection
 *
 * Mouse (canvas): click again on the same spot walks inner → outer,
 * Shift+click jumps outward, Alt+click picks the smallest thing,
 * right-click lists everything under the cursor.
 */
export function useElementKeyboard() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ignore keys meant for a form control. SELECT was missing: with the
      // Type dropdown focused, ArrowDown nudged the box instead of changing
      // the type, Esc closed the panel and Backspace deleted the box.
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" ||
          t.tagName === "BUTTON" || t.isContentEditable || t.closest?.("[role=listbox],[role=menu]"))
      ) return;

      const s = useStore.getState();
      if (!s.doc) return;
      const doc = s.doc;

      if (e.key === "Escape") {
        // first Esc cancels a pending "draw value box"
        if (s.valueTarget) { s.setValueTarget(null); s.setTool("select"); return; }
        s.setSelection(null);
        return;
      }

      // Walk nested boxes: Alt+↑ = outer, Alt+↓ = inner
      if (e.altKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
        e.preventDefault();
        const next = e.key === "ArrowUp"
          ? outerSelection(doc, s.selection)
          : innerSelection(doc, s.selection);
        if (next) s.setSelection(next);
        return;
      }

      const flat = flatten(doc);
      if (flat.length === 0) return;
      const keyOf = (sel: Exclude<Selection, null>) => JSON.stringify(sel);

      const cycle = (dir: 1 | -1) => {
        const cur = s.selection ? keyOf(s.selection) : null;
        const idx = cur ? flat.findIndex((f) => keyOf(f.sel) === cur) : -1;
        const next = (idx + dir + flat.length) % flat.length;
        s.setSelection(flat[next].sel);
      };

      // Tab / Shift+Tab / ] / [ all cycle the selection.
      if (e.key === "Tab" || e.key === "]" || e.key === "[") {
        e.preventDefault();
        const backwards = e.key === "[" || (e.key === "Tab" && e.shiftKey);
        cycle(backwards ? -1 : 1);
        return;
      }

      const step = e.shiftKey ? 10 : 1;
      const d = ({
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
        ArrowUp: [0, -step],
        ArrowDown: [0, step],
      } as Record<string, [number, number]>)[e.key];

      if (d && s.selection) {
        e.preventDefault();
        const sel = s.selection;
        s.edit((doc) => nudge(doc, sel, d[0], d[1]));
      }

      if ((e.key === "Delete" || e.key === "Backspace") && s.selection?.kind === "layout") {
        e.preventDefault();
        const id = s.selection.id;
        s.edit((d) => {
          d.layout = d.layout.filter((el) => el.id !== id);
          d.lines = d.lines.filter((ln) => ln.block_id !== id);
        });
        s.setSelection(null);
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

function flatten(doc: Doc): Flat[] {
  const out: Flat[] = [];
  for (const el of doc.layout) {
    out.push({ sel: { kind: "layout", id: el.id }, bbox: el.bbox });
    if (el.type === "form_field") {
      el.fields.forEach((f, fi) => {
        const bbox = f.label?.bbox ?? f.value?.bbox;
        if (bbox) out.push({ sel: { kind: "field", elementId: el.id, fieldIndex: fi }, bbox });
      });
    }
  }
  doc.lines.forEach((ln, i) => out.push({ sel: { kind: "line", index: i }, bbox: ln.bbox }));
  doc.tables.forEach((t, ti) =>
    t.cells.forEach((c) =>
      out.push({ sel: { kind: "cell", tableIndex: ti, cellId: c.id }, bbox: c.bbox }),
    ),
  );
  doc.extra_regions.forEach((er, i) =>
    out.push({ sel: { kind: "extraRegion", index: i }, bbox: er.bbox }),
  );
  // visual order: top-to-bottom, then left-to-right
  out.sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0]);
  return out;
}

function nudge(d: Doc, sel: Exclude<Selection, null>, dx: number, dy: number) {
  const shift = (obj: { bbox: [number, number, number, number]; quad?: [number, number][] }) => {
    obj.bbox = [obj.bbox[0] + dx, obj.bbox[1] + dy, obj.bbox[2] + dx, obj.bbox[3] + dy];
    if (obj.quad) obj.quad = obj.quad.map((p) => [p[0] + dx, p[1] + dy]) as typeof obj.quad;
  };

  switch (sel.kind) {
    case "layout": {
      translateLayoutElement(d, sel.id, dx, dy);
      break;
    }
    case "line":
      if (d.lines[sel.index]) shift(d.lines[sel.index]);
      break;
    case "cell": {
      const c = d.tables[sel.tableIndex]?.cells.find((x) => x.id === sel.cellId);
      if (c) shift(c);
      break;
    }
    case "field": {
      const el = d.layout.find((e) => e.id === sel.elementId);
      if (el && el.type === "form_field") {
        const f = el.fields[sel.fieldIndex];
        if (!f) break;
        if (f.label?.bbox) shift(f.label as { bbox: BBox; quad?: Quad });
        if (f.value) shift(f.value);
      }
      break;
    }
    case "extraRegion": {
      const er = d.extra_regions[sel.index];
      if (er) shift(er);
      break;
    }
  }
}
