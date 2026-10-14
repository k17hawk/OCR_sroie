import { useSyncExternalStore } from "react";
import type { Selection } from "../store";
import type { Doc } from "../types";
import { typeColor, prettyType, KIND_COLORS } from "../hooks/useProjectTypes";

type Sel = Exclude<Selection, null>;

export interface PickEntry {
  key: string;
  selection: Sel;
  label: string;
  sub: string;
  color: string;
  depth: number; // 0 = region, 1 = line / cell / field inside it
}

interface State {
  menu: { x: number; y: number; entries: PickEntry[] } | null;
  /** key of the entry the mouse is over in the menu (outlined on canvas) */
  preview: string | null;
}

let state: State = { menu: null, preview: null };
const listeners = new Set<() => void>();
function set(patch: Partial<State>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

export const pickMenu = {
  subscribe(l: () => void) {
    listeners.add(l);
    return () => { listeners.delete(l); };
  },
  get: () => state,
  open(x: number, y: number, entries: PickEntry[]) {
    set({ menu: { x, y, entries }, preview: null });
  },
  close() {
    if (state.menu || state.preview) set({ menu: null, preview: null });
  },
  setPreview(key: string | null) {
    if (state.preview !== key) set({ preview: key });
  },
};

export function usePickMenu(): State {
  return useSyncExternalStore(pickMenu.subscribe, pickMenu.get);
}

const snippet = (s: unknown) =>
  typeof s === "string" ? s.replace(/\s+/g, " ").trim().slice(0, 28) : "";

/** Short label + color for anything selectable. */
export function describeSelection(doc: Doc, sel: Sel): { label: string; sub: string; color: string } {
  switch (sel.kind) {
    case "layout": {
      const el = doc.layout.find((e: any) => e.id === sel.id);
      return {
        label: `#${sel.id} ${el ? prettyType(el.type) : "?"}`,
        sub: snippet(el && "text" in el ? (el as any).text : ""),
        color: typeColor(el?.type ?? "?"),
      };
    }
    case "line": {
      const ln = doc.lines[sel.index];
      return {
        label: `line ${sel.index}`,
        sub: snippet(ln?.text),
        color: ln?.is_heading ? KIND_COLORS.lineHeading : KIND_COLORS.line,
      };
    }
    case "cell": {
      const c = doc.tables[sel.tableIndex]?.cells.find((x: any) => x.id === sel.cellId);
      return {
        label: c ? `cell r${c.row} c${c.col}` : `cell ${sel.cellId}`,
        sub: snippet(c?.text),
        color: c?.is_header ? KIND_COLORS.cellHeader : KIND_COLORS.cell,
      };
    }
    case "field": {
      const el = doc.layout.find((e: any) => e.id === sel.elementId);
      const f = el?.type === "form_field" ? el.fields[sel.fieldIndex] : undefined;
      return {
        label: `field ${sel.fieldIndex} of #${sel.elementId}`,
        sub: snippet(f?.label?.text),
        color: KIND_COLORS.fieldLabel,
      };
    }
    case "extraRegion": {
      const er = doc.extra_regions[sel.index];
      const checked = er?.type === "checkbox" && er.checked;
      return {
        label: `${er?.type ?? "extra"} ${sel.index}`,
        sub: er?.type === "checkbox" ? snippet(er.option) : "",
        color: er?.type === "fingerprint"
          ? KIND_COLORS.fingerprint
          : checked ? KIND_COLORS.checked : KIND_COLORS.unchecked,
      };
    }
  }
}
