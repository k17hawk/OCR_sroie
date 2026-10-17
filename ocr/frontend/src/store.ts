import { create } from "zustand";
import { produce, enableMapSet } from "immer";
import type { Doc, LoadWarning, Project } from "./types";
import type { TableSuggestion } from "./api";
import { LAYER_DEFAULTS, type LayerKey } from "./lib";

enableMapSet();

export type Selection =
  | { kind: "layout"; id: number }
  | { kind: "line"; index: number }
  | { kind: "field"; elementId: number; fieldIndex: number }
  | { kind: "cell"; tableIndex: number; cellId: number }
  | { kind: "extraRegion"; index: number }
  | null;

export type Tool =
  | "select"
  | "pan"
  | "draw-box"
  | "draw-quad"
  | "line"
  | "form-field"
  | "table"
  | "list"
  | "stamp"
  | "signature"
  | "logo"
  | "figure"
  | "checkbox"
  | "fingerprint"
  | "reading-order";

/** How strongly annotations are painted over the scan (Layers panel). */
export interface OverlayPrefs {
  /** 0.1 … 1 — opacity of every box, line, cell and chip */
  opacity: number;
  /** draw each box's text on top of the scan (hides the scan underneath) */
  showText: boolean;
}

const OVERLAY_KEY = "devalabel.overlay";
function loadOverlay(): OverlayPrefs {
  const def: OverlayPrefs = { opacity: 0.6, showText: false };
  try {
    const v = JSON.parse(localStorage.getItem(OVERLAY_KEY) ?? "null");
    if (v && typeof v.opacity === "number") {
      return { opacity: Math.min(1, Math.max(0.1, v.opacity)), showText: !!v.showText };
    }
  } catch {
    /* storage blocked: defaults */
  }
  return def;
}

export type SaveStatus =
  | { state: "idle" }
  | { state: "saving" }
  | { state: "saved"; at: number }
  | { state: "error"; message: string };

export type RightTab =
  | "properties"
  | "tree"
  | "readingOrder"
  | "pageMeta"
  | "json";

interface State {
  doc: Doc | null;
  revision: number;
  imageUrl: string | null;
  dirty: boolean;

  past: Doc[];
  future: Doc[];

  load: (doc: Doc, revision: number, imageUrl: string | null) => void;
  reloadIfClean: (
    doc: Doc,
    revision: number,
    imageUrl: string | null
  ) => void;
  edit: (recipe: (d: Doc) => void) => void;
  undo: () => void;
  redo: () => void;
  markSaved: (revision: number) => void;
  saveStatus: SaveStatus;
  setSaveStatus: (s: SaveStatus) => void;
  /** what the server changed to load this (older) page; shown until dismissed */
  loadWarnings: LoadWarning[];
  setLoadWarnings: (w: LoadWarning[]) => void;

  selection: Selection;
  setSelection: (s: Selection) => void;

  /** a table layout suggested from the scan, shown in the right panel and on the canvas */
  tableSuggestion: { elementId: number; box: number[]; result: TableSuggestion } | null;
  setTableSuggestion: (s: { elementId: number; box: number[]; result: TableSuggestion } | null) => void;

  tool: Tool;
  setTool: (t: Tool) => void;
  drawTypeDefault: string;
  setDrawTypeDefault: (t: string) => void;

  layers: Record<LayerKey, boolean>;
  toggleLayer: (k: LayerKey) => void;
  overlay: OverlayPrefs;
  setOverlay: (p: Partial<OverlayPrefs>) => void;

  scale: number;
  offset: { x: number; y: number };
  setViewport: (scale: number, offset: { x: number; y: number }) => void;
  fitRequest: number;
  requestFit: () => void;

  rightTab: RightTab;
  setRightTab: (id: RightTab) => void;

  // model / device defaults
  ocrFamily: string;
  ocrDevice: string;
  setOcrFamily: (f: string) => void;
  setOcrDevice: (d: string) => void;

  // conflict tracking
  conflict: "none" | "remote-newer" | "remote-changed";
  setConflict: (c: State["conflict"]) => void;

  // revision tracking
  serverRevision: number; // revision currently on the server
  setServerRevision: (r: number) => void;

  // multi-page navigation
  currentProjectId: number;
  setCurrentProjectId: (id: number) => void;

  // active project (doc_types, field_types, label_map, …)
  project: Project | null;
  setProject: (p: Project | null) => void;
}

export const useStore = create<State>((set, get) => ({
  doc: null,
  revision: 1,
  imageUrl: null,
  dirty: false,
  past: [],
  future: [],

  load: (doc, revision, imageUrl) =>
    set({
      doc,
      revision,
      imageUrl,
      dirty: false,
      past: [],
      future: [],
      selection: null,
      serverRevision: revision,
      conflict: "none",
      saveStatus: { state: "idle" },
      tableSuggestion: null, // a suggestion belongs to the page it was made for
    }),

  reloadIfClean: (doc, revision, imageUrl) => {
    const s = get();
    if (s.dirty) {
      set({ conflict: "remote-newer" });
      return;
    }
    set({
      doc,
      revision,
      imageUrl,
      serverRevision: revision,
      past: [],
      future: [],
      dirty: false,
      conflict: "none",
    });
  },

  edit: (recipe) => {
    const { doc, past } = get();
    if (!doc) return;
    const next = produce(doc, recipe);
    if (next === doc) return; // no-op
    set({
      doc: next,
      past: [...past.slice(-50), doc],
      future: [],
      dirty: true,
    });
  },

  undo: () => {
    const { past, doc, future } = get();
    if (past.length === 0 || !doc) return;
    const prev = past[past.length - 1];
    set({
      doc: prev,
      past: past.slice(0, -1),
      future: [doc, ...future].slice(0, 50),
      dirty: true,
    });
  },

  redo: () => {
    const { future, doc, past } = get();
    if (future.length === 0 || !doc) return;
    const next = future[0];
    set({
      doc: next,
      past: [...past, doc].slice(-50),
      future: future.slice(1),
      dirty: true,
    });
  },

  markSaved: (revision) =>
    set({ revision, serverRevision: revision, dirty: false }),
  saveStatus: { state: "idle" },
  setSaveStatus: (saveStatus) => set({ saveStatus }),
  tableSuggestion: null,
  setTableSuggestion: (tableSuggestion) => set({ tableSuggestion }),
  loadWarnings: [],
  setLoadWarnings: (loadWarnings) => set({ loadWarnings }),

  selection: null,
  setSelection: (selection) => set({ selection }),

  tool: "select",
  setTool: (tool) => set({ tool }),
  drawTypeDefault: "paragraph",
  setDrawTypeDefault: (drawTypeDefault) => set({ drawTypeDefault }),

  layers: { ...LAYER_DEFAULTS },
  toggleLayer: (k) =>
    set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),
  overlay: loadOverlay(),
  setOverlay: (p) =>
    set((s) => {
      const overlay = { ...s.overlay, ...p };
      try { localStorage.setItem(OVERLAY_KEY, JSON.stringify(overlay)); } catch { /* ignore */ }
      return { overlay };
    }),

  scale: 1,
  offset: { x: 0, y: 0 },
  setViewport: (scale, offset) => set({ scale, offset }),
  fitRequest: 0,
  requestFit: () => set((s) => ({ fitRequest: s.fitRequest + 1 })),

  rightTab: "properties",
  setRightTab: (rightTab) => set({ rightTab }),

  // model / device defaults
  ocrFamily: "mock",
  ocrDevice: "cpu",
  setOcrFamily: (ocrFamily) => set({ ocrFamily }),
  setOcrDevice: (ocrDevice) => set({ ocrDevice }),

  // conflict tracking
  conflict: "none",
  setConflict: (conflict) => set({ conflict }),

  // revision tracking
  serverRevision: 1,
  setServerRevision: (serverRevision) => set({ serverRevision }),

  // multi-page navigation
  currentProjectId: 1,
  setCurrentProjectId: (currentProjectId) => set({ currentProjectId }),

  // active project
  project: null,
  setProject: (project) => set({ project }),
}));