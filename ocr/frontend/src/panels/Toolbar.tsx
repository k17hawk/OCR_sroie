import { useEffect } from "react";
import { saveCurrent } from "../editor/save";
import { useStore, type Tool } from "../store";
import { preannotate, getJob, getDocument } from "../api";
import { ModelPicker } from "../editor/ModelPicker";
import { DocTypePicker } from "../editor/DocTypePicker";
import { usePageNav } from "../hooks/usePageNav";
import { useProjectTypes, withCurrent, prettyType } from "../hooks/useProjectTypes";
import { exportCurrentDoc } from "../editor/exportJson";
import { useOcrPrefs } from "../hooks/useOcrPrefs";
import { autoReadingOrder, clearReadingOrder, nextReadingOrder } from "../editor/commands";

const TOOLS: { id: Tool; key: string; label: string }[] = [
  { id: "select",       key: "V", label: "Select" },
  { id: "pan",          key: "H", label: "Pan" },
  { id: "draw-box",     key: "B", label: "Box" },
  { id: "draw-quad",    key: "Q", label: "Quad" },
  { id: "line",         key: "L", label: "Line" },
  { id: "form-field",   key: "F", label: "Form field" },
  { id: "table",        key: "T", label: "Table" },
  { id: "list",         key: "I", label: "List" },
  { id: "stamp",        key: "S", label: "Stamp" },
  { id: "signature",    key: "G", label: "Signature" },
  { id: "logo",         key: "O", label: "Logo" },
  { id: "figure",       key: "U", label: "Figure" },
  { id: "checkbox",     key: "C", label: "Checkbox" },
  { id: "fingerprint",  key: "K", label: "Fingerprint" },
  { id: "reading-order",key: "R", label: "Reading order" },
];

export function Toolbar() {
  const tool = useStore((s) => s.tool);
  const setTool = useStore((s) => s.setTool);
  const doc = useStore((s) => s.doc);
  const dirty = useStore((s) => s.dirty);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const nav = usePageNav(doc?.document_id ?? null);
  const ocr = useOcrPrefs();
  const ocrFamily = ocr.family;
  const ocrDevice = ocr.device;
  const setOcrFamily = ocr.setFamily;
  const setOcrDevice = ocr.setDevice;
  const edit = useStore((s) => s.edit);
  const drawType = useStore((s) => s.drawTypeDefault);
  const types = useProjectTypes();

  const saveStatus = useStore((s) => s.saveStatus);
  const setSaveStatus = useStore((s) => s.setSaveStatus);
  const save = () => { void saveCurrent(); };

  // Ctrl+S / ⌘+S saves (also while typing in a text box)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (useStore.getState().dirty) void saveCurrent();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // "Saved" fades after a few seconds
  useEffect(() => {
    if (saveStatus.state !== "saved") return;
    const t = setTimeout(() => setSaveStatus({ state: "idle" }), 2500);
    return () => clearTimeout(t);
  }, [saveStatus, setSaveStatus]);

  async function autoAnnotate() {
    if (!doc) return;
    const job = await preannotate(doc.document_id, {
      family: ocrFamily,
      device: ocrDevice,
    });
    const t = setInterval(async () => {
      const j = await getJob(job.id);
      if (j.status === "succeeded" || j.status === "failed") {
        clearInterval(t);
        if (j.status === "succeeded") {
          const fresh = await getDocument(doc.document_id);
          useStore.getState().reloadIfClean(
            fresh.doc, fresh.revision, fresh.image_url,
          );
        }
      }
    }, 800);
  }

  return (
    <div className="flex items-center gap-1 border-b border-slate-800 bg-slate-900 px-3 py-2">
      {TOOLS.map((t) => (
        <button
          key={t.id}
          onClick={() => setTool(t.id)}
          className={
            "rounded px-2 py-1 text-[11px] font-medium transition-colors " +
            (tool === t.id
              ? "bg-sky-600 text-white"
              : "text-slate-300 hover:bg-slate-800")
          }
          title={`${t.label} (${t.key})`}
        >
          {t.label}
        </button>
      ))}

      {tool === "reading-order" && doc && (
        <div className="ml-1 flex items-center gap-1 rounded bg-sky-950/60 px-1.5 py-0.5 text-[11px]">
          <span className="text-sky-200" title="Click boxes in order. Shift+click removes one.">
            next #{nextReadingOrder(doc)}
          </span>
          <button
            onClick={() => edit((d) => autoReadingOrder(d))}
            className="rounded px-1.5 py-0.5 text-sky-100 hover:bg-sky-800"
            title="Number every box top-to-bottom, left-to-right"
          >
            Auto
          </button>
          <button
            onClick={() => edit((d) => clearReadingOrder(d))}
            className="rounded px-1.5 py-0.5 text-rose-200 hover:bg-rose-900/60"
            title="Remove all numbers, then click boxes in order"
          >
            Clear
          </button>
        </div>
      )}

      {(tool === "draw-box" || tool === "draw-quad") && (
        <select
          value={drawType}
          onChange={(e) =>
            useStore.setState({ drawTypeDefault: e.target.value } as any)
          }
          className="ml-1 rounded bg-slate-800 px-1.5 py-1 text-[11px] text-slate-100"
          title="Type given to new boxes drawn with Box / Quad"
        >
          {withCurrent(types.layout_types, drawType).map((t) => (
            <option key={t} value={t}>{prettyType(t)}</option>
          ))}
        </select>
      )}

      <div className="mx-2 h-5 w-px bg-slate-700" />

      <div className="min-w-[200px]">
        <DocTypePicker />

        <div className="flex items-center gap-1 rounded bg-slate-800 px-1 py-0.5">
  <button
    onClick={nav.goPrev}
    disabled={!nav.prev}
    className={
      "rounded px-1.5 py-0.5 text-[11px] " +
      (nav.prev
        ? "text-slate-200 hover:bg-slate-700"
        : "cursor-not-allowed text-slate-600")
    }
    title={nav.prev
      ? `Previous page (${nav.prev.document_id})`
      : "No previous page"}
  >
    ◀
  </button>
  <span className="px-1 font-mono text-[11px] text-slate-400">
    {nav.position}
  </span>
  <button
    onClick={nav.goNext}
    disabled={!nav.next}
    className={
      "rounded px-1.5 py-0.5 text-[11px] " +
      (nav.next
        ? "text-slate-200 hover:bg-slate-700"
        : "cursor-not-allowed text-slate-600")
    }
    title={nav.next
      ? `Next page (${nav.next.document_id})`
      : "No next page"}
  >
    ▶
  </button>
</div>

      </div>

      <div className="ml-auto flex items-center gap-2">
        <ModelPicker
          family={ocrFamily}
          device={ocrDevice}
          onFamily={setOcrFamily}
          onDevice={setOcrDevice}
        />

        <button
          onClick={undo}
          className="rounded px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-800"
          title="Undo (Ctrl+Z)"
        >
          Undo
        </button>
        <button
          onClick={redo}
          className="rounded px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-800"
          title="Redo (Ctrl+Shift+Z)"
        >
          Redo
        </button>
        <div className="relative">
          <button
            onClick={save}
            disabled={!dirty || saveStatus.state === "saving"}
            className={
              "rounded px-2 py-1 text-[11px] font-medium " +
              (saveStatus.state === "error"
                ? "bg-rose-600 text-white hover:bg-rose-500"
                : dirty
                  ? "bg-emerald-600 text-white hover:bg-emerald-500"
                  : "bg-slate-800 text-slate-500")
            }
            title="Save (Ctrl+S)"
          >
            {saveStatus.state === "saving"
              ? "Saving…"
              : saveStatus.state === "saved" && !dirty
                ? "Saved ✓"
                : "Save"}
          </button>
          {saveStatus.state === "error" && (
            <div className="absolute right-0 top-full z-50 mt-1 w-80 whitespace-pre-wrap rounded border border-rose-700 bg-rose-950 p-2 text-[11px] text-rose-100 shadow-lg">
              <button
                onClick={() => setSaveStatus({ state: "idle" })}
                className="float-right ml-2 text-rose-300 hover:text-white"
                title="Dismiss"
              >
                ✕
              </button>
              {saveStatus.message}
            </div>
          )}
        </div>
        <button
          onClick={() => {
            if (!doc) return;
            const w = exportCurrentDoc(doc);
            // send the annotator to the JSON tab if something needs fixing
            if (w.some((x) => x.level === "check")) {
              (useStore.getState() as any).setRightTab?.("json");
            }
          }}
          disabled={!doc}
          className="rounded px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-800 disabled:text-slate-600"
          title="Download this page in the export (v2) format, including unsaved edits"
        >
          Export JSON
        </button>
        <button
          onClick={autoAnnotate}
          className="rounded bg-violet-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-violet-500"
          title="Runs the selected model on the whole page"
        >
          Auto-annotate
        </button>
      </div>
    </div>
  );
}