import { useState } from "react";
import { detectLines } from "../api";
import { useStore } from "../store";

const TEXT_TYPES = new Set([
  "paragraph", "header", "title", "heading", "subheading", "list", "list_item",
  "form_field", "table_heading", "footer", "page_number",
]);

/**
 * "Detect lines": finds the printed lines on the scan inside each text
 * region and gives each one its own line box. Runs on the editor's current
 * (unsaved) page, so typed line breaks count; the result is an ordinary
 * edit (undo works) and still needs Save.
 */
export function DetectLinesButton() {
  const doc = useStore((s) => s.doc);
  const selection = useStore((s) => s.selection);
  const [busy, setBusy] = useState<null | "page" | "region">(null);
  const [info, setInfo] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  if (!doc) return null;
  const selected =
    selection?.kind === "layout" ? doc.layout.find((e) => e.id === selection.id) : undefined;
  const regionOk = !!selected && TEXT_TYPES.has(selected.type);

  async function run(scope: "page" | "region") {
    const sent = useStore.getState().doc;
    if (!sent) return;
    setBusy(scope); setErr(null); setInfo(null); setWarnings([]);
    try {
      const r = await detectLines(sent.document_id, {
        doc: sent,
        element_id: scope === "region" ? selected?.id ?? null : null,
      });
      if (useStore.getState().doc !== sent) {
        setErr("The page changed while lines were being detected — run it again.");
        return;
      }
      const before = sent.lines.length;
      useStore.getState().edit((d) => {
        d.lines = r.record.lines;
      });
      setWarnings(r.warnings);
      setInfo(`${r.record.lines.length} lines (was ${before}). Press Save to keep them.`);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  }

  const btn = (on: boolean) =>
    "flex-1 rounded py-1 text-[11px] font-medium " +
    (on ? "bg-teal-700 text-white hover:bg-teal-600" : "bg-slate-800 text-slate-500");

  return (
    <div className="mb-3 space-y-1 rounded border border-slate-800 p-2">
      <div className="flex gap-1">
        <button
          onClick={() => void run("page")}
          disabled={!!busy}
          className={btn(!busy)}
          title="Find the printed lines inside every text region on this page"
        >
          {busy === "page" ? "Detecting…" : "Detect lines: page"}
        </button>
        <button
          onClick={() => void run("region")}
          disabled={!!busy || !regionOk}
          className={btn(!busy && regionOk)}
          title={regionOk ? "Only the selected region" : "Select a text region first"}
        >
          {busy === "region" ? "Detecting…" : "Selected region"}
        </button>
      </div>
      {info && <div className="text-[10px] text-emerald-400">{info}</div>}
      {err && <div className="break-words text-[10px] text-rose-400">{err}</div>}
      {warnings.length > 0 && (
        <ul className="space-y-0.5 text-[10px] text-amber-300">
          {warnings.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
