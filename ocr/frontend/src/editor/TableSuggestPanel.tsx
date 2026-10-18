/**
 * "Suggested layout" for a table region: rows, columns and merged cells read
 * from the ruling lines printed on the scan inside the region's box
 * (POST /documents/{id}/suggest-table). Runs by itself when a table region
 * with no cells is selected (e.g. right after drawing it); the suggestion is
 * previewed on the canvas (dashed) and applied only on "Apply".
 */
import { useEffect, useState } from "react";

import { suggestTable, type TableSuggestion } from "../api";
import { useStore } from "../store";
import type { BBox, Doc } from "../types";
import { applyTableSuggestion } from "./commands";

/** table boxes whose suggestion was dismissed or applied: never suggested
 *  again by itself (the button still works) */
const handled = new Set<string>();
const keyOf = (docId: string, id: number, box: number[]) => `${docId}:${id}:${box.map(Math.round).join(",")}`;

const sameBox = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => Math.round(v) === Math.round(b[i]));

export function TableSuggestPanel({ doc, el }: { doc: Doc; el: any }) {
  const sug = useStore((s) => s.tableSuggestion);
  const setSug = useStore((s) => s.setTableSuggestion);
  const edit = useStore((s) => s.edit);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // the page right after Apply: Undo is offered only while Apply is still the last edit
  const [appliedDoc, setAppliedDoc] = useState<Doc | null>(null);
  const current = useStore((s) => s.doc);
  const undo = useStore((s) => s.undo);
  const key = keyOf(doc.document_id, el.id, el.bbox);

  const table = doc.tables[el.table_index];
  const hasCells = !!table?.cells?.length;
  const mine = sug && sug.elementId === el.id ? sug : null;
  const fresh = !!mine && sameBox(mine.box, el.bbox);

  const run = async () => {
    const box = [...el.bbox] as BBox;
    setBusy(true); setErr(null);
    try {
      const result = await suggestTable(doc.document_id, box);
      setSug({ elementId: el.id, box, result });
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  // a new (empty) table box: suggest by itself, once the box has stopped moving
  useEffect(() => {
    if (hasCells || fresh || busy || handled.has(key)) return;
    const t = setTimeout(() => void run(), 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [el.id, el.bbox.join(","), hasCells]);

  const apply = (r: TableSuggestion) => {
    if (hasCells && !window.confirm(
      `Replace the current ${table.cells.length} cell(s) with the suggested ${r.cells.length}? ` +
      "Text you typed is kept where it falls inside a new cell.",
    )) return;
    handled.add(key);
    edit((d) => applyTableSuggestion(d, el.id, r as any));
    // the region snaps to the table: don't suggest again for the new box either
    const now = useStore.getState().doc?.layout.find((e) => e.id === el.id);
    if (now) handled.add(keyOf(doc.document_id, el.id, now.bbox));
    setSug(null);
    setAppliedDoc(useStore.getState().doc);
  };
  const dismiss = () => {
    handled.add(key);
    setSug(null);
  };

  const r = mine?.result;
  return (
    <div className="space-y-2">
      <button
        type="button"
        disabled={busy}
        onClick={() => void run()}
        className="w-full rounded bg-teal-700 py-1 text-[11px] font-medium text-white hover:bg-teal-600 disabled:opacity-50"
        title="Read rows, columns and merged cells from the ruling lines printed inside this box"
      >
        {busy ? "Reading the scan…" : mine && !fresh ? "Box changed — suggest again" : "Suggest layout from scan"}
      </button>
      {err && <div className="break-words text-[10px] text-rose-400">{err}</div>}

      {appliedDoc && appliedDoc === current && !r && (
        <div className="flex items-center gap-2 rounded border border-slate-700 p-2 text-[11px] text-slate-300">
          <span className="flex-1">Suggested layout applied. Not right?</span>
          <button
            type="button"
            onClick={() => { undo(); setAppliedDoc(null); }}
            className="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600"
            title="Put the table back as it was before Apply"
          >Undo</button>
        </div>
      )}

      {r && !r.ok && (
        <ul className="space-y-0.5 text-[10px] text-amber-300">
          {r.warnings.map((w, i) => <li key={i}>⚠ {w}</li>)}
        </ul>
      )}

      {r && r.ok && (
        <div className="rounded border border-teal-800 bg-teal-950/40 p-2 text-[11px] text-slate-200">
          <div className="mb-1">
            <span className="font-medium">{r.num_rows} × {r.num_cols}</span>, {r.cells.length} cells
            {r.cells.length < r.num_rows * r.num_cols && (
              <span className="text-slate-400"> ({r.num_rows * r.num_cols - r.cells.length} merged)</span>
            )}
            {Math.abs(r.skew_deg) >= 0.2 && <span className="text-slate-400"> · page tilted {r.skew_deg}°</span>}
          </div>
          <GridPreview r={r} />
          {r.warnings.length > 0 && (
            <ul className="mt-1 space-y-0.5 text-[10px] text-amber-300">
              {r.warnings.map((w, i) => <li key={i}>⚠ {w}</li>)}
            </ul>
          )}
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              onClick={() => apply(r)}
              className="flex-1 rounded bg-teal-700 py-1 font-medium text-white hover:bg-teal-600"
              title="Use this grid for the table (undo works)"
            >Apply</button>
            <button
              type="button"
              onClick={dismiss}
              className="flex-1 rounded bg-slate-800 py-1 text-slate-300 hover:bg-slate-700"
              title="Keep the table as it is; not suggested again for this box"
            >Dismiss</button>
          </div>
          <div className="mt-1 text-[10px] text-slate-500">
            Dashed on the canvas. Then type the cell text (Table HTML), or run Table OCR.
          </div>
        </div>
      )}
    </div>
  );
}

/** The suggested grid as a small picture: merged cells are visible as bigger boxes. */
function GridPreview({ r }: { r: TableSuggestion }) {
  const W = 260;
  const H = Math.max(60, Math.min(200, Math.round((W * r.num_rows) / Math.max(1, r.num_cols) * 0.45)));
  const cw = W / Math.max(1, r.num_cols), rh = H / Math.max(1, r.num_rows);
  return (
    <svg width="100%" viewBox={`-1 -1 ${W + 2} ${H + 2}`} className="block rounded bg-slate-900">
      {r.cells.map((c) => {
        const merged = c.rowspan > 1 || c.colspan > 1;
        return (
          <rect key={c.id} x={c.col * cw} y={c.row * rh} width={c.colspan * cw} height={c.rowspan * rh}
                fill={merged ? "rgba(45,212,191,0.18)" : "none"} stroke="rgb(45,212,191)" strokeWidth={1} />
        );
      })}
    </svg>
  );
}
