import { useEffect, useRef, useState } from "react";
import { ocrTable } from "../api";
import { useStore } from "../store";
import { ModelPicker } from "./ModelPicker";
import { useOcrPrefs } from "../hooks/useOcrPrefs";
import { syncTable, tableText } from "./commands";
import type { Doc } from "../types";

interface Props {
  documentId: string;
  /** The table layout element's id (mutated in place on success). */
  tableElementId: number;
  /** The table's current quad, sent as the OCR region. */
  quad: [number, number][];
}

/** rows × cols + every cell's span: changes when the HTML adds/removes cells */
function shapeOf(d: Doc | null, elementId: number): string {
  const el = d?.layout.find((e) => e.id === elementId) as any;
  const t = el && d?.tables[el.table_index];
  if (!t || !t.cells?.length) return "";
  return `${t.num_rows}x${t.num_cols}:` +
    t.cells.map((c: any) => `${c.row},${c.col},${c.rowspan},${c.colspan}`).join(";");
}

export function TableOcrButton({ documentId, tableElementId, quad }: Props) {
  const { family, device, lang, setFamily, setDevice } = useOcrPrefs();

  const [busy, setBusy] = useState<null | "ocr" | "fit">(null);
  const [info, setInfo] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  function apply(table: any, sentHtml?: string) {
    useStore.getState().edit((d) => {
      const el = d.layout.find((e) => e.id === tableElementId) as any;
      if (!el || el.type !== "table") return;
      const idx = el.table_index;
      if (typeof idx !== "number" || idx < 0 || idx >= d.tables.length) return;
      // typed more while the request ran: don't overwrite it
      if (sentHtml !== undefined && tableText(d, el) !== sentHtml) return;
      // lines that were only placeholders for the OLD cells (same box as a
      // cell) would be kept as "text boxes" and sit in the wrong place
      const oldBoxes = new Set((d.tables[idx]?.cells ?? []).map((c: any) => c.bbox.join(",")));
      d.lines = d.lines.filter((ln) => !(ln.block_id === tableElementId && oldBoxes.has(ln.bbox.join(","))));
      d.tables[idx] = table;
      // one line per cell + ruling lines for the detected grid
      syncTable(d, tableElementId, { ruling: true });
    });
  }

  async function run() {
    setBusy("ocr"); setErr(null); setInfo(null);
    try {
      const r = await ocrTable(documentId, { quad, family, device, lang_hint: lang });
      apply(r.table);
      setInfo(`${r.model} · ${r.table.num_rows}×${r.table.num_cols} · ${r.table.cells.length} cells`);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  }

  /** Keep the typed rows/columns/spans/text, move the cells onto the image's lines. */
  async function fit(auto = false) {
    const st = useStore.getState();
    const el = st.doc?.layout.find((e) => e.id === tableElementId) as any;
    if (!st.doc || !el) return;
    const html = tableText(st.doc, el);
    if (!html.trim()) return;
    setBusy("fit"); setErr(null); setInfo(null);
    try {
      const r = await ocrTable(documentId, { quad: el.quad, family, device, lang_hint: lang, html });
      apply(r.table, html);
      setInfo(`${auto ? "Fitted to image" : "Fitted"} · ${r.table.num_rows}×${r.table.num_cols} · ${r.table.cells.length} cells`);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  }

  // Typed / pasted HTML that changes the table's shape used to be laid out
  // as an even grid (every row the same height), so cells missed the scan.
  // Now the new structure is fitted to the image a moment after you stop typing.
  const shape = useStore((s) => shapeOf(s.doc, tableElementId));
  const last = useRef<{ id: number; shape: string } | null>(null);
  useEffect(() => {
    const prev = last.current;
    last.current = { id: tableElementId, shape };
    if (!prev || prev.id !== tableElementId || prev.shape === shape || !shape) return;
    const t = setTimeout(() => void fit(true), 900);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shape, tableElementId]);

  return (
    <div className="space-y-1">
      <ModelPicker
        family={family}
        device={device}
        onFamily={setFamily}
        onDevice={setDevice}
        compact
      />
      <button
        onClick={run}
        disabled={!!busy}
        className={
          "w-full rounded py-1 text-[11px] font-medium " +
          (busy
            ? "bg-slate-800 text-slate-500"
            : "bg-orange-700 text-white hover:bg-orange-600")
        }
      >
        {busy === "ocr" ? "Detecting table…" : "Run table OCR"}
      </button>
      <button
        onClick={() => void fit()}
        disabled={!!busy}
        className={
          "w-full rounded py-1 text-[11px] font-medium " +
          (busy ? "bg-slate-800 text-slate-500" : "bg-slate-700 text-white hover:bg-slate-600")
        }
        title="Keep the rows, columns, spans and text of the Table HTML; move the cells onto the table's lines in the image"
      >
        {busy === "fit" ? "Fitting…" : "Fit cells to image"}
      </button>
      {info && <div className="text-[10px] text-emerald-400">{info}</div>}
      {err && <div className="break-words text-[10px] text-rose-400">{err}</div>}
    </div>
  );
}
