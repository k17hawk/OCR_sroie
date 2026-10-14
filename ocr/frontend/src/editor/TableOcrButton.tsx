import { useState } from "react";
import { ocrTable } from "../api";
import { useStore } from "../store";
import { ModelPicker } from "./ModelPicker";
import { useOcrPrefs } from "../hooks/useOcrPrefs";
import { syncTable } from "./commands";

interface Props {
  documentId: string;
  /** The table layout element's id (mutated in place on success). */
  tableElementId: number;
  /** The table's current quad, sent as the OCR region. */
  quad: [number, number][];
}

export function TableOcrButton({ documentId, tableElementId, quad }: Props) {
  const { family, device, lang, setFamily, setDevice } = useOcrPrefs();

  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    setInfo(null);
    try {
      const r = await ocrTable(documentId, {
        quad,
        family,
        device,
        lang_hint: lang,
      });

      // Replace the selected table's cells in place.
      useStore.getState().edit((d) => {
        const el = d.layout.find((e) => e.id === tableElementId) as any;
        if (!el || el.type !== "table") return;
        const idx = el.table_index;
        if (typeof idx !== "number" || idx < 0 || idx >= d.tables.length) return;
        d.tables[idx] = r.table;
        // one line per cell + ruling lines for the detected grid
        syncTable(d, tableElementId, { ruling: true });
      });

      setInfo(
        `${r.model} · ${r.table.num_rows}×${r.table.num_cols} · ${r.table.cells.length} cells`,
      );
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

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
        disabled={busy}
        className={
          "w-full rounded py-1 text-[11px] font-medium " +
          (busy
            ? "bg-slate-800 text-slate-500"
            : "bg-orange-700 text-white hover:bg-orange-600")
        }
      >
        {busy ? "Detecting table…" : "Run table OCR"}
      </button>
      {info && <div className="text-[10px] text-emerald-400">{info}</div>}
      {err && <div className="break-words text-[10px] text-rose-400">{err}</div>}
    </div>
  );
}