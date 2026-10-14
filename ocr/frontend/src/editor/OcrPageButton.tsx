import { useState } from "react";
import { ocrPage } from "../api";
import { useStore } from "../store";

export function OcrPageButton() {
  const doc = useStore((s) => s.doc);
  const edit = useStore((s) => s.edit);
  const family = useStore((s) => s.ocrFamily);
  const device = useStore((s) => s.ocrDevice);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  async function run() {
    if (!doc) return;
    setBusy(true);
    setErr(null);
    setInfo(null);
    try {
      // Server runs OCR for every line in the doc (or every layout element
      // if there are no lines), and returns the fresh text keyed by id.
      const r = await ocrPage(doc.document_id, {
        family,
        device,
        lang_hint: doc.language,
      });
      edit((d) => {
        for (const ln of r.lines) {
          const target = d.lines[ln.index];
          if (target) target.text = ln.text;
        }
        for (const el of r.layout) {
          const target = d.layout.find((x) => x.id === el.id) as any;
          if (target && "text" in target) target.text = el.text;
        }
      });
      setInfo(`${r.model} · ${r.lines.length} lines`);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col items-end gap-0.5">
      <button
        onClick={run}
        disabled={busy || !doc}
        className={
          "rounded px-2 py-1 text-[11px] font-medium " +
          (busy || !doc
            ? "bg-slate-800 text-slate-500"
            : "bg-cyan-700 text-white hover:bg-cyan-600")
        }
        title="Re-run OCR on every region of this page"
      >
        {busy ? "OCR…" : "Re-OCR page"}
      </button>

      {info && (
        <div className="max-w-[200px] truncate text-[10px] text-emerald-400" title={info}>
          {info}
        </div>
      )}
      {err && (
        <div className="max-w-[200px] break-words text-right text-[10px] text-rose-400" title={err}>
          {err}
        </div>
      )}
    </div>
  );
}