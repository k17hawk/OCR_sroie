import { useState } from "react";
import { ocrRegion } from "../api";
import { useOcrPrefs, type OcrLang } from "../hooks/useOcrPrefs";
import { ModelPicker } from "./ModelPicker";

type Lang = OcrLang;

const LANGS: { id: Lang; label: string }[] = [
  { id: "ne",    label: "नेपाली (Nepali)" },
  { id: "en",    label: "English" },
  { id: "mixed", label: "Mixed ne+en" },
  { id: "hi",    label: "हिन्दी (Hindi)" },
];

interface Props {
  documentId: string;
  quad: [number, number][];
  onText: (text: string) => void;
  label?: string;
}

const CONF_FALLBACK = 0.75;

export function OcrRegionButton({ documentId, quad, onText, label = "Re-OCR region" }: Props) {
  // shared + remembered: survives selecting another box and page reloads
  const {
    family, device, lang, autoFallback,
    setFamily, setDevice, setLang, setAutoFallback,
  } = useOcrPrefs();

  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState<string | null>(null);
  const [err, setErr]   = useState<string | null>(null);

  async function run() {
    setBusy(true); setErr(null); setInfo(null);
    try {
      const primary = await ocrRegion(documentId, { quad, family, device, lang_hint: lang });
      let best = primary;

      // Chain a couple of alternatives if the primary pass looks unsure.
      // This is what makes bilingual forms "just work".
      if (autoFallback && primary.confidence < CONF_FALLBACK) {
        const alts: Lang[] = (["mixed", "en", "ne"] as Lang[]).filter((l) => l !== lang);
        for (const alt of alts) {
          try {
            const r = await ocrRegion(documentId, { quad, family, device, lang_hint: alt });
            if (r.confidence > best.confidence) best = r;
            if (best.confidence >= 0.9) break;
          } catch { /* keep going */ }
        }
      }

      if (best.text) onText(best.text);
      setInfo(`${best.model} · ${best.device} · conf ${best.confidence.toFixed(2)} · lang ${lang}`);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-1">
      <ModelPicker family={family} device={device} onFamily={setFamily} onDevice={setDevice} compact />

      <div className="flex gap-1">
        <select
          value={lang}
          onChange={(e) => setLang(e.target.value as Lang)}
          className="flex-1 rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-200"
          title="Language hint passed to the OCR engine"
        >
          {LANGS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
        </select>
        <label className="flex items-center gap-1 rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-300"
               title="Retry with alternative language hints when confidence is low">
          <input type="checkbox" checked={autoFallback} onChange={(e) => setAutoFallback(e.target.checked)} />
          auto
        </label>
      </div>

      <button
        onClick={run}
        disabled={busy}
        className={
          "w-full rounded py-1 text-[11px] font-medium " +
          (busy ? "bg-slate-800 text-slate-500" : "bg-violet-700 text-white hover:bg-violet-600")
        }
      >
        {busy ? "Running…" : label}
      </button>

      {info && <div className="text-[10px] text-emerald-400">{info}</div>}
      {err  && <div className="break-words text-[10px] text-rose-400">{err}</div>}
    </div>
  );
}