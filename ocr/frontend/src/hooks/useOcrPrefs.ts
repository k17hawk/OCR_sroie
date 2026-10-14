import { useEffect, useSyncExternalStore } from "react";
import { useStore } from "../store";

/**
 * One shared, remembered set of OCR settings.
 *
 * Before, every OCR panel kept its own copy of model / device / language
 * in component state, so it reset to the default every time you selected
 * another box. Now:
 *   - model + device live in the main store (same values the toolbar and
 *     Auto-annotate use) and are saved to localStorage
 *   - language + "auto" fallback live here and are saved too
 * Change it once and it sticks, across selections and page reloads.
 */

export type OcrLang = "ne" | "en" | "mixed" | "hi";

interface Saved {
  family?: string;
  device?: string;
  lang?: OcrLang;
  autoFallback?: boolean;
}

const KEY = "devalabel.ocrPrefs.v1";

function load(): Saved {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Saved) : {};
  } catch {
    return {};
  }
}

function save(s: Saved) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch { /* private mode etc. — settings just won't persist */ }
}

// tiny shared store for lang / autoFallback
let saved: Saved = load();
const listeners = new Set<() => void>();
function update(patch: Partial<Saved>) {
  saved = { ...saved, ...patch };
  save(saved);
  listeners.forEach((l) => l());
}
function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}
const getSnapshot = () => saved;

let restored = false;

export function useOcrPrefs() {
  const family = useStore((s) => s.ocrFamily);
  const device = useStore((s) => s.ocrDevice);
  const setOcrFamily = useStore((s) => s.setOcrFamily);
  const setOcrDevice = useStore((s) => s.setOcrDevice);
  const docLang = useStore((s) => s.doc?.language) as OcrLang | undefined;
  const snap = useSyncExternalStore(subscribe, getSnapshot);

  // Once per page load: put the remembered model/device back into the store.
  useEffect(() => {
    if (restored) return;
    restored = true;
    if (saved.family && saved.family !== family) setOcrFamily(saved.family);
    if (saved.device && saved.device !== device) setOcrDevice(saved.device);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    family,
    device,
    // remembered language wins; otherwise follow the document's language
    lang: (snap.lang ?? docLang ?? "ne") as OcrLang,
    autoFallback: snap.autoFallback ?? true,
    setFamily: (f: string) => { setOcrFamily(f); update({ family: f }); },
    setDevice: (d: string) => { setOcrDevice(d); update({ device: d }); },
    setLang: (l: OcrLang) => update({ lang: l }),
    setAutoFallback: (v: boolean) => update({ autoFallback: v }),
  };
}
