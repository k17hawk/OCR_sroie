/**
 * Nepali spell checking for the text boxes: on/off switch, the worker, a
 * cache of checked words, "Ignore" (this browser session) and "Add to
 * dictionary" (kept in this browser).
 */
import { create } from "zustand";
import type { SpellRequest, SpellResponse } from "./spell.worker";

const ENABLED_KEY = "devalabel.spell.enabled";
const USER_WORDS_KEY = "devalabel.spell.userWords";
const IGNORED_KEY = "devalabel.spell.ignored";

function readJson<T>(store: () => Storage, key: string, def: T): T {
  try {
    const v = JSON.parse(store().getItem(key) ?? "null");
    return v ?? def;
  } catch {
    return def;
  }
}
function writeJson(store: () => Storage, key: string, v: unknown) {
  try { store().setItem(key, JSON.stringify(v)); } catch { /* storage blocked */ }
}
const local = () => localStorage;
const session = () => sessionStorage;

type Status = "off" | "loading" | "ready" | "error";

interface SpellState {
  enabled: boolean;
  status: Status;
  error: string | null;
  /** bumps whenever new check results arrive, so underlines re-render */
  version: number;
  ignored: Set<string>;
  userWords: string[];
  setEnabled: (on: boolean) => void;
  ignore: (word: string) => void;
  addWord: (word: string) => void;
  removeWord: (word: string) => void;
}

/** word → true if misspelled. Filled by the worker. */
const results = new Map<string, boolean>();
const queued = new Set<string>();
let worker: Worker | null = null;
let nextId = 1;
const suggestWaiters = new Map<number, (s: string[]) => void>();
const suggestCache = new Map<string, string[]>();

function send(m: SpellRequest) { worker?.postMessage(m); }

function startWorker() {
  if (worker) return;
  useSpell.setState({ status: "loading", error: null });
  worker = new Worker(new URL("./spell.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent<SpellResponse>) => {
    const m = e.data;
    if (m.type === "ready") useSpell.setState({ status: "ready" });
    else if (m.type === "error") useSpell.setState({ status: "error", error: m.message });
    else if (m.type === "checked") {
      const wrong = new Set(m.wrong);
      for (const w of m.words) results.set(w, wrong.has(w));
      useSpell.setState((s) => ({ version: s.version + 1 }));
    } else if (m.type === "suggested") {
      suggestCache.set(m.word, m.suggestions);
      suggestWaiters.get(m.id)?.(m.suggestions);
      suggestWaiters.delete(m.id);
    }
  };
  worker.onerror = (e) => useSpell.setState({ status: "error", error: e.message || "spell checker failed to start" });
  send({ type: "init", base: `${import.meta.env.BASE_URL}dict/`, userWords: useSpell.getState().userWords });
  // anything asked for before the switch was on
  flush();
}

let flushScheduled = false;
function flush() {
  flushScheduled = false;
  if (!worker || !queued.size) return;
  send({ type: "check", id: nextId++, words: [...queued] });
  queued.clear();
}

export const useSpell = create<SpellState>((set, get) => ({
  enabled: readJson(local, ENABLED_KEY, false),
  status: "off",
  error: null,
  version: 0,
  ignored: new Set(readJson<string[]>(session, IGNORED_KEY, [])),
  userWords: readJson<string[]>(local, USER_WORDS_KEY, []),

  setEnabled: (on) => {
    writeJson(local, ENABLED_KEY, on);
    set({ enabled: on });
    if (on) startWorker();
  },
  ignore: (word) => {
    const ignored = new Set(get().ignored).add(word);
    writeJson(session, IGNORED_KEY, [...ignored]);
    set({ ignored });
  },
  addWord: (word) => {
    if (get().userWords.includes(word)) return;
    const userWords = [...get().userWords, word];
    writeJson(local, USER_WORDS_KEY, userWords);
    results.set(word, false);
    send({ type: "add", word });
    set((s) => ({ userWords, version: s.version + 1 }));
  },
  removeWord: (word) => {
    const userWords = get().userWords.filter((w) => w !== word);
    writeJson(local, USER_WORDS_KEY, userWords);
    // the worker can't forget a word; it is checked again after a reload
    set({ userWords });
  },
}));

// switched on in an earlier visit: load the dictionary right away
if (useSpell.getState().enabled) queueMicrotask(startWorker);

/**
 * true = misspelled, false = fine, undefined = not checked yet (it is
 * queued; components re-render when the answer arrives via `version`).
 */
export function isWrong(word: string): boolean | undefined {
  const st = useSpell.getState();
  if (st.ignored.has(word) || st.userWords.includes(word)) return false;
  const r = results.get(word);
  if (r !== undefined) return r;
  if (!queued.has(word)) {
    queued.add(word);
    if (!flushScheduled) { flushScheduled = true; setTimeout(flush, 0); }
  }
  return undefined;
}

export function suggest(word: string): Promise<string[]> {
  const hit = suggestCache.get(word);
  if (hit) return Promise.resolve(hit);
  if (!worker) return Promise.resolve([]);
  const id = nextId++;
  return new Promise((resolve) => {
    suggestWaiters.set(id, resolve);
    send({ type: "suggest", id, word });
  });
}
