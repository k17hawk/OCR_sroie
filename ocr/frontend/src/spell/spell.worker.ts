/// <reference lib="webworker" />
/**
 * Runs the Nepali checker off the main thread: building it from the
 * dictionary takes ~1 s, which would freeze typing. After that a word is
 * checked in well under a millisecond and suggested in ~5–20 ms.
 */
import { makeChecker, type Checker } from "./core";

export type SpellRequest =
  | { type: "init"; base: string; userWords: string[] }
  | { type: "check"; id: number; words: string[] }
  | { type: "suggest"; id: number; word: string }
  | { type: "add"; word: string };

export type SpellResponse =
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "checked"; id: number; wrong: string[]; words: string[] }
  | { type: "suggested"; id: number; word: string; suggestions: string[] };

let checker: Checker | null = null;
let pending: SpellRequest[] = [];

const post = (m: SpellResponse) => (self as unknown as Worker).postMessage(m);

function handle(m: SpellRequest) {
  if (!checker) { pending.push(m); return; }
  if (m.type === "check") {
    post({ type: "checked", id: m.id, words: m.words, wrong: m.words.filter((w) => !checker!.correct(w)) });
  } else if (m.type === "suggest") {
    post({ type: "suggested", id: m.id, word: m.word, suggestions: checker.suggest(m.word) });
  } else if (m.type === "add") {
    checker.add(m.word);
  }
}

self.onmessage = async (e: MessageEvent<SpellRequest>) => {
  const m = e.data;
  if (m.type !== "init") { handle(m); return; }
  try {
    const [aff, dic] = await Promise.all(
      ["ne_NP.aff", "ne_NP.dic"].map(async (f) => {
        const r = await fetch(m.base + f);
        if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
        return r.text();
      }),
    );
    checker = makeChecker(aff, dic, m.userWords);
    post({ type: "ready" });
    const queued = pending;
    pending = [];
    queued.forEach(handle);
  } catch (err) {
    post({ type: "error", message: String((err as Error)?.message ?? err) });
  }
};
