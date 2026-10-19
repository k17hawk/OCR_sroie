/**
 * Nepali spell checking: word splitting and the checker itself. Pure code,
 * shared by the Web Worker (spell.worker.ts) and the tests.
 *
 * Dictionary: Hunspell ne_NP (Madan Puraskar Pustakalaya, LGPL 2.1), the one
 * LibreOffice ships. Served from /dict/ne_NP.{aff,dic}.
 */
import nspell from "nspell";

export interface Token {
  word: string;
  start: number;
  end: number;
}

// A Devanagari word: letters, vowel signs, virama, nukta, candrabindu …
// plus ZWJ / ZWNJ (half forms). Not digits (०-९, U+0966-096F), not the
// dandas । ॥ (U+0964-0965), not the abbreviation sign ॰ (U+0970).
const WORD = /[ऀ-ॣॱ-ॿ‌‍]+/g;

/** Devanagari words of a text with their offsets. One-letter words are skipped. */
export function tokenize(text: string): Token[] {
  const out: Token[] = [];
  for (const m of text.matchAll(WORD)) {
    const word = m[0].replace(/^[‌‍]+|[‌‍]+$/g, "");
    if (Array.from(word).length < 2) continue;
    const lead = m[0].indexOf(word);
    const start = (m.index ?? 0) + lead;
    out.push({ word, start, end: start + word.length });
  }
  return out;
}

/** The token that holds (or touches) a caret position, if any. */
export function tokenAt(tokens: Token[], pos: number): Token | undefined {
  return tokens.find((t) => pos >= t.start && pos <= t.end);
}

/**
 * Words the 2006 dictionary is missing but every Nepali form uses
 * (federal-era local government terms, common loanwords).
 */
export const EXTRA_WORDS = [
  "गाउँपालिका", "नगरपालिका", "उपमहानगरपालिका", "महानगरपालिका", "वडा",
  "नम्बर", "प्रदेश", "मोबाइल", "फोन", "इमेल", "फारम", "फाराम", "कार्ड",
  "प्रमाणपत्र", "बैंक", "खाता", "ठेगाना", "हस्ताक्षर", "औंठाछाप", "ल्याप्चे",
];

export interface Checker {
  correct(word: string): boolean;
  suggest(word: string): string[];
  add(word: string): void;
}

export function makeChecker(aff: string, dic: string, extra: string[] = []): Checker {
  const sp = nspell(aff, dic);
  for (const w of [...EXTRA_WORDS, ...extra]) sp.add(w.normalize("NFC"));
  return {
    correct: (w) => sp.correct(w.normalize("NFC")),
    suggest: (w) => sp.suggest(w.normalize("NFC")).slice(0, 6),
    add: (w) => { sp.add(w.normalize("NFC")); },
  };
}
