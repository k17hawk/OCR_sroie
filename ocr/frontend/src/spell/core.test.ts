import { describe, expect, it } from "vitest";
import aff from "../../public/dict/ne_NP.aff?raw";
import dic from "../../public/dict/ne_NP.dic?raw";
import { makeChecker, tokenize, tokenAt } from "./core";

const checker = makeChecker(aff, dic);

describe("tokenize", () => {
  it("splits Devanagari words, skipping digits, dandas and Latin", () => {
    const t = tokenize("नाम: राम बहादुर, वडा नं. ५ । abc १२३");
    expect(t.map((x) => x.word)).toEqual(["नाम", "राम", "बहादुर", "वडा", "नं"]);
    expect("नाम: राम".slice(t[1].start, t[1].end)).toBe("राम");
  });
  it("finds the word at the caret", () => {
    const t = tokenize("राम बहादुर");
    expect(tokenAt(t, 6)?.word).toBe("बहादुर");
    expect(tokenAt(t, 3)?.word).toBe("राम"); // end of a word counts
  });
});

describe("Nepali checker", () => {
  it("accepts common form words", () => {
    for (const w of ["नेपाल", "सरकार", "नागरिकता", "जिल्ला", "गाउँपालिका", "वडा", "परिवारका"]) {
      expect(checker.correct(w), w).toBe(true);
    }
  });
  it("flags ि/ी and स/श mix-ups and suggests the fix", () => {
    expect(checker.correct("राष्ट्रीय")).toBe(false);
    expect(checker.suggest("राष्ट्रीय")).toContain("राष्ट्रिय");
    expect(checker.correct("सदश्य")).toBe(false);
    expect(checker.suggest("सदश्य")).toContain("सदस्य");
  });
  it("learns added words", () => {
    expect(checker.correct("कखगघङ")).toBe(false);
    checker.add("कखगघङ");
    expect(checker.correct("कखगघङ")).toBe(true);
  });
});
