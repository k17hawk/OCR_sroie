import { describe, expect, it } from "vitest";

import { syncRegionTextFromLines } from "./commands";

const line = (id: number, text: string, i: number, extra: object = {}) => ({
  id, text, bbox: [0, i * 40, 900, i * 40 + 30], block_id: 12, line_in_block: i,
  is_heading: false, font_size: 20, font_id: null, quad: [], ...extra,
});

const page = (lines: any[]) => ({
  layout: [{ id: 12, type: "table_heading", bbox: [0, 0, 900, 80], reading_order: 0, quad: [],
             text: "(अ) वर्ग / समूह: (… कुनै एकमा √ लगाउनुहोस)\n4" }],
  lines, tables: [],
}) as any;

describe("region text follows its lines in the editor", () => {
  it("deleting the stray '4' line removes it from the text (gov_005 #12)", () => {
    const d = page([line(1, "(अ) वर्ग / समूह: (… कुनै एकमा √ लगाउनुहोस)", 0), line(2, "4", 1)]);
    d.lines.splice(1, 1);                       // the Delete line button
    syncRegionTextFromLines(d, 12);
    expect(d.layout[0].text).toBe("(अ) वर्ग / समूह: (… कुनै एकमा √ लगाउनुहोस)");
  });

  it("editing a line's text updates the region text, in line order", () => {
    const d = page([line(2, "दिन", 1), line(1, "संख्या", 0)]);
    d.lines[0].text = "दिन:";
    syncRegionTextFromLines(d, 12);
    expect(d.layout[0].text).toBe("संख्या\nदिन:");
  });

  it("a region left with no lines keeps its text; table lines are ignored", () => {
    const d = page([line(9, "x", 0, { table_index: 0, cell_id: 3 })]);
    syncRegionTextFromLines(d, 12);
    expect(d.layout[0].text).toContain("\n4");
  });
});
