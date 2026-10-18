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

import { textIntoLines, textVsLines } from "./commands";

const item18 = () => ({
  layout: [{ id: 8, type: "form_field", bbox: [0, 0, 900, 140], reading_order: 0, quad: [], fields: [],
             text: "१८. काममा खटिन चाहेको सम्भावित समय:\nमहिना देखि\nमहिना सम्म:" }],
  lines: [
    { id: 1, text: "१८. काममा खटिन चाहेको सम्भावित समय:", bbox: [0, 0, 600, 40], block_id: 8, line_in_block: 0,
      is_heading: false, font_size: 20, font_id: null, quad: [] },
    { id: 2, text: "महिना देखि", bbox: [0, 50, 300, 90], block_id: 8, line_in_block: 1,
      is_heading: false, font_size: 20, font_id: null, quad: [] },
    { id: 3, text: "महिना सम्म", bbox: [0, 100, 300, 140], block_id: 8, line_in_block: 2,
      is_heading: false, font_size: 20, font_id: null, quad: [] },
  ],
  tables: [],
}) as any;

describe("text and lines that differ (gov_005 item १८)", () => {
  it("is reported, with both sides", () => {
    const diff = textVsLines(item18(), 8)!;
    expect(diff.text.endsWith("महिना सम्म:")).toBe(true);
    expect(diff.lines.endsWith("महिना सम्म")).toBe(true);
    expect(diff.sameRows).toBe(true);
  });

  it("'Copy text into lines' puts the colon on the line", () => {
    const d = item18();
    textIntoLines(d, 8);
    expect(d.lines[2].text).toBe("महिना सम्म:");
    expect(textVsLines(d, 8)).toBeNull();
  });

  it("'Use the lines' drops it from the text", () => {
    const d = item18();
    syncRegionTextFromLines(d, 8);
    expect(d.layout[0].text.endsWith("महिना सम्म")).toBe(true);
    expect(textVsLines(d, 8)).toBeNull();
  });
});

import { regionLines } from "./commands";

/** gov_005 table #3: the table text was corrected to "रोजगार", the line still says "रोजार" */
const table3 = () => {
  const cell = (id: number, row: number, col: number, text: string) => ({
    id, row, col, rowspan: 1, colspan: 1, text, is_header: row === 0, handwritten: false, empty: !text,
    bbox: [col * 300, row * 60, col * 300 + 300, row * 60 + 60], quad: [],
  });
  const ln = (id: number, text: string, cell_id: number, i: number, bbox: number[]) => ({
    id, text, bbox, block_id: 3, line_in_block: i, is_heading: false, font_size: 20, font_id: null,
    table_index: 0, cell_id, quad: [],
  });
  return {
    layout: [{ id: 3, type: "table", bbox: [0, 0, 600, 120], quad: [], reading_order: 0, table_index: 0,
               text: "सि.नं\nश्रम तथा रोजगार\nकार्यालयहरु" }],
    tables: [{ bbox: [0, 0, 600, 120], quad: [], num_rows: 2, num_cols: 2, header_rows: 1, ruling_style: "full",
               html: "", reading_order: [0, 1, 2, 3],
               cells: [cell(0, 0, 0, "सि.नं"), cell(1, 0, 1, "श्रम तथा रोजार\nकार्यालयहरु"), cell(2, 1, 0, ""), cell(3, 1, 1, "")] }],
    // line_in_block out of cell order on purpose: the table text follows the cell reading order
    lines: [ln(11, "श्रम तथा रोजार", 1, 0, [310, 5, 590, 28]), ln(12, "कार्यालयहरु", 1, 1, [310, 32, 590, 55]),
            ln(10, "सि.नं", 0, 2, [10, 5, 200, 28])],
  } as any;
};

describe("tables: text vs lines (gov_005 #3)", () => {
  it("the difference is shown for a table too, lines in cell order", () => {
    const d = table3();
    expect(regionLines(d, 3).map((l: any) => l.id)).toEqual([10, 11, 12]);
    const diff = textVsLines(d, 3)!;
    expect(diff.sameRows).toBe(true);
    expect(diff.lines).toBe("सि.नं\nश्रम तथा रोजार\nकार्यालयहरु");
  });

  it("'Copy text into lines' fixes the line and its cell", () => {
    const d = table3();
    textIntoLines(d, 3);
    expect(d.lines.find((l: any) => l.id === 11).text).toBe("श्रम तथा रोजगार");
    expect(d.tables[0].cells[1].text).toBe("श्रम तथा रोजगार\nकार्यालयहरु");
    expect(textVsLines(d, 3)).toBeNull();
  });
});
