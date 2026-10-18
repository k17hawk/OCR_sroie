// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { setElementText, tableText } from "./commands";

const q = (b: number[]) => [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];

/** doc_0000015-like table: 3 x 3 with numbers and words, one OCR line per cell */
function page() {
  const texts = [["सि.नं", "विवरण", "संख्या"], ["१", "दर्ता प्रकृयाम रहेको", "९ वटा"], ["२", "नविकरण", "४०२ जना"]];
  const cells: any[] = [];
  const lines: any[] = [];
  texts.forEach((row, r) => row.forEach((t, c) => {
    const b = [c * 300, r * 60, c * 300 + 300, r * 60 + 60];
    const id = cells.length;
    cells.push({ id, row: r, col: c, rowspan: 1, colspan: 1, text: t, is_header: r === 0, handwritten: false,
                 empty: false, bbox: b, quad: q(b) });
    lines.push({ id: 100 + id, text: t, bbox: [b[0] + 10, b[1] + 10, b[0] + 200, b[1] + 40], block_id: 5,
                 line_in_block: id, is_heading: false, font_size: 20, font_id: null, table_index: 0, cell_id: id,
                 quad: q([b[0] + 10, b[1] + 10, b[0] + 200, b[1] + 40]) });
  }));
  return {
    document_id: "d", next_line_id: 200, blocks: [],
    layout: [{ id: 5, type: "table", bbox: [0, 0, 900, 180], quad: q([0, 0, 900, 180]), reading_order: 0,
               table_index: 0, text: texts.flat().join("\n") }],
    tables: [{ bbox: [0, 0, 900, 180], quad: q([0, 0, 900, 180]), num_rows: 3, num_cols: 3, header_rows: 1,
               ruling_style: "full", theme: null, col_types: null, html: "", reading_order: cells.map((c) => c.id), cells }],
    lines, ruling_lines: [],
  } as any;
}

/** type `insert` at `at` in the box one key at a time, as the text box does */
function typeInto(d: any, at: string, insert: string) {
  const start = tableText(d, d.layout[0]);
  const pos = start.indexOf(at) + at.length;
  let cur = start;
  for (const ch of insert) {
    const p = pos + (cur.length - start.length);
    cur = cur.slice(0, p) + ch + cur.slice(p);
    setElementText(d, 5, cur);
  }
  return d;
}

const allText = (d: any) => d.tables[0].cells.map((c: any) => c.text).join("|");

describe("typing in the Table HTML box", () => {
  it("typing <br> one key at a time keeps every other cell", () => {
    const d = typeInto(page(), "सि.", "<br>");
    expect(d.tables[0].cells[0].text).toBe("सि.\nनं");
    expect(allText(d)).toBe("सि.\nनं|विवरण|संख्या|१|दर्ता प्रकृयाम रहेको|९ वटा|२|नविकरण|४०२ जना");
  });

  it("keeps the measured cell boxes and the OCR line boxes", () => {
    const d = typeInto(page(), "सि.", "<br>");
    expect(d.tables[0].cells[4].bbox).toEqual([300, 60, 600, 120]);
    const l = d.lines.find((x: any) => x.cell_id === 4);
    expect(l.bbox).toEqual([310, 70, 500, 100]);
  });

  it("pressing Enter inside a cell", () => {
    const d = typeInto(page(), "सि.", "\n");
    expect(allText(d)).toContain("विवरण|संख्या|१|दर्ता प्रकृयाम रहेको|९ वटा|२|नविकरण|४०२ जना");
  });
});

describe("old pages: a cell with its OCR line AND an automatic whole-cell line", () => {
  function dupPage() {
    const d = page();
    // cell 4 also got an automatic line covering the whole cell (older saves did this)
    d.lines.push({ id: 150, text: "दर्ता प्रकृयाम रहेको", bbox: [300, 60, 600, 120], block_id: 5, line_in_block: 9,
                   is_heading: false, font_size: 20, font_id: null, table_index: 0, cell_id: 4,
                   quad: q([300, 60, 600, 120]) });
    d.layout[0].text = d.lines.map((l: any) => l.text).join("\n");   // as the server built it: text twice
    return d;
  }

  it("after an HTML edit the table text is its lines again (one copy of each cell)", () => {
    const d = typeInto(dupPage(), "सि.", "<br>");
    const text = d.layout[0].text as string;
    expect(text.split("दर्ता प्रकृयाम रहेको").length - 1).toBe(1);
    expect(d.lines.filter((l: any) => l.cell_id === 4).length).toBe(1);
  });
});
