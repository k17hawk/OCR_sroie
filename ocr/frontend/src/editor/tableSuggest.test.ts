import { describe, expect, it } from "vitest";

import { applyTableSuggestion } from "./commands";

const q = (b: number[]) => [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]] as [number, number][];

// suggested: 2 x 3, header cell spanning 3 columns on row 0
const suggestion = {
  num_rows: 2, num_cols: 3,
  region_bbox: [100, 100, 400, 200] as [number, number, number, number],
  region_quad: q([100, 100, 400, 200]),
  cells: [
    { id: 0, row: 0, col: 0, rowspan: 1, colspan: 3, bbox: [100, 100, 400, 150] as any, quad: q([100, 100, 400, 150]) },
    { id: 1, row: 1, col: 0, rowspan: 1, colspan: 1, bbox: [100, 150, 200, 200] as any, quad: q([100, 150, 200, 200]) },
    { id: 2, row: 1, col: 1, rowspan: 1, colspan: 1, bbox: [200, 150, 300, 200] as any, quad: q([200, 150, 300, 200]) },
    { id: 3, row: 1, col: 2, rowspan: 1, colspan: 1, bbox: [300, 150, 400, 200] as any, quad: q([300, 150, 400, 200]) },
  ],
};

const page = (cells: any[] = []) => ({
  document_id: "d", layout: [{ id: 5, type: "table", bbox: [90, 90, 410, 215], quad: q([90, 90, 410, 215]),
                                text: "", table_index: 0, reading_order: 0 }],
  tables: [{ bbox: [90, 90, 410, 215], quad: q([90, 90, 410, 215]), num_rows: 0, num_cols: 0, header_rows: 0,
             ruling_style: "full", theme: null, col_types: null, html: "", reading_order: [], cells }],
  lines: [], ruling_lines: [], blocks: [], next_line_id: 0,
}) as any;

describe("apply a suggested table layout", () => {
  it("an empty table gets the grid, merged cells and the snapped region", () => {
    const d = page();
    applyTableSuggestion(d, 5, suggestion);
    const t = d.tables[0];
    expect([t.num_rows, t.num_cols, t.cells.length]).toEqual([2, 3, 4]);
    expect(t.cells[0]).toMatchObject({ row: 0, col: 0, colspan: 3, text: "", empty: true });
    expect(d.layout[0].bbox).toEqual([100, 100, 400, 200]);
    expect(d.ruling_lines.length).toBeGreaterThan(0);
    expect(d.ruling_lines.every((r: any) => r.table_index === 0)).toBe(true);
  });

  it("text already typed is kept in the cell it falls in, header flag too", () => {
    const d = page([
      { id: 0, row: 0, col: 0, rowspan: 1, colspan: 1, text: "परिवारका सदस्य", is_header: true, handwritten: false,
        empty: false, bbox: [100, 100, 250, 150], quad: q([100, 100, 250, 150]) },
      { id: 1, row: 1, col: 0, rowspan: 1, colspan: 1, text: "पुरुष", is_header: false, handwritten: false,
        empty: false, bbox: [100, 150, 200, 200], quad: q([100, 150, 200, 200]) },
    ]);
    applyTableSuggestion(d, 5, suggestion);
    const t = d.tables[0];
    expect(t.cells[0]).toMatchObject({ text: "परिवारका सदस्य", is_header: true, empty: false });
    expect(t.cells[1]).toMatchObject({ text: "पुरुष", is_header: false });
    expect(t.header_rows).toBe(1);
    // a line per non-empty cell, tied to its cell
    const ls = d.lines.filter((l: any) => l.block_id === 5);
    expect(ls.map((l: any) => [l.cell_id, l.text])).toEqual([[0, "परिवारका सदस्य"], [1, "पुरुष"]]);
  });

  it("does nothing for a region that is not a table or an empty suggestion", () => {
    const d = page();
    applyTableSuggestion(d, 5, { ...suggestion, cells: [] });
    expect(d.tables[0].cells).toEqual([]);
  });
});
