import { describe, expect, it } from "vitest";

import { addLine, afterListChange } from "./commands";

const ln = (id: number, text: string, bbox: number[], extra: object = {}) => ({
  id, text, bbox, block_id: 3, line_in_block: 0, is_heading: false, font_size: 20,
  font_id: null, quad: [], ...extra,
});

/** a list with two items, each already measured as one line "क) नयाँ" / "ख) नविकरण" */
const page = () => ({
  layout: [{ id: 3, type: "list", bbox: [100, 100, 900, 220], reading_order: 0, quad: [],
             text: "क) नयाँ\nख) नविकरण", list_style: "dev",
             items: [{ marker: "क)", level: 0, text: "नयाँ" }, { marker: "ख)", level: 0, text: "नविकरण" }] }],
  blocks: [], tables: [], next_line_id: 20,
  lines: [ln(10, "क) नयाँ", [110, 110, 400, 150], { list_level: 0, item_index: 0 }),
          ln(11, "ख) नविकरण", [110, 170, 450, 210], { list_level: 0, item_index: 1, line_in_block: 1 })],
}) as any;

describe("lines drawn inside a list", () => {
  it("a new line beside an item joins that item", () => {
    const d = page();
    const i = addLine(d, [110, 172, 150, 208], 3);            // the "ख)" marker
    expect(d.lines[i].item_index).toBe(1);
    expect(d.lines[i].list_level).toBe(0);
  });

  it("marker and words drawn as separate lines survive a list edit", () => {
    const d = page();
    // split item 0 into a marker line and a words line
    d.lines[0].text = "नयाँ";
    d.lines[0].bbox = [160, 110, 400, 150];
    const m = addLine(d, [110, 112, 150, 148], 3);
    d.lines[m].text = "क)";
    afterListChange(d, 3);
    const mine = d.lines.filter((l: any) => l.block_id === 3).map((l: any) => [l.text, l.item_index]);
    expect(mine).toEqual([["क)", 0], ["नयाँ", 0], ["ख) नविकरण", 1]]);
  });

  it("a line that belongs to no item is kept, never dropped", () => {
    const d = page();
    const n = addLine(d, [110, 300, 600, 340], 3);              // a note below the items
    d.lines[n].text = "(कुनै एकमा √ लगाउनुहोस्)";
    expect(d.lines[n].item_index).toBeUndefined();
    afterListChange(d, 3);
    expect(d.lines.some((l: any) => l.text === "(कुनै एकमा √ लगाउनुहोस्)")).toBe(true);
    const order = d.lines.filter((l: any) => l.block_id === 3).map((l: any) => l.line_in_block);
    expect(order).toEqual([0, 1, 2]);
  });
});
