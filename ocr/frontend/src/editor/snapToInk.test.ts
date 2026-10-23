import { afterEach, describe, expect, it, vi } from "vitest";
import { snapChildToInk } from "./snapToInk";
import { useStore } from "../store";

const q = (b: number[]) => [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
const drawn: [number, number, number, number] = [100, 100, 900, 300];

function load() {
  const doc: any = {
    document_id: "D1", page_number: 1, page_count: 1, doc_type: "other",
    layout: [{ id: 1, type: "paragraph", bbox: [0, 0, 1000, 1000], reading_order: 0, text: "x", quad: q([0, 0, 1000, 1000]) }],
    lines: [{ id: 5, text: "x", bbox: [...drawn], block_id: 1, line_in_block: 0, is_heading: false,
              font_size: 18, font_id: null, quad: q(drawn) }],
    tables: [], extra_regions: [],
  };
  useStore.setState({ doc, past: [], future: [] } as any);
}

function answer(body: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => body, text: async () => "" })));
}

afterEach(() => vi.unstubAllGlobals());

describe("snap to ink follows tilted text", () => {
  it("a tilted answer becomes the line's quad, with its outer bbox", async () => {
    load();
    const quad = [[120, 160], [880, 120], [884, 200], [124, 240]];
    answer({ bbox: [120, 120, 884, 240], changed: true, quad });
    await snapChildToInk({ get: (d) => d.lines[0] as any }, drawn, false);
    const ln = useStore.getState().doc!.lines[0];
    expect(ln.quad).toEqual(quad);
    expect(ln.bbox).toEqual([120, 120, 884, 240]);
  });

  it("a straight answer keeps a rectangle", async () => {
    load();
    answer({ bbox: [110, 150, 890, 210], changed: true, quad: null });
    await snapChildToInk({ get: (d) => d.lines[0] as any }, drawn, false);
    expect(useStore.getState().doc!.lines[0].quad).toEqual(q([110, 150, 890, 210]));
  });

  it("Alt skips the snap", async () => {
    load();
    answer({ bbox: [110, 150, 890, 210], changed: true, quad: null });
    await snapChildToInk({ get: (d) => d.lines[0] as any }, drawn, true);
    expect(useStore.getState().doc!.lines[0].bbox).toEqual(drawn);
  });
});
