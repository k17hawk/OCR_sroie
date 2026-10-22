// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RightPanel } from "./Right";
import { useStore } from "../store";

(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;

const q = (b: number[]) => [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
const line = (id: number, text: string, block: number, item: number, b: number[]) => ({
  id, text, bbox: b, block_id: block, line_in_block: item, is_heading: false,
  font_size: 18, font_id: null, quad: q(b), item_index: item, list_level: 0,
});

describe("tree: list and sublist items are selectable", () => {
  it("clicking a sublist item selects its line; clicking an attached sublist selects it", async () => {
    const doc: any = {
      document_id: "D1", page_number: 1, page_count: 1, doc_type: "other",
      layout: [
        { id: 1, type: "list", bbox: [0, 0, 500, 300], reading_order: 0, text: "", list_style: "bullet",
          quad: q([0, 0, 500, 300]), items: [{ marker: "•", level: 0, text: "parent", children: [2] }] },
        { id: 2, type: "sublist", bbox: [40, 40, 500, 200], reading_order: null, parent_id: 1, parent_item: 0,
          text: "", list_style: "bullet", quad: q([40, 40, 500, 200]),
          items: [{ marker: "•", level: 0, text: "child one" }, { marker: "•", level: 0, text: "child two" }] },
      ],
      lines: [
        line(10, "• parent", 1, 0, [0, 0, 500, 30]),
        line(11, "• child one", 2, 0, [40, 40, 500, 70]),
        line(12, "• child two", 2, 1, [40, 80, 500, 110]),
      ],
      tables: [], extra_regions: [],
    };
    useStore.setState({ doc, rightTab: "tree", selection: null } as any);
    const host = document.createElement("div");
    document.body.appendChild(host);
    await act(async () => {
      createRoot(host).render(
        <QueryClientProvider client={new QueryClient()}><RightPanel /></QueryClientProvider>,
      );
    });
    const btn = (text: string) =>
      [...host.querySelectorAll("button")].find((b) => b.textContent?.trim() === text)!;

    await act(async () => { btn("• child two").click(); });
    expect(useStore.getState().selection).toEqual({ kind: "line", index: 2 });

    await act(async () => { btn("• parent").click(); });
    expect(useStore.getState().selection).toEqual({ kind: "line", index: 0 });

    const sub = [...host.querySelectorAll("button")].filter((b) => b.textContent?.includes("sublist"));
    await act(async () => { sub[sub.length - 1].click(); });
    expect(useStore.getState().selection).toEqual({ kind: "layout", id: 2 });
  });
});
