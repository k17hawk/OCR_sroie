import { describe as suite, expect, it, vi } from "vitest";

vi.mock("../store", () => ({ useStore: { getState: () => ({}) } }));
const { describe } = await import("./save");

suite("save error text", () => {
  it("a text-loss block shows what is missing, the text and the lines", () => {
    const body = JSON.stringify({ detail: [{
      loc: [], where: "layout[8] (#8 form_field)", type: "text_loss",
      msg: "printed text would be lost on save — add lines for it (Detect lines)",
      got: ":", region_id: 8, before: "गर्भवती:\nमहिना", after: "गर्भवती\nमहिना",
    }] });
    const out = describe(body);
    expect(out).toContain('layout[8] (#8 form_field): printed text would be lost — ":" is in the text but in none of its lines.');
    expect(out).toContain("text : गर्भवती: ⏎ महिना");
    expect(out).toContain("lines: गर्भवती ⏎ महिना");
  });

  it("other errors keep the old format", () => {
    const body = JSON.stringify({ detail: [{ where: "layout[0].type", msg: "bad", got: "x" }] });
    expect(describe(body)).toBe('layout[0].type: bad (got "x")');
  });
});
