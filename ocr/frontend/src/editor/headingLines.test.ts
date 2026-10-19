import { describe, expect, it } from "vitest";
import { setElementText, textVsLines, syncRegionTextFromLines } from "./commands";

function page(): any {
  return {
    layout: [{ id: 1, type: "heading", level: 2, bbox: [0, 0, 400, 120], quad: [[0, 0], [400, 0], [400, 120], [0, 120]], text: "" }],
    lines: [], tables: [], extra_regions: [],
  };
}
const rows = ["श्रम, रोजगार तथा सामाजिक सुरक्षा मन्त्रालय", "श्रम तथा व्यवसायजन्य सुरक्षा विभाग", "मिनभवन, काठमाण्डौ"];
const own = (d: any) => d.lines.filter((l: any) => l.block_id === 1).sort((a: any, b: any) => a.line_in_block - b.line_in_block);

describe("multi-line heading", () => {
  it("keeps every typed row as a line", () => {
    const d = page();
    setElementText(d, 1, "## " + rows.join("\n"));
    expect(own(d).map((l: any) => l.text)).toEqual(rows);
    expect(d.layout[0].text).toBe("## " + rows.join("\n"));
  });
  it("keeps a just-typed empty row (Enter) in the lines", () => {
    const d = page();
    setElementText(d, 1, "## " + rows[0] + "\n");
    expect(own(d)).toHaveLength(2);
    expect(d.layout[0].text).toBe("## " + rows[0]);
  });
  it("does not report the ## marker as a text / lines difference", () => {
    const d = page();
    setElementText(d, 1, "## " + rows.join("\n"));
    expect(textVsLines(d, 1)).toBeNull();
    syncRegionTextFromLines(d, 1);
    expect(d.layout[0].text).toBe("## " + rows.join("\n"));
    expect(textVsLines(d, 1)).toBeNull();
  });
  it("still reports a real difference", () => {
    const d = page();
    setElementText(d, 1, "## " + rows.join("\n"));
    d.layout[0].text = "## " + rows.slice(0, 2).join("\n");
    expect(textVsLines(d, 1)).not.toBeNull();
  });
});
