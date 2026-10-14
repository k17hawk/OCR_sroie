import type {
  BBox, Doc, ExtraRegion, FormFieldEntry, FormFieldLabel, LayoutElement,
  Line, NormalFieldValue, Quad, Table, TableCell,
} from "../types";
import type { Selection } from "../store";

function nextLayoutId(doc: Doc): number {
  return Math.max(-1, ...doc.layout.map((e) => e.id), ...doc.blocks.map((b) => b.block_id)) + 1;
}

export function quadFromBBox(b: BBox): Quad {
  const [x0, y0, x1, y1] = b;
  return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
}

export function emptyLayout(type: string, bbox: BBox, id: number): LayoutElement {
  const quad = quadFromBBox(bbox);
  const base = { id, bbox, reading_order: null, quad, text: "" };
  switch (type) {
    case "header":   return { ...base, type: "header" };
    case "footer":   return { ...base, type: "footer" };
    case "paragraph":return { ...base, type: "paragraph" };
    case "signature_caption": return { ...base, type: "signature_caption" };
    case "heading":  return { ...base, type: "heading", level: 2, text: "# " };
    case "figure":   return { ...base, type: "figure", subtype: "photo", text: "<figure/>" };
    case "form_field":
      return { ...base, type: "form_field", fields: [] };
    case "table":
      return { ...base, type: "table", table_index: -1 };
    case "list":
      return { ...base, type: "list", list_style: "bullet", items: [] };
    case "logo":      return { ...base, type: "logo" } as any;
    case "signature": return { ...base, type: "signature", signer: "other", text: "" } as any;
    case "stamp":     return { ...base, type: "stamp", stamp_text: "" };
    // any other type from /projects/:id/types (title, subheading,
    // handwriting, …) is a plain text region of that type
    default:          return { ...base, type: type as any } as any;
  }
}

export function addLayoutElement(d: Doc, type: string, bbox: BBox): number {
  const id = nextLayoutId(d);
  const el = emptyLayout(type, bbox, id);
  d.layout.push(el as LayoutElement);
  if (type === "table") {
    const el2 = el as any;
    el2.table_index = d.tables.length;
    d.tables.push(emptyTable(bbox));
  }
  return id;
}

export function emptyTable(bbox: BBox): Table {
  const quad = quadFromBBox(bbox);
  const [x0, y0, x1, y1] = bbox;
  const num_rows = 2, num_cols = 2;
  const cells: TableCell[] = [];
  for (let r = 0; r < num_rows; r++) {
    for (let c = 0; c < num_cols; c++) {
      const cx0 = x0 + ((x1 - x0) * c) / num_cols;
      const cx1 = x0 + ((x1 - x0) * (c + 1)) / num_cols;
      const cy0 = y0 + ((y1 - y0) * r) / num_rows;
      const cy1 = y0 + ((y1 - y0) * (r + 1)) / num_rows;
      const cb: BBox = [Math.floor(cx0), Math.floor(cy0), Math.ceil(cx1), Math.ceil(cy1)];
      cells.push({
        id: r * num_cols + c,
        row: r, col: c, rowspan: 1, colspan: 1,
        text: "", is_header: r === 0, handwritten: false, empty: true,
        bbox: cb,
        quad: quadFromBBox(cb),
      });
    }
  }
  return {
    bbox, quad, num_rows, num_cols, header_rows: 1,
    ruling_style: "full", theme: null, col_types: null,
    html: "", reading_order: cells.map((c) => c.id), cells,
  };
}

export function addLine(d: Doc, bbox: BBox, blockId: number | null): number {
  const quad = quadFromBBox(bbox);
  let parentId = blockId;
  if (parentId == null) {
    parentId = addLayoutElement(d, "paragraph", bbox);
  }
  const line: Line = {
    text: "",
    bbox, block_id: parentId,
    line_in_block: d.lines.filter((l) => l.block_id === parentId).length,
    is_heading: false,
    font_size: Math.max(12, bbox[3] - bbox[1]),
    font_id: null,
    quad,
  };
  d.lines.push(line);
  return d.lines.length - 1;
}

export function addFormField(d: Doc, labelBox: BBox, valueBox: BBox | null): number {
  const id = addLayoutElement(d, "form_field", labelBox);
  const el = d.layout.find((e) => e.id === id) as any;
  const label: FormFieldLabel = {
    text: "Label:",
    bbox: labelBox, quad: quadFromBBox(labelBox),
  };
  let value: FormFieldEntry["value"] = null;
  if (valueBox) {
    value = {
      text: "",
      bbox: valueBox,
      quad: quadFromBBox(valueBox),
      handwritten: false,
    } as NormalFieldValue;
  }
  el.fields.push({ label, field_type: "text", value });
  return id;
}

export function setReadingOrder(d: Doc, ids: number[]): void {
  const locked = new Set(
    d.layout.filter((e) => e.type === "logo" || e.type === "signature" || e.type === "stamp").map((e) => e.id),
  );
  let n = 0;
  for (const id of ids) {
    const el = d.layout.find((e) => e.id === id);
    if (!el || locked.has(id)) continue;
    el.reading_order = n++;
  }
  for (const el of d.layout) {
    if (!locked.has(el.id) && !ids.includes(el.id)) el.reading_order = null;
  }
}

export function mirrorCheckboxExtra(d: Doc): void {
  // Rebuild the checkboxes that mirror form-field options, but KEEP
  // standalone checkboxes (drawn / converted from a box via the Type
  // dropdown). Before, every checkbox region was deleted here.
  const optionBoxes = new Set<string>();
  for (const el of d.layout) {
    if (el.type !== "form_field") continue;
    for (const f of el.fields) {
      if (!f.value || !("options" in f.value)) continue;
      for (const opt of f.value.options) {
        const q = opt.box_quad;
        optionBoxes.add([
          Math.floor(Math.min(q[0][0], q[2][0])), Math.floor(Math.min(q[0][1], q[2][1])),
          Math.ceil(Math.max(q[0][0], q[2][0])), Math.ceil(Math.max(q[0][1], q[2][1])),
        ].join(","));
      }
    }
  }
  d.extra_regions = d.extra_regions.filter(
    (er) => er.type !== "checkbox" || !optionBoxes.has(er.bbox.join(",")),
  );
  for (const el of d.layout) {
    if (el.type !== "form_field") continue;
    for (const f of el.fields) {
      if (!f.value || !("options" in f.value)) continue;
      for (const opt of f.value.options) {
        d.extra_regions.push({
          type: "checkbox",
          checked: opt.checked,
          option: opt.text,
          bbox: [
            Math.floor(Math.min(opt.box_quad[0][0], opt.box_quad[2][0])),
            Math.floor(Math.min(opt.box_quad[0][1], opt.box_quad[2][1])),
            Math.ceil(Math.max(opt.box_quad[0][0], opt.box_quad[2][0])),
            Math.ceil(Math.max(opt.box_quad[0][1], opt.box_quad[2][1])),
          ],
          quad: opt.box_quad,
        } as ExtraRegion);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// setElementText
// ---------------------------------------------------------------------------

/**
 * Set a layout element's text AND keep its underlying lines in sync.
 *
 * Layout element types whose text is derived from lines (header, paragraph,
 * footer, signature_caption, heading) will have their lines created, updated,
 * or trimmed to match the new text. Element types whose text is not derived
 * (figure, form_field, table, list, stamp) only have ``el.text`` updated.
 */
export function setElementText(d: Doc, elementId: number, newText: string): void {
  const el = d.layout.find((e) => e.id === elementId) as any;
  if (!el) return;

  el.text = newText;

  // A list's text and its items must stay in sync: the server rebuilds the
  // text from `items` on every save, so text typed here without updating
  // the items was thrown away.
  if (el.type === "list") {
    el.items = listItemsFromText(newText);
    return;
  }

  const LINE_BASED = new Set([
    "header", "title", "heading", "subheading", "paragraph", "text_line",
    "footer", "page_number", "signature_caption", "table_heading",
    "table_footer", "list_item", "form_label", "form_value", "label",
    "handwriting", "handwritten_annotation", "handwritten_correction",
    "handwritten_number", "mixed_text",
  ]);
  if (!LINE_BASED.has(el.type)) return;

  let displayLines = newText.split("\n");

  if (el.type === "heading") {
    const prefix = "#".repeat(el.level ?? 1) + " ";
    if (displayLines[0]?.startsWith(prefix)) {
      displayLines[0] = displayLines[0].slice(prefix.length);
    }
    displayLines = [displayLines[0] ?? ""];
  }

  const existing = d.lines
    .map((ln, i) => ({ ln, i }))
    .filter(({ ln }) => ln.block_id === elementId)
    .sort((a, b) => a.ln.line_in_block - b.ln.line_in_block);

  const [x0, y0, x1, y1] = el.bbox;
  const n = Math.max(1, displayLines.length);
  const h = Math.max(1, Math.floor((y1 - y0) / n));

  for (let i = 0; i < Math.min(existing.length, n); i++) {
    const { ln } = existing[i];
    ln.text = displayLines[i];
    ln.line_in_block = i;
    ln.is_heading = el.type === "heading";
    ln.bbox = [x0, y0 + i * h, x1, y0 + (i + 1) * h];
    ln.quad = [
      [ln.bbox[0], ln.bbox[1]],
      [ln.bbox[2], ln.bbox[1]],
      [ln.bbox[2], ln.bbox[3]],
      [ln.bbox[0], ln.bbox[3]],
    ];
    ln.font_size = Math.max(8, h * 0.7);
  }

  for (let i = existing.length; i < n; i++) {
    const bbox: BBox = [x0, y0 + i * h, x1, y0 + (i + 1) * h];
    d.lines.push({
      text: displayLines[i],
      bbox,
      block_id: elementId,
      line_in_block: i,
      is_heading: el.type === "heading",
      font_size: Math.max(8, h * 0.7),
      font_id: null,
      quad: [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
      ],
    });
  }

  if (existing.length > n) {
    const toRemove = new Set(existing.slice(n).map(({ ln }) => ln));
    d.lines = d.lines.filter((ln) => !toRemove.has(ln));
  }
}

// ---------------------------------------------------------------------------
// translate / resize
// ---------------------------------------------------------------------------

function moveObject(
  o: { bbox: BBox; quad?: Quad },
  dx: number,
  dy: number,
): void {
  o.bbox = [
    o.bbox[0] + dx,
    o.bbox[1] + dy,
    o.bbox[2] + dx,
    o.bbox[3] + dy,
  ];
  if (o.quad) {
    o.quad = o.quad.map((p) => [p[0] + dx, p[1] + dy]) as Quad;
  }
}

function moveQuad(
  o: Record<string, unknown>,
  key: string,
  dx: number,
  dy: number,
): void {
  const q = o[key] as Quad | undefined;
  if (!q) return;
  o[key] = q.map((p) => [p[0] + dx, p[1] + dy]) as Quad;
}

/**
 * Move a layout element and translate every child that belongs to it:
 * lines, form-field label/value bboxes, table cells, ruling lines.
 */
export function translateLayoutElement(
  d: Doc,
  id: number,
  dx: number,
  dy: number,
): void {
  if (dx === 0 && dy === 0) return;
  const el = d.layout.find((e) => e.id === id);
  if (!el) return;

  moveObject(el, dx, dy);

  for (const ln of d.lines) {
    if (ln.block_id === id) moveObject(ln, dx, dy);
  }

  if (el.type === "form_field") {
    for (const f of el.fields) {
      moveObject(f.label, dx, dy);
      if (f.value && "bbox" in f.value) moveObject(f.value, dx, dy);
      if (f.value && "options" in f.value) {
        for (const opt of f.value.options) {
          moveQuad(opt as unknown as Record<string, unknown>, "box_quad", dx, dy);
          moveQuad(opt as unknown as Record<string, unknown>, "text_quad", dx, dy);
        }
      }
    }
  }

  if (el.type === "table") {
    const t = d.tables[(el as any).table_index];
    if (t) {
      for (const c of t.cells) moveObject(c, dx, dy);
      moveObject(t, dx, dy);
      for (const rl of d.ruling_lines) {
        if (rl.table_index === (el as any).table_index) {
          rl.p0 = [rl.p0[0] + dx, rl.p0[1] + dy];
          rl.p1 = [rl.p1[0] + dx, rl.p1[1] + dy];
          rl.bbox = [
            rl.bbox[0] + dx, rl.bbox[1] + dy,
            rl.bbox[2] + dx, rl.bbox[3] + dy,
          ];
        }
      }
    }
  }
}

/**
 * Resize a layout element by mapping its current bbox to `newBbox`. Every
 * descendant is transformed by the same affine (uniform scale per axis).
 */
export function resizeLayoutElement(
  d: Doc,
  id: number,
  newBbox: BBox,
): void {
  const el = d.layout.find((e) => e.id === id);
  if (!el) return;
  const [ox0, oy0, ox1, oy1] = el.bbox;
  const [nx0, ny0, nx1, ny1] = newBbox;

  const oldW = Math.max(1, ox1 - ox0);
  const oldH = Math.max(1, oy1 - oy0);
  const sx = (nx1 - nx0) / oldW;
  const sy = (ny1 - ny0) / oldH;

  const mapPoint = (p: [number, number]): [number, number] => [
    nx0 + (p[0] - ox0) * sx,
    ny0 + (p[1] - oy0) * sy,
  ];
  const mapBox = (o: { bbox: BBox; quad?: Quad }) => {
    const [bx0, by0, bx1, by1] = o.bbox;
    o.bbox = [
      Math.round(nx0 + (bx0 - ox0) * sx),
      Math.round(ny0 + (by0 - oy0) * sy),
      Math.round(nx0 + (bx1 - ox0) * sx),
      Math.round(ny0 + (by1 - oy0) * sy),
    ];
    if (o.quad) {
      o.quad = o.quad.map(mapPoint) as Quad;
    }
  };

  const oldQuad = el.quad;
  el.bbox = [Math.round(nx0), Math.round(ny0), Math.round(nx1), Math.round(ny1)];
  if (oldQuad) el.quad = oldQuad.map(mapPoint) as Quad;

  for (const ln of d.lines) if (ln.block_id === id) mapBox(ln);

  if (el.type === "form_field") {
    for (const f of el.fields) {
      mapBox(f.label);
      if (f.value && "bbox" in f.value) mapBox(f.value);
      if (f.value && "options" in f.value) {
        for (const opt of f.value.options) {
          opt.box_quad = opt.box_quad.map(mapPoint) as Quad;
          opt.text_quad = opt.text_quad.map(mapPoint) as Quad;
        }
      }
    }
  }

  if (el.type === "table") {
    const t = d.tables[(el as any).table_index];
    if (t) {
      mapBox(t);
      for (const c of t.cells) mapBox(c);
      for (const rl of d.ruling_lines) {
        if (rl.table_index === (el as any).table_index) {
          const p0 = mapPoint(rl.p0 as [number, number]);
          const p1 = mapPoint(rl.p1 as [number, number]);
          rl.p0 = p0;
          rl.p1 = p1;
          rl.bbox = [p0[0], p0[1], p1[0], p1[1]];
        }
      }
    }
  }
}
// ---------------------------------------------------------------------------
// selection helpers
// ---------------------------------------------------------------------------

/** Which layout region does a selection belong to? (layout -> itself) */
export function parentLayoutOf(doc: Doc, sel: Selection): number | null {
  if (!sel) return null;
  switch (sel.kind) {
    case "layout":
      return sel.id;
    case "field":
      return sel.elementId;
    case "line": {
      const ln = doc.lines[sel.index];
      if (!ln || ln.block_id == null) return null;
      return doc.layout.some((e) => e.id === ln.block_id) ? ln.block_id : null;
    }
    case "cell": {
      const el = doc.layout.find(
        (e) => e.type === "table" && (e as any).table_index === sel.tableIndex,
      );
      return el ? el.id : null;
    }
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// reading order (auto-increment)
// ---------------------------------------------------------------------------

const RO_LOCKED = new Set(["logo", "signature", "stamp"]);

export function isReadingOrderLocked(type: string): boolean {
  return RO_LOCKED.has(type);
}

/** Renumber 0..n-1 keeping the current relative order (closes gaps). */
export function compactReadingOrder(d: Doc): void {
  const ordered = d.layout
    .filter((e) => e.reading_order != null && !RO_LOCKED.has(e.type))
    .sort((a, b) => (a.reading_order as number) - (b.reading_order as number));
  ordered.forEach((e, i) => { e.reading_order = i; });
}

/** Next number to hand out = one past the current highest. */
export function nextReadingOrder(d: Doc): number {
  let max = -1;
  for (const e of d.layout) {
    if (e.reading_order != null && !RO_LOCKED.has(e.type)) {
      max = Math.max(max, e.reading_order);
    }
  }
  return max + 1;
}

/**
 * Give `id` the next reading-order number (moving it to the end if it
 * already had one), then close any gaps. Returns the number assigned,
 * or null for locked types (logo / signature / stamp).
 */
export function appendReadingOrder(d: Doc, id: number): number | null {
  const el = d.layout.find((e) => e.id === id);
  if (!el || RO_LOCKED.has(el.type)) return null;
  el.reading_order = null;
  compactReadingOrder(d);
  const n = nextReadingOrder(d);
  el.reading_order = n;
  return n;
}

export function removeFromReadingOrder(d: Doc, id: number): void {
  const el = d.layout.find((e) => e.id === id);
  if (!el) return;
  el.reading_order = null;
  compactReadingOrder(d);
}

export function clearReadingOrder(d: Doc): void {
  for (const e of d.layout) e.reading_order = null;
}

/**
 * Number every unlocked region top-to-bottom, left-to-right. Boxes whose
 * vertical centres are within half a typical line height count as the
 * same row, so side-by-side columns read left before right.
 */
export function autoReadingOrder(d: Doc): void {
  const els = d.layout.filter((e) => !RO_LOCKED.has(e.type));
  const heights = els.map((e) => e.bbox[3] - e.bbox[1]).sort((a, b) => a - b);
  const median = heights.length ? heights[Math.floor(heights.length / 2)] : 20;
  const tol = Math.max(4, median / 2);
  const cy = (e: LayoutElement) => (e.bbox[1] + e.bbox[3]) / 2;

  const sorted = [...els].sort((a, b) => cy(a) - cy(b));
  const rows: LayoutElement[][] = [];
  for (const e of sorted) {
    const row = rows[rows.length - 1];
    if (row && Math.abs(cy(e) - cy(row[0])) <= tol) row.push(e);
    else rows.push([e]);
  }
  let n = 0;
  for (const row of rows) {
    row.sort((a, b) => a.bbox[0] - b.bbox[0]);
    for (const e of row) e.reading_order = n++;
  }
  for (const e of d.layout) if (RO_LOCKED.has(e.type)) e.reading_order = null;
}

// ---------------------------------------------------------------------------
// list text <-> items
// ---------------------------------------------------------------------------

export interface ListItemData { marker: string; level: number; text: string }

const LIST_MARKER_RE =
  /^(\s*)((?:[०-९]+|[0-9]+)[.)]|\(?[क-ह]\)|\([०-९0-9]+\)|[a-zA-Z][.)]|(?:[ivxlcdm]+|[IVXLCDM]+)[.)]|[-•*●▪◦–])\s+(.*)$/;

/** "१. a\n  - b" -> [{marker:"१.",level:0,text:"a"},{marker:"•",level:1,text:"b"}] */
export function listItemsFromText(text: string): ListItemData[] {
  const items: ListItemData[] = [];
  for (const raw of text.replace(/\r\n?/g, "\n").split("\n")) {
    if (!raw.trim()) continue;
    const m = raw.match(LIST_MARKER_RE);
    if (m) {
      const marker = m[2] === "-" || m[2] === "*" || m[2] === "●" ? "•" : m[2];
      items.push({ marker, level: Math.floor(m[1].replace(/\t/g, "  ").length / 2), text: m[3].trim() });
    } else if (items.length) {
      items[items.length - 1].text += " " + raw.trim(); // wrapped line
    } else {
      items.push({ marker: "", level: 0, text: raw.trim() });
    }
  }
  return items;
}

/** Same format the server's serialize_list produces. */
export function listTextFromItems(items: ListItemData[]): string {
  return items
    .map((it) => "  ".repeat(Math.max(0, it.level || 0)) +
      (it.marker ? (it.marker === "•" ? "-" : it.marker) + " " : "") + (it.text ?? ""))
    .join("\n");
}

/** Call after changing a list's items so its text matches. */
export function syncListText(el: any): void {
  if (el && el.type === "list" && Array.isArray(el.items)) el.text = listTextFromItems(el.items);
}
