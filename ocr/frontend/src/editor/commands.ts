import type {
  BBox, Doc, ExtraRegion, FormFieldEntry, FormFieldLabel, LayoutElement,
  Line, NormalFieldValue, Quad, Table,
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

/**
 * A new table is one box with no cells. It used to start as a placeholder
 * 2×2 grid, which split every newly marked table into four boxes. Cells
 * come from the text box (one row per line, cells split by |) or from
 * "Table OCR".
 */
export function emptyTable(bbox: BBox): Table {
  return {
    bbox, quad: quadFromBBox(bbox), num_rows: 0, num_cols: 0, header_rows: 0,
    ruling_style: "full", theme: null, col_types: null,
    html: "", reading_order: [], cells: [],
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
// ---------------------------------------------------------------------------
// Text for types whose text is derived (form fields, tables)
//
// Form field: one line per field, "label: value".
// Table:      the table's HTML (<tr>, <td>/<th>, rowspan, colspan);
//             rows of cells separated by " | " also work.
// ---------------------------------------------------------------------------

function rowBox(b: BBox, i: number, n: number): BBox {
  const h = Math.max(1, (b[3] - b[1]) / Math.max(1, n));
  return [b[0], Math.round(b[1] + i * h), b[2], Math.round(b[1] + (i + 1) * h)];
}

/** What the generic text box shows for a form field. */
export function formFieldText(el: any): string {
  return (el.fields ?? [])
    .map((f: FormFieldEntry) => {
      const label = f.label?.text ?? "";
      const v = f.value as any;
      if (!v || "options" in v) return label;
      const sep = label.trimEnd().endsWith(":") ? " " : ": ";
      return `${label}${sep}${v.text ?? ""}`;
    })
    .join("\n");
}

function setFormFieldText(el: any, text: string): void {
  if (!Array.isArray(el.fields)) el.fields = [];
  const before = formFieldText(el).split("\n");
  const rows = text.split("\n");
  // a trailing empty line is just the cursor on a new line, not a field
  while (rows.length > 1 && rows[rows.length - 1].trim() === "") rows.pop();
  const n = rows.length;

  rows.forEach((row, i) => {
    // a line that wasn't changed leaves its field exactly as it was
    if (el.fields[i] && row === before[i]) return;

    // the label is what is typed up to and including the first ":"
    const colon = row.indexOf(":");
    const labelText = colon >= 0 ? row.slice(0, colon + 1) : row;
    const valueText = colon >= 0 ? row.slice(colon + 1).trim() : null;

    let f = el.fields[i] as FormFieldEntry | undefined;
    if (!f) {
      const b = rowBox(el.bbox, i, n);
      const split = Math.round(b[0] + (b[2] - b[0]) * 0.4);
      const lb: BBox = [b[0], b[1], split, b[3]];
      f = { label: { text: "", bbox: lb, quad: quadFromBBox(lb) }, field_type: "text", value: null };
      el.fields.push(f);
    }
    f.label.text = labelText.trim();

    const v = f.value as any;
    if (v && "options" in v) return; // checkbox: options are edited below
    if (valueText === null) {
      if (v) v.text = "";
    } else if (v) {
      v.text = valueText;
    } else {
      const lb = f.label.bbox;
      const vb: BBox = [lb[2], lb[1], Math.max(lb[2] + 4, el.bbox[2]), lb[3]];
      f.value = { text: valueText, bbox: vb, quad: quadFromBBox(vb), handwritten: false };
    }
  });

  if (el.fields.length > n) el.fields.splice(n);
}

const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/**
 * The table's HTML, as the text box shows it: same rules as the server's
 * serialize_table_html (<th> for headers, colspan/rowspan only when > 1,
 * cells covered by a span skipped), one <tr> per line for readability.
 */
export function tableText(d: Doc, el: any): string {
  const t = d.tables[el.table_index];
  if (!t || !t.cells.length) return "";
  const maxRow = Math.max(...t.cells.map((c) => c.row + c.rowspan));
  const maxCol = Math.max(...t.cells.map((c) => c.col + c.colspan));
  const at = new Map(t.cells.map((c) => [`${c.row},${c.col}`, c]));
  const covered = new Set<string>();
  for (const c of t.cells) {
    for (let r = c.row; r < c.row + c.rowspan; r++) {
      for (let k = c.col; k < c.col + c.colspan; k++) {
        if (r !== c.row || k !== c.col) covered.add(`${r},${k}`);
      }
    }
  }
  const rows: string[] = [];
  for (let r = 0; r < maxRow; r++) {
    let tr = "<tr>";
    for (let k = 0; k < maxCol; k++) {
      if (covered.has(`${r},${k}`)) continue;
      const c = at.get(`${r},${k}`);
      if (!c) continue;
      const tag = c.is_header ? "th" : "td";
      const span =
        (c.colspan > 1 ? ` colspan="${c.colspan}"` : "") +
        (c.rowspan > 1 ? ` rowspan="${c.rowspan}"` : "") +
        (c.handwritten ? " data-handwritten" : "");
      tr += `<${tag}${span}>${escapeHtml(c.text ?? "")}</${tag}>`;
    }
    rows.push(tr + "</tr>");
  }
  return ["<table>", ...rows, "</table>"].join("\n");
}

/**
 * Rebuild a table's cells from HTML. Returns false (table untouched) when
 * the HTML has no <tr> yet, e.g. while it is still being typed.
 * Cell boxes: kept from an existing cell at the same row/column/span while
 * the grid keeps its shape,
 * otherwise an even grid over the table box.
 */
function setTableFromHtml(t: any, tableBox: BBox, html: string): boolean {
  const src = /<table[\s>]/i.test(html) ? html : `<table>${html}</table>`;
  const dom = new DOMParser().parseFromString(src, "text/html");
  const trs = Array.from(dom.querySelectorAll("tr"));
  if (trs.length === 0) return false;

  type Parsed = { row: number; col: number; rowspan: number; colspan: number; text: string; header: boolean; hw: boolean };
  const parsed: Parsed[] = [];
  const taken = new Set<string>();
  trs.forEach((tr, r) => {
    let col = 0;
    for (const cellEl of Array.from(tr.children)) {
      const tag = cellEl.tagName.toLowerCase();
      if (tag !== "td" && tag !== "th") continue;
      while (taken.has(`${r},${col}`)) col++;
      const rowspan = Math.max(1, parseInt(cellEl.getAttribute("rowspan") ?? "1", 10) || 1);
      const colspan = Math.max(1, parseInt(cellEl.getAttribute("colspan") ?? "1", 10) || 1);
      for (let i = r; i < r + rowspan; i++) {
        for (let k = col; k < col + colspan; k++) taken.add(`${i},${k}`);
      }
      const text = (cellEl.textContent ?? "").replace(/\s+/g, " ").trim();
      const hw = cellEl.hasAttribute("data-handwritten") || cellEl.classList.contains("handwritten");
      parsed.push({ row: r, col, rowspan, colspan, text, header: tag === "th", hw });
      col += colspan;
    }
  });
  if (parsed.length === 0) return false;

  const nr = Math.max(...parsed.map((p) => p.row + p.rowspan));
  const nc = Math.max(...parsed.map((p) => p.col + p.colspan));
  const [x0, y0, x1, y1] = tableBox;
  const cw = (x1 - x0) / nc, rh = (y1 - y0) / nr;
  // Reuse an old cell's box only while the grid keeps its shape (e.g. you
  // fixed a cell's text). If rows/columns changed, old boxes would overlap
  // the new ones, so every cell gets a fresh even grid.
  const sameShape = t.num_rows === nr && t.num_cols === nc;
  const old = new Map<string, any>(
    sameShape
      ? (t.cells ?? []).map((c: any) => [`${c.row},${c.col},${c.rowspan},${c.colspan}`, c])
      : [],
  );

  t.cells = parsed.map((p, id) => {
    const prev = old.get(`${p.row},${p.col},${p.rowspan},${p.colspan}`);
    const b: BBox = prev?.bbox ?? [
      Math.round(x0 + p.col * cw), Math.round(y0 + p.row * rh),
      Math.round(x0 + (p.col + p.colspan) * cw), Math.round(y0 + (p.row + p.rowspan) * rh),
    ];
    return {
      id, row: p.row, col: p.col, rowspan: p.rowspan, colspan: p.colspan,
      text: p.text, is_header: p.header,
      handwritten: p.hw, empty: p.text === "",
      bbox: b, quad: prev?.quad ?? quadFromBBox(b),
    };
  });
  t.num_rows = nr;
  t.num_cols = nc;
  // leading rows whose cells are all <th>
  let hr = 0;
  while (hr < nr && parsed.filter((p) => p.row === hr).every((p) => p.header)) hr++;
  t.header_rows = hr;
  t.reading_order = t.cells.map((c: any) => c.id);
  return true;
}

// ---------------------------------------------------------------------------
// Keep a table in the reference shape (see a finished page such as
// doc_0000096): one OCR line per non-empty cell, linked by
// block_id = the table's layout id, table_index and cell_id, with the
// cell's text; and ruling lines for the grid.
//
// Line boxes: an existing OCR line inside the cell keeps its tight text box
// (and quad / font size); otherwise the line takes the cell's box.
// ---------------------------------------------------------------------------

const center = (b: BBox): [number, number] => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
const inBox = (p: [number, number], b: BBox) =>
  p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
/** Replace the lines matched by `mine` with `fresh`, at the position of the
 *  first of them, so the order of `lines` stays as it was. */
function replaceLines(d: Doc, mine: (ln: any) => boolean, fresh: any[]): void {
  const at = d.lines.findIndex(mine);
  const rest = d.lines.filter((ln) => !mine(ln));
  const pos = at < 0 ? rest.length : d.lines.slice(0, at).filter((ln) => !mine(ln)).length;
  d.lines = [...rest.slice(0, pos), ...fresh, ...rest.slice(pos)];
}

const unionBox = (bs: BBox[]): BBox => [
  Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])),
  Math.max(...bs.map((b) => b[2])), Math.max(...bs.map((b) => b[3])),
];

/** Ruling lines from the cell grid, honouring the table's ruling_style. */
function rulingLinesFor(t: Table, ti: number): any[] {
  if (!t.cells.length || t.ruling_style === "none") return [];
  const cells = t.cells;
  const nr = Math.max(...cells.map((c) => c.row + c.rowspan));
  const nc = Math.max(...cells.map((c) => c.col + c.colspan));
  const x0 = Math.min(...cells.map((c) => c.bbox[0])), x1 = Math.max(...cells.map((c) => c.bbox[2]));
  const y0 = Math.min(...cells.map((c) => c.bbox[1])), y1 = Math.max(...cells.map((c) => c.bbox[3]));
  const avg = (v: number[], dflt: number) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : dflt);

  const rowEdge = (r: number) => {
    if (r === 0) return y0;
    if (r === nr) return y1;
    const above = cells.filter((c) => c.row + c.rowspan === r).map((c) => c.bbox[3]);
    const below = cells.filter((c) => c.row === r).map((c) => c.bbox[1]);
    return Math.round(avg([...above, ...below], y0));
  };
  const colEdge = (k: number) => {
    if (k === 0) return x0;
    if (k === nc) return x1;
    const left = cells.filter((c) => c.col + c.colspan === k).map((c) => c.bbox[2]);
    const right = cells.filter((c) => c.col === k).map((c) => c.bbox[0]);
    return Math.round(avg([...left, ...right], x0));
  };

  const out: any[] = [];
  for (let r = 0; r <= nr; r++) {
    const y = rowEdge(r);
    out.push({ table_index: ti, orientation: "h", width: 1, p0: [x0, y], p1: [x1, y], bbox: [x0, y, x1, y + 1] });
  }
  if (t.ruling_style !== "horizontal") {
    for (let k = 0; k <= nc; k++) {
      const x = colEdge(k);
      out.push({ table_index: ti, orientation: "v", width: 1, p0: [x, y0], p1: [x, y1], bbox: [x, y0, x + 1, y1] });
    }
  }
  return out;
}

/**
 * Bring one table's lines (and, if asked, ruling lines) in line with its
 * cells. Call after the cells change: Table OCR, HTML / text edit, cell edit.
 * `ruling`: regenerate ruling lines from the cell grid. Pass false for a
 * text-only change so hand-made or imported ruling lines are kept; they are
 * still created if the table has none.
 */
export function syncTable(d: Doc, elementId: number, opts: { ruling?: boolean } = {}): void {
  const el = d.layout.find((e) => e.id === elementId) as any;
  if (!el || el.type !== "table") return;
  const ti = el.table_index;
  const t = d.tables[ti];
  if (!t) return;

  // lines that belong to this table: its own, plus orphans inside it
  const ids = new Set(d.layout.map((e) => e.id));
  const mine = (ln: any) =>
    ln.block_id === elementId ||
    (!ids.has(ln.block_id) && inBox(center(ln.bbox), el.bbox));
  const old = d.lines.filter(mine);

  const order = t.reading_order?.length ? t.reading_order : t.cells.map((c) => c.id);
  const byId = new Map(t.cells.map((c) => [c.id, c]));
  const fresh: Line[] = [];
  for (const cid of order) {
    const c = byId.get(cid);
    if (!c || !(c.text ?? "").trim()) continue;
    const inside = old.filter((ln) => inBox(center(ln.bbox), c.bbox));
    const one = inside.length === 1 ? inside[0] : null;
    const bbox: BBox = inside.length ? unionBox(inside.map((ln) => ln.bbox)) : c.bbox;
    const quad = one?.quad ?? (inside.length ? quadFromBBox(bbox) : c.quad);
    const line: any = {
      text: c.text,
      bbox,
      block_id: elementId,
      line_in_block: fresh.length,
      is_heading: false,
      font_size: inside[0]?.font_size ?? Math.max(8, (bbox[3] - bbox[1]) * 0.7),
      font_id: inside[0]?.font_id ?? null,
      table_index: ti,
      cell_id: c.id,
      quad,
    };
    if (c.handwritten) line.handwritten = true;
    fresh.push(line);
  }
  replaceLines(d, mine, fresh);

  const hasRuling = d.ruling_lines.some((rl) => rl.table_index === ti);
  if (opts.ruling || !hasRuling) {
    d.ruling_lines = [
      ...d.ruling_lines.filter((rl) => rl.table_index !== ti),
      ...rulingLinesFor(t, ti),
    ];
  }
}

function setTableText(d: Doc, el: any, text: string): void {
  if (typeof el.table_index !== "number" || el.table_index < 0 || !d.tables[el.table_index]) {
    el.table_index = d.tables.length;
    d.tables.push(emptyTable(el.bbox));
  }
  const t = d.tables[el.table_index];
  const shape = () => `${t.num_rows}x${t.num_cols}:${t.cells.length}`;
  const before = shape();
  // lines and ruling lines follow the cells; the grid is redrawn only when
  // rows / columns changed, so a text fix keeps the existing ruling lines
  const done = () => syncTable(d, el.id, { ruling: shape() !== before });

  // cleared box: no cells
  if (text.trim() === "") {
    t.cells = [];
    t.num_rows = 0; t.num_cols = 0; t.header_rows = 0; t.reading_order = [];
    done();
    return;
  }
  // HTML: the HTML decides rows, columns, spans and headers
  if (text.trim().startsWith("<")) {
    if (setTableFromHtml(t, el.bbox, text)) done();
    return;
  }

  // otherwise: one row per line, cells separated by | (or a tab).
  // Turned into HTML so it follows exactly the same rules.
  const rows = text.split("\n").filter((r, i, a) => i < a.length - 1 || r.trim() !== "");
  const html = rows
    .map((r) => "<tr>" + r.split(/\s*\|\s*|\t/).map((c) => `<td>${escapeHtml(c.trim())}</td>`).join("") + "</tr>")
    .join("");
  if (setTableFromHtml(t, el.bbox, `<table>${html}</table>`)) done();
}

export function setElementText(d: Doc, elementId: number, newText: string): void {
  const el = d.layout.find((e) => e.id === elementId) as any;
  if (!el) return;

  // These types don't keep their own text on save: the server rebuilds it
  // from fields / cells, or has no text field at all. Text typed into the
  // generic text box (or filled by "OCR region") has to go into the data the
  // server keeps, or it is thrown away on the next save.
  if (el.type === "form_field") { setFormFieldText(el, newText); return; }
  if (el.type === "table") { setTableText(d, el, newText); return; }
  if (el.type === "stamp") { el.stamp_text = newText; return; }
  if (el.type === "figure" || el.type === "signature" || el.type === "logo") return;

  el.text = newText;

  // A list's text and its items must stay in sync: the server rebuilds the
  // text from `items` on every save, so text typed here without updating
  // the items was thrown away.
  if (el.type === "list") {
    const known = (el.items ?? []).flatMap((it: ListItemData) =>
      (it.children ?? []).map((id) => ({ id, text: childTextOf(d, id) })));
    el.items = listItemsFromText(newText, known);
    afterListChange(d, el.id);
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

  // the text box shows one row per line; the region's own text joins them
  // with a space, as in the reference pages (the server does the same)
  if (el.type !== "heading") {
    el.text = newText.split("\n").map((t) => t.trim()).filter(Boolean).join(" ");
  }

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

  // Same number of lines: only the text changes, so OCR line boxes stay
  // where they are. A different number of lines has no real geometry, so
  // the lines are laid out as even strips over the region.
  const relayout = existing.length !== n;
  for (let i = 0; i < Math.min(existing.length, n); i++) {
    const { ln } = existing[i];
    ln.text = displayLines[i];
    ln.line_in_block = i;
    ln.is_heading = el.type === "heading";
    if (!relayout) continue;
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

export interface ListItemData {
  marker: string;
  level: number;
  text: string;
  /** ids of boxes that belong to this item (a note, a figure, …) */
  children?: number[];
}

const LIST_MARKER_RE =
  /^(\s*)((?:[०-९]+|[0-9]+)[.)]|\(?[क-ह]\)|\([०-९0-9]+\)|[a-zA-Z][.)]|(?:[ivxlcdm]+|[IVXLCDM]+)[.)]|[-•*●▪◦–])\s+(.*)$/;

/**
 * "१. a\n  - b" -> [{marker:"१.",level:0,text:"a"},{marker:"•",level:1,text:"b"}]
 *
 * `children`: boxes that already belong to items of this list, with their
 * text. A marker-less line equal to one of them is that child (it stays
 * attached to the item above it) instead of being glued onto the item text.
 */
export function listItemsFromText(
  text: string,
  children: { id: number; text: string }[] = [],
): ListItemData[] {
  const items: ListItemData[] = [];
  const unused = [...children];
  for (const raw of text.replace(/\r\n?/g, "\n").split("\n")) {
    if (!raw.trim()) continue;
    const m = raw.match(LIST_MARKER_RE);
    const ci = m ? -1 : unused.findIndex((c) => c.text.trim() === raw.trim());
    if (m) {
      const marker = m[2] === "-" || m[2] === "*" || m[2] === "●" ? "•" : m[2];
      items.push({ marker, level: Math.floor(m[1].replace(/\t/g, "  ").length / 2), text: m[3].trim() });
    } else if (ci >= 0 && items.length) {
      const last = items[items.length - 1];
      (last.children ??= []).push(unused[ci].id);
      unused.splice(ci, 1);
    } else if (items.length) {
      items[items.length - 1].text += " " + raw.trim(); // wrapped line
    } else {
      items.push({ marker: "", level: 0, text: raw.trim() });
    }
  }
  return items;
}

/** A child box's text as it appears in its list's text. */
export function childTextOf(d: Doc | null | undefined, id: number): string {
  const c = d?.layout.find((e) => e.id === id) as any;
  if (!c) return "";
  if (c.type === "figure") return "<figure/>";
  return c.text ?? "";
}

/**
 * Same format the server's serialize_list produces: each item on its own
 * line, 2 spaces per level, and each child's text under its item one level
 * deeper.
 */
export function listTextFromItems(items: ListItemData[], d?: Doc | null): string {
  const out: string[] = [];
  for (const it of items) {
    const indent = "  ".repeat(Math.max(0, it.level || 0));
    out.push(indent + (it.marker ? (it.marker === "•" ? "-" : it.marker) + " " : "") + (it.text ?? ""));
    for (const cid of it.children ?? []) {
      const t = childTextOf(d, cid);
      if (t) for (const part of t.split("\n")) out.push(indent + "  " + part);
    }
  }
  return out.join("\n");
}

/** Call after changing a list's items so its text matches. */
export function syncListText(el: any, d?: Doc | null): void {
  if (el && el.type === "list" && Array.isArray(el.items)) el.text = listTextFromItems(el.items, d);
}

/**
 * Keep the boxes inside a list in step with its items' `children`:
 * each child gets parent_id = the list, parent_item = the item's index and
 * no reading order of its own (it is read as part of the item). Ids that no
 * longer exist are dropped; boxes no item points to lose their parent.
 */
export function linkListChildren(d: Doc, listId: number): void {
  const el = d.layout.find((e) => e.id === listId) as any;
  if (!el || el.type !== "list") return;
  const ids = new Set(d.layout.map((e) => e.id));
  const owned = new Map<number, number>();
  el.items.forEach((it: ListItemData, i: number) => {
    if (!it.children) return;
    it.children = it.children.filter((c) => ids.has(c) && c !== listId && !owned.has(c));
    it.children.forEach((c) => owned.set(c, i));
    if (!it.children.length) delete it.children;
  });
  for (const e of d.layout as any[]) {
    if (owned.has(e.id)) {
      e.parent_id = listId;
      e.parent_item = owned.get(e.id);
      e.reading_order = null;
    } else if (e.parent_id === listId) {
      delete e.parent_id;
      delete e.parent_item;
    }
  }
}

/** Attach a box to item `itemIndex` of a list (moving it from any other item). */
export function attachToListItem(d: Doc, listId: number, itemIndex: number, childId: number): void {
  for (const e of d.layout as any[]) {
    if (e.type !== "list") continue;
    for (const it of e.items as ListItemData[]) {
      if (it.children) it.children = it.children.filter((c) => c !== childId);
    }
  }
  const el = d.layout.find((e) => e.id === listId) as any;
  if (!el?.items?.[itemIndex]) return;
  (el.items[itemIndex].children ??= []).push(childId);
  for (const e of d.layout as any[]) if (e.type === "list") linkListChildren(d, e.id);
  afterListChange(d, listId);
}

/** Detach a box from its list item. */
export function detachFromList(d: Doc, listId: number, childId: number): void {
  const el = d.layout.find((e) => e.id === listId) as any;
  if (!el) return;
  for (const it of el.items as ListItemData[]) {
    if (it.children) it.children = it.children.filter((c) => c !== childId);
  }
  afterListChange(d, listId);
}

/** Boxes lying (80 %+) inside the list that could belong to one of its items. */
export function listChildCandidates(d: Doc, listId: number): any[] {
  const el = d.layout.find((e) => e.id === listId) as any;
  if (!el) return [];
  const inside = (b: BBox, o: BBox) => {
    const ix = Math.max(0, Math.min(b[2], o[2]) - Math.max(b[0], o[0]));
    const iy = Math.max(0, Math.min(b[3], o[3]) - Math.max(b[1], o[1]));
    const a = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
    return (ix * iy) / a >= 0.8;
  };
  return (d.layout as any[]).filter(
    (e) => e.id !== listId && e.type !== "list" && inside(e.bbox, el.bbox),
  );
}

/**
 * Attach every box inside the list to the item it sits under: the last
 * item whose first line starts above the box.
 */
export function autoAttachListChildren(d: Doc, listId: number): number {
  const el = d.layout.find((e) => e.id === listId) as any;
  if (!el) return 0;
  const tops = el.items.map((_: any, i: number) => {
    const ls = d.lines.filter((ln) => ln.block_id === listId && (ln as any).item_index === i);
    return ls.length ? Math.min(...ls.map((ln) => ln.bbox[1])) : null;
  });
  let n = 0;
  for (const c of listChildCandidates(d, listId)) {
    let best = -1;
    tops.forEach((t: number | null, i: number) => { if (t != null && t <= c.bbox[1] + 2) best = i; });
    if (best < 0) best = 0;
    if (!el.items[best]) continue;
    for (const it of el.items as ListItemData[]) {
      if (it.children) it.children = it.children.filter((x) => x !== c.id);
    }
    (el.items[best].children ??= []).push(c.id);
    n++;
  }
  afterListChange(d, listId);
  return n;
}

/**
 * One line per item, as in the reference pages: text "marker text",
 * list_level = the item's level, item_index = its position. An existing
 * line of the item keeps its box (and quad / font) when the text did not
 * change; a changed item gets one line spanning its old lines.
 */
export function syncListLines(d: Doc, listId: number): void {
  const el = d.layout.find((e) => e.id === listId) as any;
  if (!el || el.type !== "list") return;
  const items: ListItemData[] = el.items ?? [];
  const old = d.lines
    .filter((ln) => ln.block_id === listId)
    .sort((a, b) => a.line_in_block - b.line_in_block) as any[];
  const indexed = old.some((ln) => ln.item_index != null);
  const lineText = (it: ListItemData) => (it.marker ? it.marker + " " : "") + (it.text ?? "");

  const fresh: any[] = [];
  items.forEach((it, i) => {
    const want = lineText(it);
    const group = indexed
      ? old.filter((ln) => ln.item_index === i)
      : old.length === items.length ? [old[i]] : old.filter((ln) => ln.text.trim() === want.trim());
    const base = { block_id: listId, is_heading: false, list_level: it.level || 0, item_index: i };
    if (group.length && group.map((ln) => ln.text.trim()).join(" ") === want.trim()) {
      for (const ln of group) fresh.push({ ...ln, ...base });
    } else if (group.length) {
      const bbox = unionBox(group.map((ln) => ln.bbox));
      fresh.push({
        ...group[0], ...base, text: want, bbox,
        quad: group.length === 1 ? group[0].quad : quadFromBBox(bbox),
      });
    } else {
      const bbox = rowBox(el.bbox, i, Math.max(1, items.length));
      fresh.push({
        text: want, bbox, ...base, line_in_block: 0,
        font_size: Math.max(8, (bbox[3] - bbox[1]) * 0.7), font_id: null, quad: quadFromBBox(bbox),
      });
    }
  });
  fresh.forEach((ln, k) => { ln.line_in_block = k; });
  replaceLines(d, (ln) => ln.block_id === listId, fresh);
}

/** After any change to a list's items: children links, lines and text. */
export function afterListChange(d: Doc, listId: number): void {
  linkListChildren(d, listId);
  syncListLines(d, listId);
  const el = d.layout.find((e) => e.id === listId) as any;
  syncListText(el, d);
}
