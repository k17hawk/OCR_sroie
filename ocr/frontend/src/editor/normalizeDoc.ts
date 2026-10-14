/**
 * normalizeForExport — turn what the annotator holds in memory into the
 * flat "v2" page schema (the shape of sample.json).
 *
 * The editor lets annotators draw boxes inside boxes (a text_line inside a
 * paragraph, a form_label inside a form_field, …). The v2 schema has no
 * nesting: those things live in `lines`, `fields`, `tables`, … of ONE flat
 * layout element. This function does that conversion and rebuilds every
 * derived part (page text, blocks, lines, html, reading order, quality
 * counts), and returns warnings for what it could not fix on its own.
 *
 * Pure function: never mutates its input, no React / store imports.
 */

type BBox = [number, number, number, number];
type Pt = [number, number];
type Quad = Pt[];

export interface ExportWarning {
  id?: number;          // layout id in the ORIGINAL (editor) document
  level: "fixed" | "check";
  message: string;
}

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

const LOCKED = new Set(["logo", "signature", "stamp"]);
const NO_NESTING_CHECK = new Set(["logo", "signature", "stamp", "figure"]);

/** Child kinds that are folded into the region that contains them. */
const LINE_KINDS = new Set(["text_line"]);
const FIELD_LABEL_KINDS = new Set(["form_label", "label"]);
const FIELD_VALUE_KINDS = new Set(["form_value"]);

// const TEXT_TYPES = new Set([
//   "header", "title", "heading", "subheading", "paragraph", "footer",
//   "page_number", "signature_caption", "table_heading", "table_footer",
//   "list_item", "mixed_text", "handwriting", "handwritten_annotation",
//   "handwritten_correction", "handwritten_number", "text_line",
//   "form_label", "form_value", "label", "options",
// ]);

// ---------------------------------------------------------------------------
// geometry helpers
// ---------------------------------------------------------------------------

const r1 = (n: number) => Math.round(n * 10) / 10;
const area = (b: BBox) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);

function quadOf(b: BBox): Quad {
  return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
}
function bboxOfQuad(q: Quad): BBox {
  const xs = q.map((p) => p[0]), ys = q.map((p) => p[1]);
  return [
    Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)),
    Math.ceil(Math.max(...xs)), Math.ceil(Math.max(...ys)),
  ];
}
function union(a: BBox, b: BBox): BBox {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
}
/** fraction of `inner` that lies inside `outer` */
function insideFrac(inner: BBox, outer: BBox): number {
  const ix = Math.max(0, Math.min(inner[2], outer[2]) - Math.max(inner[0], outer[0]));
  const iy = Math.max(0, Math.min(inner[3], outer[3]) - Math.max(inner[1], outer[1]));
  const a = area(inner);
  return a > 0 ? (ix * iy) / a : 0;
}
function vOverlap(a: BBox, b: BBox): number {
  const o = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return o / Math.max(1, Math.min(a[3] - a[1], b[3] - b[1]));
}
/** Make bbox/quad consistent: quad rounded to 0.1, bbox = int extent of quad. */
function fixGeom(o: any) {
  if (!o) return;
  if (Array.isArray(o.quad) && o.quad.length === 4 && o.quad.every((p: any) => Array.isArray(p))) {
    o.quad = o.quad.map((p: number[]) => [r1(p[0]), r1(p[1])]);
    o.bbox = bboxOfQuad(o.quad);
  } else if (Array.isArray(o.bbox)) {
    o.bbox = o.bbox.map((v: number, i: number) => (i < 2 ? Math.floor(v) : Math.ceil(v))) as BBox;
    o.quad = quadOf(o.bbox);
  }
}

// ---------------------------------------------------------------------------
// text helpers
// ---------------------------------------------------------------------------

/** "\\n" typed literally (backslash + n) → real newline; NFC. */
function cleanText(s: unknown): string {
  if (typeof s !== "string") return "";
  return s.replace(/\\n/g, "\n").replace(/\r\n?/g, "\n").normalize("NFC");
}

function stripHeadingPrefix(s: string): string {
  return s.replace(/^#+\s*/, "");
}

function escHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---------------------------------------------------------------------------
// list parsing
// ---------------------------------------------------------------------------

const MARKER_RE =
  /^(\s*)((?:[०-९]+|[0-9]+)[.)]|\(?[क-ह]\)|\([०-९0-9]+\)|[a-zA-Z][.)]|(?:[ivxlcdm]+|[IVXLCDM]+)[.)]|[-•*●▪◦–])\s+(.*)$/;

function parseListText(text: string): { marker: string; level: number; text: string }[] {
  const items: { marker: string; level: number; text: string }[] = [];
  for (const raw of text.split("\n")) {
    if (!raw.trim()) continue;
    const m = raw.match(MARKER_RE);
    if (m) {
      const marker = m[2] === "-" || m[2] === "*" || m[2] === "●" ? "•" : m[2];
      items.push({ marker, level: Math.floor(m[1].replace(/\t/g, "  ").length / 2), text: m[3].trim() });
    } else if (items.length) {
      items[items.length - 1].text += " " + raw.trim(); // wrapped continuation line
    } else {
      items.push({ marker: "", level: 0, text: raw.trim() });
    }
  }
  return items;
}

function listStyleOf(items: { marker: string }[]): string {
  const m = items.find((i) => i.marker)?.marker ?? "";
  if (/^[०-९]/.test(m) || /^\(?[क-ह]/.test(m)) return "dev";
  if (/^[0-9]/.test(m) || /^[a-zA-Z][.)]$/.test(m)) return /^[ivxIVX]+[.)]$/.test(m) ? "roman" : "latin";
  return "bullet";
}

function listText(
  items: { marker: string; level: number; text: string; children?: number[] }[],
  childText: (id: number) => string = () => "",
): string {
  const out: string[] = [];
  for (const it of items) {
    const indent = "  ".repeat(it.level);
    out.push(indent + (it.marker === "•" ? "-" : it.marker) + (it.marker ? " " : "") + it.text);
    for (const cid of it.children ?? []) {
      const t = childText(cid);
      if (t) for (const part of t.split("\n")) out.push(indent + "  " + part);
    }
  }
  return out.join("\n");
}

// ---------------------------------------------------------------------------
// form fields
// ---------------------------------------------------------------------------

function checkboxText(options: any[]): string {
  return options.map((o) => `[${o.checked ? "x" : " "}] ${o.text}`).join(" ");
}

function fieldLine(f: any): string {
  const label = cleanText(f.label?.text ?? "");
  if (!f.value) return label;
  if ("options" in f.value) return `${label} ${checkboxText(f.value.options ?? [])}`.trim();
  return `${label} ${cleanText(f.value.text ?? "")}`.trim();
}

/** Sort rows top→bottom, then left→right within a row. */
function readingSort<T>(arr: T[], box: (t: T) => BBox): T[] {
  const hs = arr.map((a) => box(a)[3] - box(a)[1]).sort((a, b) => a - b);
  const tol = Math.max(4, (hs[Math.floor(hs.length / 2)] ?? 20) / 2);
  const cy = (t: T) => (box(t)[1] + box(t)[3]) / 2;
  const sorted = [...arr].sort((a, b) => cy(a) - cy(b));
  const rows: T[][] = [];
  for (const t of sorted) {
    const row = rows[rows.length - 1];
    if (row && Math.abs(cy(t) - cy(row[0])) <= tol) row.push(t);
    else rows.push([t]);
  }
  return rows.flatMap((r) => r.sort((a, b) => box(a)[0] - box(b)[0]));
}

/** Pair label boxes with the value box to their right on the same row. */
function buildFields(labels: any[], values: any[], warn: (w: ExportWarning) => void): any[] {
  const fields: any[] = [];
  const usedValues = new Set<any>();
  const sortedLabels = readingSort(labels, (l) => l.bbox);
  for (const lab of sortedLabels) {
    const cands = values
      .filter((v) => !usedValues.has(v) && vOverlap(lab.bbox, v.bbox) >= 0.3 && v.bbox[0] >= lab.bbox[0])
      .sort((a, b) => a.bbox[0] - b.bbox[0]);
    // nearest value to the right that is not past another label
    const nextLabelX = Math.min(
      ...sortedLabels
        .filter((o) => o !== lab && vOverlap(lab.bbox, o.bbox) >= 0.3 && o.bbox[0] > lab.bbox[0])
        .map((o) => o.bbox[0]),
    );
    const val = cands.find((v) => v.bbox[0] < nextLabelX) ?? null;
    if (val) usedValues.add(val);
    fields.push({
      label: { text: cleanText(lab.text), bbox: lab.bbox, quad: lab.quad },
      field_type: lab.field_type ?? "text",
      value: val
        ? { text: cleanText(val.text), bbox: val.bbox, quad: val.quad, handwritten: !!val.handwritten }
        : null,
    });
  }
  for (const v of values) {
    if (usedValues.has(v)) continue;
    warn({ id: v.id, level: "check", message: `form_value #${v.id} "${cleanText(v.text).slice(0, 30)}" has no label to its left on the same row — added with an empty label.` });
    fields.push({
      label: { text: "", bbox: v.bbox, quad: v.quad },
      field_type: "text",
      value: { text: cleanText(v.text), bbox: v.bbox, quad: v.quad, handwritten: !!v.handwritten },
    });
  }
  return fields;
}

// ---------------------------------------------------------------------------
// tables
// ---------------------------------------------------------------------------

function tableHtml(t: any): string {
  const rows = new Map<number, any[]>();
  for (const c of t.cells) {
    if (!rows.has(c.row)) rows.set(c.row, []);
    rows.get(c.row)!.push(c);
  }
  const out: string[] = ["<table>"];
  for (const r of [...rows.keys()].sort((a, b) => a - b)) {
    out.push("<tr>");
    for (const c of rows.get(r)!.sort((a, b) => a.col - b.col)) {
      const tag = c.is_header ? "th" : "td";
      const attrs =
        (c.colspan > 1 ? ` colspan="${c.colspan}"` : "") +
        (c.rowspan > 1 ? ` rowspan="${c.rowspan}"` : "");
      out.push(`<${tag}${attrs}>${escHtml(cleanText(c.text ?? ""))}</${tag}>`);
    }
    out.push("</tr>");
  }
  out.push("</table>");
  return out.join("");
}

function normalizeTable(t: any): any {
  t.cells = (t.cells ?? []).map((c: any) => {
    const cell = {
      id: c.id, row: c.row, col: c.col,
      rowspan: Math.max(1, c.rowspan ?? 1), colspan: Math.max(1, c.colspan ?? 1),
      text: cleanText(c.text ?? ""),
      is_header: !!c.is_header, handwritten: !!c.handwritten,
      empty: false, bbox: c.bbox, quad: c.quad,
    };
    cell.empty = cell.text.trim() === "";
    fixGeom(cell);
    return cell;
  });
  t.cells.sort((a: any, b: any) => a.row - b.row || a.col - b.col);
  const num_rows = Math.max(0, ...t.cells.map((c: any) => c.row + c.rowspan));
  const num_cols = Math.max(0, ...t.cells.map((c: any) => c.col + c.colspan));
  let header_rows = 0;
  for (let r = 0; r < num_rows; r++) {
    const rowCells = t.cells.filter((c: any) => c.row === r);
    if (rowCells.length && rowCells.every((c: any) => c.is_header)) header_rows++;
    else break;
  }
  fixGeom(t);
  return {
    bbox: t.bbox, quad: t.quad, num_rows, num_cols, header_rows,
    ruling_style: t.ruling_style ?? "full", theme: t.theme ?? null, col_types: t.col_types ?? null,
    html: tableHtml(t),
    reading_order: t.cells.map((c: any) => c.id),
    cells: t.cells,
  };
}

// ---------------------------------------------------------------------------
// lines
// ---------------------------------------------------------------------------

function makeLine(text: string, bbox: BBox, quad: Quad | undefined, blockId: number, isHeading: boolean, extra: any = {}) {
  const h = bbox[3] - bbox[1];
  return {
    text,
    bbox,
    block_id: blockId,
    line_in_block: 0,
    is_heading: isHeading,
    font_size: extra.font_size ?? Math.max(8, Math.round(h * 0.9)),
    font_id: extra.font_id ?? null,
    ...(extra.role ? { role: extra.role } : {}),
    ...(extra.handwritten ? { handwritten: true } : {}),
    quad: quad ?? quadOf(bbox),
  };
}

/** Split a box into n equal horizontal strips (for synthesized lines). */
function strips(b: BBox, n: number): BBox[] {
  const h = (b[3] - b[1]) / Math.max(1, n);
  return Array.from({ length: n }, (_, i) =>
    [b[0], Math.floor(b[1] + i * h), b[2], Math.ceil(b[1] + (i + 1) * h)] as BBox);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

export function normalizeForExport(input: any): { doc: any; warnings: ExportWarning[] } {
  const src = JSON.parse(JSON.stringify(input));
  const warnings: ExportWarning[] = [];
  const warn = (w: ExportWarning) => warnings.push(w);

  let layout: any[] = (src.layout ?? []).map((e: any) => ({ ...e }));
  let lines: any[] = (src.lines ?? []).map((l: any) => ({ ...l }));
  const tables: any[] = src.tables ?? [];

  for (const el of layout) {
    fixGeom(el);
    if ("text" in el) el.text = cleanText(el.text);
  }
  for (const ln of lines) { fixGeom(ln); ln.text = cleanText(ln.text); }

  // const byId = new Map<number, any>(layout.map((e) => [e.id, e]));
  /** smallest element (of allowed types) that contains `el` */
  const container = (el: any, ok: (p: any) => boolean) =>
    layout
      .filter((p) => p !== el && ok(p) && insideFrac(el.bbox, p.bbox) >= 0.8 && area(p.bbox) >= area(el.bbox))
      .sort((a, b) => area(a.bbox) - area(b.bbox))[0];

  const removed = new Set<any>();

  // 1. text_line boxes drawn as layout → lines of the region that holds them
  for (const el of layout) {
    if (!LINE_KINDS.has(el.type)) continue;
    const parent = container(el, (p) => !LINE_KINDS.has(p.type) && !LOCKED.has(p.type) && p.type !== "figure");
    if (!parent) {
      el.type = "paragraph";
      warn({ id: el.id, level: "fixed", message: `text_line #${el.id} is not inside any region — exported as its own paragraph.` });
      continue;
    }
    // drop the line(s) that belonged to the text_line element itself
    lines = lines.filter((l) => l.block_id !== el.id);
    lines.push({ ...makeLine(el.text ?? "", el.bbox, el.quad, parent.id, parent.type === "heading"), __fromTextLine: true });
    removed.add(el);
    warn({ id: el.id, level: "fixed", message: `text_line #${el.id} moved into the lines of ${parent.type} #${parent.id}.` });
  }
  layout = layout.filter((e) => !removed.has(e));

  // 2. form_label / form_value boxes → fields of the form_field that holds them
  const formParts = layout.filter((e) => FIELD_LABEL_KINDS.has(e.type) || FIELD_VALUE_KINDS.has(e.type));
  const groups = new Map<any, { labels: any[]; values: any[] }>();
  const orphans: any[] = [];
  for (const part of formParts) {
    const parent = container(part, (p) => p.type === "form_field");
    if (!parent) { orphans.push(part); continue; }
    if (!groups.has(parent)) groups.set(parent, { labels: [], values: [] });
    (FIELD_LABEL_KINDS.has(part.type) ? groups.get(parent)!.labels : groups.get(parent)!.values).push(part);
    removed.add(part);
    lines = lines.filter((l) => l.block_id !== part.id);
  }
  for (const [parent, g] of groups) {
    const newFields = buildFields(g.labels, g.values, warn);
    parent.fields = [...(parent.fields ?? []), ...newFields];
    warn({ id: parent.id, level: "fixed", message: `form_field #${parent.id}: ${g.labels.length} label / ${g.values.length} value box(es) turned into ${newFields.length} field(s).` });
  }
  // orphans: wrap each label (+ value on its row) into its own form_field
  if (orphans.length) {
    const labs = orphans.filter((o) => FIELD_LABEL_KINDS.has(o.type));
    const vals = orphans.filter((o) => FIELD_VALUE_KINDS.has(o.type));
    const fields = buildFields(labs, vals, warn);
    for (const f of fields) {
      const srcEl = [...labs, ...vals].find((o) => o.bbox === f.label.bbox || o.bbox === f.value?.bbox);
      const bbox = f.value ? union(f.label.bbox, f.value.bbox) : f.label.bbox;
      const ff: any = {
        id: srcEl?.id, type: "form_field", bbox, quad: quadOf(bbox),
        reading_order: srcEl?.reading_order ?? null, text: "", fields: [f],
      };
      layout.push(ff);
      warn({ id: srcEl?.id, level: "fixed", message: `${srcEl?.type ?? "form part"} #${srcEl?.id} was not inside a form_field — wrapped in a new form_field.` });
    }
    for (const o of orphans) { removed.add(o); lines = lines.filter((l) => l.block_id !== o.id); }
  }
  layout = layout.filter((e) => !removed.has(e));

  // 3. form_field inside form_field → merge into the outer one
  for (const inner of [...layout]) {
    if (inner.type !== "form_field" || removed.has(inner)) continue;
    const outer = container(inner, (p) => p.type === "form_field" && !removed.has(p));
    if (!outer) continue;
    outer.fields = [...(outer.fields ?? []), ...(inner.fields ?? [])];
    removed.add(inner);
    lines = lines.filter((l) => l.block_id !== inner.id);
    warn({ id: inner.id, level: "fixed", message: `form_field #${inner.id} was inside form_field #${outer.id} — merged into it.` });
    if (!(inner.fields ?? []).length && (inner.text ?? "").trim()) {
      warn({ id: outer.id, level: "check", message: `form_field #${inner.id} had text "${inner.text.trim().slice(0, 30)}" but no label/value boxes — that text was lost; draw its label and value.` });
    }
  }
  layout = layout.filter((e) => !removed.has(e));

  // 4. per-type cleanup
  for (const el of layout) {
    if (LOCKED.has(el.type)) {
      el.reading_order = null;
      delete el.text;
      if (el.type === "signature" && !el.signer) el.signer = "other";
      if (el.type === "stamp" && el.stamp_text == null) el.stamp_text = "";
      continue;
    }
    if (el.type === "heading") {
      const level = Math.min(6, Math.max(1, Number(el.level) || 2));
      el.level = level;
      el.text = "#".repeat(level) + " " + stripHeadingPrefix(el.text ?? "");
    }
    if (el.type === "figure") {
      el.text = "<figure/>";
      el.subtype = el.subtype ?? "photo";
    }
    if (el.type === "list") {
      let items = Array.isArray(el.items) ? el.items : [];
      const parsed = !items.length && !!(el.text ?? "").trim();
      if (parsed) {
        items = parseListText(el.text);
        warn({ id: el.id, level: "fixed", message: `list #${el.id}: ${items.length} item(s) parsed from its text.` });
        if (items.some((i: any) => !i.marker)) {
          warn({ id: el.id, level: "check", message: `list #${el.id}: some items have no marker (१. / क) / •) — check them.` });
        }
      }
      el.items = items.map((i: any) => ({
        marker: i.marker ?? "", level: Number(i.level) || 0, text: cleanText(i.text),
        ...(Array.isArray(i.children) && i.children.length ? { children: i.children.map(Number) } : {}),
      }));
      el.list_style = !parsed && el.list_style ? el.list_style : listStyleOf(el.items);
    }
    if (el.type === "form_field") {
      el.fields = (el.fields ?? []).map((f: any) => {
        const label = { text: cleanText(f.label?.text ?? ""), bbox: f.label?.bbox, quad: f.label?.quad };
        fixGeom(label);
        let value: any = null;
        if (f.value && "options" in f.value) {
          const options = (f.value.options ?? []).map((o: any) => ({
            text: cleanText(o.text), checked: !!o.checked,
            box_quad: (o.box_quad ?? []).map((p: number[]) => [r1(p[0]), r1(p[1])]),
            text_quad: (o.text_quad ?? []).map((p: number[]) => [r1(p[0]), r1(p[1])]),
          }));
          value = { text: checkboxText(options), options };
        } else if (f.value) {
          value = { text: cleanText(f.value.text), bbox: f.value.bbox, quad: f.value.quad, handwritten: !!f.value.handwritten };
          fixGeom(value);
        }
        return { label, field_type: f.field_type ?? "text", value };
      });
      el.fields = readingSort(el.fields, (f: any) => f.label.bbox);
      if (el.fields.length) el.text = el.fields.map(fieldLine).join("\n");
      else warn({ id: el.id, level: "check", message: `form_field #${el.id} has no fields — draw its label/value with the Form field tool, or change its type.` });
    }
    if (el.type !== "table" && /<table[\s>]/i.test(el.text ?? "")) {
      warn({ id: el.id, level: "check", message: `${el.type} #${el.id} contains table HTML — change its type to "table" and run Table OCR so it gets real cells.` });
    }
  }

  // 4b. list items own the boxes named in their `children`: those boxes
  //     get parent_id / parent_item and no reading order of their own, and
  //     the list text shows each child's text under its item
  {
    const byId = new Map<number, any>(layout.map((e) => [e.id, e]));
    const owned = new Map<number, [number, number]>();
    for (const el of layout) {
      if (el.type !== "list") continue;
      el.items.forEach((it: any, i: number) => {
        if (!it.children) return;
        it.children = it.children.filter((c: number) => byId.has(c) && c !== el.id && !owned.has(c));
        it.children.forEach((c: number) => owned.set(c, [el.id, i]));
        if (!it.children.length) delete it.children;
      });
    }
    for (const el of layout) {
      const o = owned.get(el.id);
      if (o) { el.parent_id = o[0]; el.parent_item = o[1]; el.reading_order = null; }
      else { delete el.parent_id; delete el.parent_item; }
    }
    const childText = (id: number) => {
      const c = byId.get(id);
      return !c ? "" : c.type === "figure" ? "<figure/>" : (c.text ?? "");
    };
    for (const el of layout) if (el.type === "list") el.text = listText(el.items, childText);
  }

  // 5. nesting that can't be resolved automatically
  for (const el of layout) {
    if (NO_NESTING_CHECK.has(el.type)) continue;
    if (el.parent_id != null) continue; // belongs to a list item: allowed
    const outer = container(el, (p) => !NO_NESTING_CHECK.has(p.type));
    if (outer) {
      warn({ id: el.id, level: "check", message: `${el.type} #${el.id} sits inside ${outer.type} #${outer.id}. The export format has no nesting — shrink #${outer.id} or delete one of them.` });
    }
  }

  // 6. reading order: keep if already a clean 0..n-1 sequence, else renumber
  const ordered = layout.filter((e) => !LOCKED.has(e.type) && e.parent_id == null);
  const ros = ordered.map((e) => e.reading_order);
  const clean =
    ros.every((r) => Number.isInteger(r)) &&
    new Set(ros).size === ros.length &&
    [...ros].sort((a, b) => a - b).every((r, i) => r === i);
  if (!clean) {
    readingSort(ordered, (e) => e.bbox).forEach((e, i) => { e.reading_order = i; });
    warn({ level: "fixed", message: "Reading order had duplicates or gaps — renumbered 0…n-1 top-to-bottom, left-to-right. Check it in the Reading order panel." });
  }

  // 7. final element order + new ids (0..n-1): reading order first, locked after
  const origIdx = new Map<any, number>(layout.map((e, i) => [e, i]));
  const group = (e: any) => (LOCKED.has(e.type) ? 2 : e.parent_id != null ? 1 : 0);
  layout.sort((a, b) => {
    const la = group(a), lb = group(b);
    return la - lb || (a.reading_order ?? 0) - (b.reading_order ?? 0) ||
      origIdx.get(a)! - origIdx.get(b)!;
  });
  const idMap = new Map<number, number>();
  layout.forEach((e, i) => { if (e.id != null) idMap.set(e.id, i); e.__newId = i; });

  // 8. tables: re-index the ones still referenced by a table element
  const outTables: any[] = [];
  const tableIdxMap = new Map<number, number>();
  for (const el of layout) {
    if (el.type !== "table") continue;
    const t = tables[el.table_index];
    if (!t) {
      warn({ id: el.id, level: "check", message: `table #${el.id} has no cell data — run Table OCR on it.` });
      el.table_index = -1;
      continue;
    }
    const nt = normalizeTable(JSON.parse(JSON.stringify(t)));
    tableIdxMap.set(el.table_index, outTables.length);
    el.table_index = outTables.length;
    el.text = nt.html;
    outTables.push(nt);
  }
  const ruling_lines = (src.ruling_lines ?? [])
    .filter((rl: any) => tableIdxMap.has(rl.table_index))
    .map((rl: any) => ({ ...rl, table_index: tableIdxMap.get(rl.table_index) }));

  // 9. lines: rebuild for structured types, keep/synthesize for text types
  const keptLines = lines.filter((l) => idMap.has(l.block_id) && !/^<table/i.test(l.text ?? ""));
  const outLines: any[] = [];
  // reuse font_size / font_id of an existing line with the same text
  const reuse = (oldId: number, text: string, bbox: BBox) => {
    const m = keptLines
      .filter((l) => l.block_id === oldId && l.text === text)
      .sort((a, b) => Math.abs(a.bbox[1] - bbox[1]) - Math.abs(b.bbox[1] - bbox[1]))[0];
    return m ? { font_size: m.font_size, font_id: m.font_id } : {};
  };
  for (const el of layout) {
    const nid = el.__newId;
    const isHeading = el.type === "heading";
    if (LOCKED.has(el.type) || el.type === "figure") continue;

    if (el.type === "form_field") {
      for (const f of el.fields) {
        if (f.label.text) {
          outLines.push(makeLine(f.label.text, f.label.bbox, f.label.quad, nid, false,
            { ...reuse(el.id, f.label.text, f.label.bbox), role: "label" }));
        }
        if (f.value && "options" in f.value) {
          for (const o of f.value.options) {
            if (!o.text || !o.text_quad?.length) continue;
            const ob = bboxOfQuad(o.text_quad);
            outLines.push(makeLine(o.text, ob, o.text_quad, nid, false,
              { ...reuse(el.id, o.text, ob), role: "option" }));
          }
        } else if (f.value?.text) {
          outLines.push(makeLine(f.value.text, f.value.bbox, f.value.quad, nid, false,
            { ...reuse(el.id, f.value.text, f.value.bbox), role: "value", handwritten: f.value.handwritten }));
        }
      }
      continue;
    }
    if (el.type === "table") {
      const t = outTables[el.table_index];
      for (const c of t?.cells ?? []) {
        if (c.empty) continue;
        // keep the tight text box OCR produced for this cell, if any
        const prev = keptLines.find((l) => l.block_id === el.id && l.cell_id === c.id && l.text === c.text);
        const base = prev
          ? makeLine(c.text, prev.bbox, prev.quad, nid, false,
              { font_size: prev.font_size, font_id: prev.font_id, handwritten: c.handwritten || prev.handwritten })
          : makeLine(c.text, c.bbox, c.quad, nid, false,
              { ...reuse(el.id, c.text, c.bbox), handwritten: c.handwritten });
        const { quad, handwritten, ...rest } = base as any;
        outLines.push({
          ...rest, table_index: el.table_index, cell_id: c.id,
          ...(handwritten ? { handwritten } : {}), quad,
        });
      }
      continue;
    }
    const own = keptLines
      .filter((l) => l.block_id === el.id)
      .filter((l) => (l.text ?? "").trim() !== "");
    // If the region also has a single whole-region line plus real text_line
    // children, drop the whole-region one.
    const fromTL = own.filter((l) => l.__fromTextLine);
    const use = fromTL.length ? fromTL : own;
    if (use.length) {
      for (const l of readingSort(use, (x) => x.bbox)) {
        const { __fromTextLine, ...rest } = l;
        outLines.push({ ...rest, block_id: nid, is_heading: isHeading, text: isHeading ? stripHeadingPrefix(rest.text) : rest.text });
      }
      continue;
    }
    // nothing usable: synthesize from the element's text
    const text = el.type === "list"
      ? el.items.map((i: any) => (i.marker ? i.marker + " " : "") + i.text).join("\n")
      : isHeading ? stripHeadingPrefix(el.text ?? "") : (el.text ?? "");
    const parts = text.split("\n").map((s: string) => s.trim()).filter(Boolean);
    strips(el.bbox, parts.length).forEach((b, i) =>
      outLines.push(makeLine(parts[i], b, undefined, nid, isHeading, reuse(el.id, parts[i], b))));
  }
  // line_in_block in order within each block
  const counters = new Map<number, number>();
  for (const l of outLines) {
    const n = counters.get(l.block_id) ?? 0;
    l.line_in_block = n;
    counters.set(l.block_id, n + 1);
  }

  // 10. checkbox extra regions mirror the form options; keep fingerprints
  const extra: any[] = [];
  for (const el of layout) {
    if (el.type !== "form_field") continue;
    for (const f of el.fields) {
      if (!f.value || !("options" in f.value)) continue;
      for (const o of f.value.options) {
        if (!o.box_quad?.length) continue;
        extra.push({ type: "checkbox", checked: o.checked, option: o.text, bbox: bboxOfQuad(o.box_quad), quad: o.box_quad });
      }
    }
  }
  // keep other regions; keep standalone checkboxes (not a copy of a form option)
  const mirrored = new Set(extra.map((e) => e.bbox.join(",")));
  for (const er of src.extra_regions ?? []) {
    const e2 = { ...er }; fixGeom(e2);
    if (er.type === "checkbox") {
      if (mirrored.has(e2.bbox.join(","))) continue;
      e2.checked = !!e2.checked;
      e2.option = cleanText(e2.option ?? "");
    }
    extra.push(e2);
  }

  // 11. write elements in the sample's key order
  const outLayout = layout.map((el) => {
    const base: any = { id: el.__newId, type: el.type, bbox: el.bbox, reading_order: el.reading_order ?? null };
    if (!LOCKED.has(el.type)) base.text = el.text ?? "";
    switch (el.type) {
      case "heading": base.level = el.level; break;
      case "figure": base.subtype = el.subtype; break;
      case "form_field": base.fields = el.fields; break;
      case "table": base.table_index = el.table_index; break;
      case "list":
        base.list_style = el.list_style;
        base.items = el.items.map((it: any) => it.children
          ? { ...it, children: it.children.map((c: number) => idMap.get(c)).filter((c: any) => c != null) }
          : it);
        break;
      case "signature": base.signer = el.signer; break;
      case "stamp": base.stamp_text = cleanText(el.stamp_text); break;
    }
    if (el.type !== "figure" && el.subtype != null) base.subtype = el.subtype;
    if (el.handwritten != null && !LOCKED.has(el.type)) base.handwritten = el.handwritten;
    if (el.parent_id != null && idMap.has(el.parent_id)) {
      base.parent_id = idMap.get(el.parent_id);
      base.parent_item = el.parent_item;
    }
    base.quad = el.quad;
    return base;
  });

  // 12. blocks = one per element
  const blocks = outLayout.map((e) => ({ type: e.type, bbox: e.bbox, block_id: e.id, font_id: null, quad: e.quad }));

  // 13. page text in reading order
  const pageText = outLayout
    .filter((e) => e.reading_order != null)
    .sort((a, b) => a.reading_order - b.reading_order)
    .map((e) => e.text)
    .filter((t) => t && t.trim())
    .join("\n\n");

  // 14. quality counts
  const q = { ...(src.quality ?? {}) };
  const allFields = outLayout.filter((e) => e.type === "form_field").flatMap((e) => e.fields);
  q.line_count = outLines.length;
  q.table_count = outTables.length;
  q.figure_count = outLayout.filter((e) => e.type === "figure").length;
  q.has_stamp = outLayout.some((e) => e.type === "stamp");
  q.has_signature = outLayout.some((e) => e.type === "signature");
  q.stamp_bboxes = outLayout.filter((e) => e.type === "stamp").map((e) => e.bbox);
  q.signature_bboxes = outLayout.filter((e) => e.type === "signature").map((e) => e.bbox);
  q.has_handwriting =
    allFields.some((f: any) => f.value?.handwritten) ||
    outTables.some((t) => t.cells.some((c: any) => c.handwritten)) ||
    outLines.some((l) => l.handwritten) ||
    !!src.quality?.has_handwriting;
  q.raw_lengths = {
    paragraph_chars: outLayout.filter((e) => e.type === "paragraph").reduce((n, e) => n + (e.text?.length ?? 0), 0),
    list_items: outLayout.filter((e) => e.type === "list").reduce((n, e) => n + e.items.length, 0),
    form_fields: allFields.length,
    table_cells: outTables.reduce((n, t) => n + t.cells.length, 0),
  };

  const doc = {
    v: 2,
    image: src.image,
    width: src.width,
    height: src.height,
    document_id: src.document_id,
    page_number: src.page_number ?? 1,
    page_count: src.page_count ?? 1,
    language: src.language,
    digit_style: src.digit_style,
    doc_type: src.doc_type,
    text: pageText,
    fonts: src.fonts ?? [],
    font_ids: src.font_ids ?? [],
    layout: outLayout,
    blocks,
    lines: outLines,
    tables: outTables,
    ruling_lines,
    extra_regions: extra,
    quality: q,
    annotation: src.annotation ?? { source: "real", tool: "devalabel", status: "in_progress" },
  };

  if (doc.doc_type === "other" || !doc.doc_type) {
    warn({ level: "check", message: 'doc_type is "other" — pick the document type in Page meta.' });
  }
  return { doc: JSON.parse(JSON.stringify(doc).normalize("NFC")), warnings };
}
