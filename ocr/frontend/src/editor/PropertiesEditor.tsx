import { useEffect, useState } from "react";
import { useStore } from "../store";
import type {
  Doc, LayoutElement, FormFieldEntry, FieldOption, BBox,
} from "../types";
import {
  useProjectTypes, withCurrent, prettyType, typeColor,
} from "../hooks/useProjectTypes";
import { DevanagariTextarea } from "./DevanagariTextarea";
import { detectRegionLines } from "./DetectLinesButton";
import { OcrRegionButton } from "./OcrRegionButton";
import { TableOcrButton } from "./TableOcrButton";
import { TableSuggestPanel } from "./TableSuggestPanel";
import {
  setElementText, resizeLayoutElement, emptyTable, parentLayoutOf,
  appendReadingOrder, removeFromReadingOrder, nextReadingOrder,
  isReadingOrderLocked, addLayoutElement, syncListText, listItemsFromText,
  tableText, syncTable, syncRegionTextFromLines, textVsLines, textIntoLines,
  afterListChange, attachToListItem, detachFromList, listChildCandidates,
  autoAttachListChildren, syncCellFromLines,
} from "./commands";

/** Types with no text of their own on the server: no generic text box. */
const NO_TEXT_BOX = new Set(["logo", "figure", "signature", "stamp"]);
const TEXT_HINT: Record<string, string> = {
  form_field: "One field per line: label: value",
  table: "Table HTML: <tr><td>…</td></tr>, <th> for headers, rowspan / colspan allowed, <td data-handwritten> for handwritten cells. (Rows of cells separated by | also work.)",
};

/**
 * Text box that keeps exactly what you type while it has focus. Form-field
 * and table text is rebuilt from the saved fields / cells; showing that
 * rebuilt text on every keystroke moved the cursor and swallowed new lines
 * and half-typed HTML. Every keystroke is still applied to the document.
 */
function DraftTextarea({
  value, onChange, rows,
}: { value: string; onChange: (v: string) => void; rows?: number }) {
  const [draft, setDraft] = useState(value);
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setDraft(value);
  }, [value, focused]);
  return (
    <div
      onFocus={() => setFocused(true)}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFocused(false);
      }}
    >
      <DevanagariTextarea
        value={draft}
        onChange={(v) => { setDraft(v); onChange(v); }}
        rows={rows}
      />
    </div>
  );
}

/**
 * The stored text has something its lines don't (or the other way round).
 * A save keeps the lines, so show both and let the annotator decide.
 */
function TextLinesMismatch({ doc, id, edit }: {
  doc: Doc; id: number; edit: (recipe: (d: Doc) => void) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const diff = textVsLines(doc, id);
  if (!diff) return null;
  const show = (t: string) => t.replace(/\n/g, " ⏎ ");
  const textRows = diff.text.split("\n").length;
  const lineRows = diff.lines.split("\n").length;
  // more rows in the text than lines: printed rows have no line yet
  const isTable = doc.layout.find((e) => e.id === id)?.type === "table";
  // more rows in the text than lines: printed rows have no line yet
  // (a table's lines come from its cells / Table OCR, not Detect lines)
  const missingLines = textRows > lineRows && !isTable;
  const detect = async () => {
    setBusy(true); setMsg(null);
    try {
      const w = await detectRegionLines(id);
      setMsg(w.length ? w.join("\n") : null);
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };
  const useLines = () => {
    if (missingLines && !window.confirm(
      `The text has ${textRows} rows but only ${lineRows} line(s). "Use the lines" deletes the ` +
      `other ${textRows - lineRows} row(s) from the text. If they are printed, use "Detect lines" instead.\n\nDelete them?`,
    )) return;
    edit((d) => syncRegionTextFromLines(d, id));
  };
  const btn = "rounded px-2 py-1 hover:bg-amber-700 disabled:opacity-40 ";
  return (
    <div className="mt-2 rounded border border-amber-700 bg-amber-950/60 p-2 text-[11px] text-amber-100">
      <div className="mb-1 font-medium">The text and the lines differ — a save keeps the lines.</div>
      <div className="font-mono"><span className="text-amber-300">text : </span>{show(diff.text)}</div>
      <div className="font-mono"><span className="text-amber-300">lines: </span>{show(diff.lines)}</div>
      {missingLines && (
        <div className="mt-1 text-amber-300">
          {textRows - lineRows} row(s) of the text have no line. If they are printed, press Detect lines.
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-2">
        {!isTable && <button
          type="button"
          disabled={busy}
          className={btn + (missingLines ? "bg-teal-700 text-white hover:bg-teal-600" : "bg-amber-800/70")}
          title="Measure one line per text row on the scan (keeps the existing lines' ids)"
          onClick={() => void detect()}
        >{busy ? "Detecting…" : "Detect lines"}</button>}
        <button
          type="button"
          disabled={!diff.sameRows}
          className={btn + "bg-amber-800/70"}
          title={diff.sameRows
            ? "The text is right: copy each row of the text into its line"
            : "Different number of rows: use Detect lines"}
          onClick={() => edit((d) => textIntoLines(d, id))}
        >Copy text into lines</button>
        <button
          type="button"
          className={btn + "bg-amber-800/70"}
          title="The lines are right: make the text match them (what only the text has is dropped)"
          onClick={useLines}
        >Use the lines</button>
      </div>
      {msg && <div className="mt-1 whitespace-pre-wrap text-amber-300">⚠ {msg}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function elementText(el: LayoutElement, doc: Doc): string {
  // derived types: show the data the server actually keeps
  if (el.type === "table") return tableText(doc, el);
  // a region made of OCR lines: one row per line, so editing keeps the lines
  // (its own text joins them with a space and would merge them into one)
  if (el.type !== "heading" && el.type !== "list") {
    const own = doc.lines
      .filter((ln) => ln.block_id === el.id)
      .sort((a, b) => a.line_in_block - b.line_in_block);
    if (own.length > 1) return own.map((ln) => ln.text).join("\n");
  }
  const own = "text" in el ? ((el as any).text as string) : "";
  if (own && own.length > 0) return own;
  return doc.lines
    .filter((ln) => ln.block_id === el.id)
    .sort((a, b) => a.line_in_block - b.line_in_block)
    .map((ln) => ln.text)
    .join("\n");
}

function lineCountFor(el: LayoutElement, doc: Doc): number {
  return doc.lines.filter((ln) => ln.block_id === el.id).length;
}

// ---------------------------------------------------------------------------
// Selection header — always visible, tells the user exactly what is selected
// ---------------------------------------------------------------------------

function SelectionHeader({ doc }: { doc: Doc }) {
  const selection = useStore((s) => s.selection);
  if (!selection) return null;

  let color = "#64748b";
  let kindLabel = "?";
  let idLabel = "";
  let subLabel = "";

  if (selection.kind === "layout") {
    const el = doc.layout.find((e) => e.id === selection.id);
    color = el ? typeColor(el.type) : color;
    kindLabel = "layout";
    idLabel = `#${selection.id}`;
    subLabel = el ? el.type : "missing";
  } else if (selection.kind === "line") {
    color = "#22d3ee";
    kindLabel = "line";
    idLabel = `#${selection.index}`;
    subLabel = doc.lines[selection.index]?.role ?? "no role";
  } else if (selection.kind === "cell") {
    color = "#ef4444";
    kindLabel = "cell";
    idLabel = `t${selection.tableIndex} / c${selection.cellId}`;
    const c = doc.tables[selection.tableIndex]?.cells.find(
      (x) => x.id === selection.cellId,
    );
    subLabel = c ? (c.is_header ? "header" : "body") : "missing";
  } else if (selection.kind === "field") {
    color = "#38bdf8";
    kindLabel = "field";
    idLabel = `el ${selection.elementId} / #${selection.fieldIndex}`;
    const el = doc.layout.find((e) => e.id === selection.elementId);
    if (el && el.type === "form_field") {
      subLabel = el.fields[selection.fieldIndex]?.field_type ?? "";
    }
  } else if (selection.kind === "extraRegion") {
    color = "#a78bfa";
    kindLabel = "extra";
    idLabel = `#${selection.index}`;
    subLabel = doc.extra_regions[selection.index]?.type ?? "";
  }

  return (
    <div className="mb-3 flex items-center gap-2 rounded border border-slate-700 bg-slate-800/60 px-2 py-1.5">
      <span
        className="inline-block h-2.5 w-2.5 rounded-sm"
        style={{ background: color }}
      />
      <span className="text-[10px] uppercase tracking-wider text-slate-400">
        {kindLabel}
      </span>
      <span className="font-mono text-[11px] text-slate-100">{idLabel}</span>
      {subLabel && (
        <span className="ml-auto font-mono text-[10px] text-slate-400">
          {subLabel}
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Top-level dispatcher
// ---------------------------------------------------------------------------

export function PropertiesEditor({ doc }: { doc: Doc }) {
  const selection = useStore((s) => s.selection);

  if (!selection) {
    return (
      <div className="text-xs text-slate-400">
        Select an element on the canvas or in the Tree panel to edit it.
      </div>
    );
  }

  return (
    <div>
      <SelectionHeader doc={doc} />
      {selection.kind !== "layout" && <ParentRegionBar doc={doc} />}
      {selection.kind === "layout" && (
        (() => {
          const el = doc.layout.find((e) => e.id === selection.id);
          if (!el) return <div className="text-xs text-rose-400">missing element</div>;
          return <LayoutEditor doc={doc} el={el} />;
        })()
      )}
      {selection.kind === "line" && (
        <LineEditor doc={doc} index={selection.index} />
      )}
      {selection.kind === "cell" && (
        <CellEditor
          doc={doc}
          tableIndex={selection.tableIndex}
          cellId={selection.cellId}
        />
      )}
      {selection.kind === "field" && (
        <FieldEditor doc={doc} elementId={selection.elementId} />
      )}
      {selection.kind === "extraRegion" && (
        <ExtraRegionEditor doc={doc} index={selection.index} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Layout element editor
// ---------------------------------------------------------------------------

/**
 * Change a layout element's type in ONE undoable edit, filling in the
 * fields the new type needs and cleaning up the ones it doesn't.
 */
function applyTypeChange(d: Doc, id: number, next: string) {
  const target = d.layout.find((e) => e.id === id) as any;
  if (!target || target.type === next) return;
  const prev = target.type as string;

  // leaving heading: drop the "## " prefix from the text
  if (prev === "heading" && typeof target.text === "string") {
    target.text = target.text.replace(/^#+\s*/, "");
  }
  // leaving figure: drop the placeholder text
  if (prev === "figure" && target.text === "<figure/>") target.text = "";

  target.type = next;

  if (next === "heading") {
    if (target.level == null) target.level = 2;
    target.text = "#".repeat(target.level) + " " + (target.text ?? "");
  }
  if (next === "figure") {
    if (target.subtype == null) target.subtype = "photo";
    if (!target.text) target.text = "<figure/>";
  }
  if (next === "form_field" && !Array.isArray(target.fields)) target.fields = [];
  if (next === "table") {
    const ti = target.table_index;
    if (typeof ti !== "number" || ti < 0 || d.tables[ti] == null) {
      target.table_index = d.tables.length;
      d.tables.push(emptyTable(target.bbox));
    }
  }
  if (next === "list") {
    if (!Array.isArray(target.items) || target.items.length === 0) {
      const txt = typeof target.text === "string" ? target.text : "";
      target.items = listItemsFromText(txt);
    }
    if (!target.list_style) target.list_style = "bullet";
    syncListText(target, d);
  }
  if (next === "handwriting") {
    if (target.subtype == null) target.subtype = "note";
    target.handwritten = true;
  } else if (prev === "handwriting") {
    delete target.handwritten;
    if (next !== "figure") delete target.subtype;
  }
  if (next === "signature" && target.signer == null) target.signer = "other";
  if (next === "stamp" && target.stamp_text == null) target.stamp_text = "";
  if (next === "logo") delete target.text;
  else if (target.text == null) target.text = "";

  // logo / signature / stamp never carry a reading order
  if (next === "logo" || next === "signature" || next === "stamp") {
    target.reading_order = null;
  }
}

// ---------------------------------------------------------------------------
// Checkbox / fingerprint <-> layout box
//
// In doc_v2 a checkbox is NOT a layout type: it lives in `extra_regions`
// as { type: "checkbox", checked, option, bbox, quad }. Picking "checkbox"
// in the Type dropdown therefore moves the box from `layout` into
// `extra_regions` (same position; its text becomes the option label), and a
// checkbox can be turned back into any layout type the same way.
// ---------------------------------------------------------------------------

/** Types that are extra regions, not layout boxes. */
const REGION_TYPES = ["checkbox", "fingerprint"] as const;
type RegionType = (typeof REGION_TYPES)[number];
const isRegionType = (t: string): t is RegionType =>
  (REGION_TYPES as readonly string[]).includes(t);

/** Plain text of a layout box (own text, else its lines), for the option label. */
function plainText(d: Doc, el: any): string {
  let t = typeof el.text === "string" ? el.text : "";
  if (el.type === "heading") t = t.replace(/^#+\s*/, "");
  if (el.type === "figure" && t === "<figure/>") t = "";
  if (!t.trim()) {
    t = d.lines
      .filter((ln) => ln.block_id === el.id)
      .sort((a, b) => a.line_in_block - b.line_in_block)
      .map((ln) => ln.text)
      .join(" ");
  }
  return t.replace(/\s+/g, " ").trim();
}

/** Move layout box `id` into extra_regions as a checkbox/fingerprint. Returns its index. */
function layoutToRegion(d: Doc, id: number, kind: RegionType): number {
  const el = d.layout.find((e) => e.id === id) as any;
  if (!el) return -1;
  const bbox = [...el.bbox] as BBox;
  const quad = (el.quad ?? [
    [bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]],
  ]).map((p: number[]) => [p[0], p[1]]);
  const region: any = kind === "checkbox"
    ? { type: "checkbox", checked: false, option: plainText(d, el), bbox, quad }
    : { type: "fingerprint", hand: "right", bbox, quad };
  d.layout = d.layout.filter((e) => e.id !== id);
  d.lines = d.lines.filter((ln) => ln.block_id !== id);
  d.extra_regions.push(region);
  return d.extra_regions.length - 1;
}

/** Turn extra region `index` into a layout box of `type`. Returns the new layout id. */
function regionToLayout(d: Doc, index: number, type: string): number | null {
  const er = d.extra_regions[index] as any;
  if (!er) return null;
  const id = addLayoutElement(d, type, [...er.bbox] as BBox);
  const el = d.layout.find((e) => e.id === id) as any;
  if (el) {
    el.quad = er.quad.map((p: number[]) => [p[0], p[1]]);
    if (er.type === "checkbox" && er.option && "text" in el && type !== "figure") {
      el.text = type === "heading" ? `## ${er.option}` : er.option;
    }
  }
  d.extra_regions.splice(index, 1);
  return id;
}

/** Switch a region between checkbox and fingerprint, in place. */
function regionToRegion(d: Doc, index: number, kind: RegionType) {
  const er = d.extra_regions[index] as any;
  if (!er || er.type === kind) return;
  d.extra_regions[index] = kind === "checkbox"
    ? { type: "checkbox", checked: false, option: "", bbox: er.bbox, quad: er.quad }
    : { type: "fingerprint", hand: "right", bbox: er.bbox, quad: er.quad };
}

/** Type <select> options: layout types, then the region types. */
function TypeOptions({ layoutTypes, current }: { layoutTypes: string[]; current: string }) {
  const list = withCurrent(layoutTypes, isRegionType(current) ? null : current)
    .filter((t) => !isRegionType(t));
  return (
    <>
      <optgroup label="Layout box">
        {list.map((t) => <option key={t} value={t}>{prettyType(t)}</option>)}
      </optgroup>
      <optgroup label="Mark (extra region)">
        <option value="checkbox">checkbox ☐</option>
        <option value="fingerprint">fingerprint</option>
      </optgroup>
    </>
  );
}

/**
 * Shown when a line / cell / field is selected: names the region it sits
 * in and lets you change that region's type or jump to it for resizing.
 */
function ParentRegionBar({ doc }: { doc: Doc }) {
  const selection = useStore((s) => s.selection);
  const setSelection = useStore((s) => s.setSelection);
  const edit = useStore((s) => s.edit);
  const types = useProjectTypes();
  const pid = parentLayoutOf(doc, selection);
  const parent = pid != null ? doc.layout.find((e) => e.id === pid) : undefined;
  if (!parent) return null;

  return (
    <div className="mb-3 rounded border border-slate-700 bg-slate-800/40 p-2 text-xs">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-wider text-slate-400">
          inside region
        </span>
        <span className="font-mono text-[11px] text-slate-200">#{parent.id}</span>
        <button
          onClick={() => setSelection({ kind: "layout", id: parent.id })}
          className="ml-auto rounded bg-sky-700 px-2 py-0.5 text-[10px] text-white hover:bg-sky-600"
          title="Select the region to resize / move it (Esc then click also works)"
        >
          Select region
        </button>
      </div>
      <select
        value={parent.type}
        onChange={(e) => edit((d) => applyTypeChange(d, parent.id, e.target.value))}
        className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
        title="Region type"
      >
        {withCurrent(types.layout_types, parent.type).map((t) => (
          <option key={t} value={t}>{prettyType(t)}</option>
        ))}
      </select>
    </div>
  );
}

function LayoutEditor({ doc, el }: { doc: Doc; el: LayoutElement }) {
  const edit = useStore((s) => s.edit);
  const types = useProjectTypes();
  const setSelection = useStore((s) => s.setSelection);


  function changeType(next: string) {
    if (isRegionType(next)) {
      let index = -1;
      edit((d) => { index = layoutToRegion(d, el.id, next); });
      if (index >= 0) setSelection({ kind: "extraRegion", index });
      return;
    }
    edit((d) => applyTypeChange(d, el.id, next));
  }

  function remove() {
    edit((d) => {
      d.layout = d.layout.filter((e) => e.id !== el.id);
      d.lines = d.lines.filter((ln) => ln.block_id !== el.id);
    });
    setSelection(null);
  }

  const joinedFromLines = lineCountFor(el, doc);
  const ownText = "text" in el ? ((el as any).text as string) : "";
  const showingJoined = !ownText && joinedFromLines > 0;

  return (
    <div className="space-y-3 text-xs">
      {/* Type — always first, always visible */}
      <Section label="type">
        <select
          value={el.type}
          onChange={(e) => changeType(e.target.value)}
          className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
        >
          <TypeOptions layoutTypes={types.layout_types} current={el.type} />
        </select>
        <div className="mt-1 text-[10px] text-slate-500">
          Changes propagate to the canvas and the tree.
        </div>
      </Section>

      <Section label="reading_order">
        <div className="flex gap-1">
        <input
          type="number"
          value={el.reading_order ?? ""}
          disabled={
            el.type === "logo" ||
            el.type === "signature" ||
            el.type === "stamp"
          }
          onChange={(e) => {
            const v = e.target.value === "" ? null : Number(e.target.value);
            edit((d) => {
              const t = d.layout.find((x) => x.id === el.id);
              if (t) t.reading_order = v;
            });
          }}
          className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100 disabled:opacity-40"
        />
        <button
          disabled={isReadingOrderLocked(el.type)}
          onClick={() => edit((d) => appendReadingOrder(d, el.id))}
          className="shrink-0 rounded bg-sky-700 px-2 text-[10px] text-white hover:bg-sky-600 disabled:opacity-40"
          title="Give this box the next number (moves it to the end)"
        >
          next #{nextReadingOrder(doc)}
        </button>
        <button
          disabled={el.reading_order == null}
          onClick={() => edit((d) => removeFromReadingOrder(d, el.id))}
          className="shrink-0 rounded bg-slate-800 px-2 text-[10px] text-slate-300 hover:bg-slate-700 disabled:opacity-40"
          title="Remove from reading order and renumber the rest"
        >
          ×
        </button>
        </div>
        <div className="mt-1 text-[10px] text-slate-500">
          Faster: pick the Reading order tool (R) and click boxes in order.
        </div>
      </Section>

      <Section label="bbox (px)">
        <BBoxEditor
          bbox={el.bbox}
          onChange={(b) => edit((d) => resizeLayoutElement(d, el.id, b))}
        />
        <div className="mt-1 text-[10px] text-slate-500">
          Editing bbox scales all child lines, form fields, and table cells
          with the region. Corner handles on the canvas do the same.
        </div>
      </Section>

      {!NO_TEXT_BOX.has(el.type) && (
        <Section label={el.type === "table" ? "table html" : "text"}>
          <DraftTextarea
            key={el.id}
            value={elementText(el, doc)}
            onChange={(v) => edit((d) => setElementText(d, el.id, v))}
            rows={el.type === "table" ? 8 : 4}
          />
          {TEXT_HINT[el.type] && (
            <div className="mt-1 text-[10px] text-slate-500">{TEXT_HINT[el.type]}</div>
          )}
          <TextLinesMismatch doc={doc} id={el.id} edit={edit} />
          {showingJoined && !TEXT_HINT[el.type] && (
            <div className="mt-1 text-[10px] text-slate-500">
              Showing text joined from {joinedFromLines} line
              {joinedFromLines === 1 ? "" : "s"}. Editing here rewrites
              those lines.
            </div>
          )}
        </Section>
      )}

      {el.type === "heading" && (
        <Section label={`level (${el.level})`}>
          <div className="flex gap-1">
            {[1, 2, 3, 4, 5, 6].map((n) => (
              <button
                key={n}
                onClick={() => edit((d) => {
                  const t = d.layout.find((x) => x.id === el.id) as any;
                  if (!t) return;
                  t.level = n;
                  if ("text" in t) {
                    const plain = (t.text as string).replace(/^#+\s*/, "");
                    t.text = "#".repeat(n) + " " + plain;
                  }
                })}
                className={
                  "flex-1 rounded py-0.5 text-[11px] " +
                  (el.level === n
                    ? "bg-sky-600 text-white"
                    : "bg-slate-800 text-slate-300 hover:bg-slate-700")
                }
              >
                {n}
              </button>
            ))}
          </div>
        </Section>
      )}

      {el.type === "figure" && (
        <Section label="subtype">
          <select
            value={el.subtype}
            onChange={(e) => edit((d) => {
              const t = d.layout.find((x) => x.id === el.id) as any;
              if (t) t.subtype = e.target.value;
            })}
            className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
          >
            {["photo", "chart", "diagram", "graph", "map", "other"].map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Section>
      )}

      {el.type === "handwriting" && (
        <Section label="subtype">
          <select
            value={(el as any).subtype ?? "note"}
            onChange={(e) => edit((d) => {
              const t = d.layout.find((x) => x.id === el.id) as any;
              if (t) { t.subtype = e.target.value; t.handwritten = true; }
            })}
            className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
          >
            {["note", "annotation", "correction", "number", "date", "name", "other"].map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Section>
      )}

      {el.type === "signature" && (
        <Section label="signer">
          <select
            value={el.signer}
            onChange={(e) => edit((d) => {
              const t = d.layout.find((x) => x.id === el.id) as any;
              if (t) t.signer = e.target.value;
            })}
            className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
          >
            {["applicant", "bank_officer", "witness", "other"].map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Section>
      )}

      {el.type === "stamp" && (
        <Section label="stamp_text">
          <DevanagariTextarea
            value={el.stamp_text}
            onChange={(v) => edit((d) => {
              const t = d.layout.find((x) => x.id === el.id) as any;
              if (t) t.stamp_text = v;
            })}
            rows={2}
          />
        </Section>
      )}

      {el.type === "list" && <ListEditor doc={doc} el={el} />}
      {el.type === "form_field" && <FormFieldEditor el={el} />}

      {el.type === "table" && (
        <Section label="suggested layout">
          <TableSuggestPanel doc={doc} el={el} />
        </Section>
      )}

      {el.type === "table" && (
        <Section label="table OCR">
          <TableOcrButton
            documentId={doc.document_id}
            tableElementId={el.id}
            quad={el.quad}
          />
          <div className="mt-1 text-[10px] text-slate-500">
            Run table OCR: finds the cells and fills their text from the OCR
            lines (replaces the current cells). Fit cells to scan: keeps the
            Table HTML you typed and snaps its cells onto the ruling lines
            printed on the scan — done automatically when your HTML adds or
            removes cells. If its rows / columns don't match the printed
            table, the table is left as it is and a warning is shown.
          </div>
        </Section>
      )}

      <Section label="OCR">
        <OcrRegionButton
          documentId={doc.document_id}
          quad={el.quad}
          onText={(text) => edit((d) => setElementText(d, el.id, text))}
        />
      </Section>

      <button
        onClick={remove}
        className="w-full rounded bg-rose-900/50 py-1 text-[11px] text-rose-200 hover:bg-rose-800/60"
      >
        Delete element
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// List item editor
// ---------------------------------------------------------------------------

function ListEditor({ doc, el }: { doc: Doc; el: any }) {
  const edit = useStore((s) => s.edit);
  const setSelection = useStore((s) => s.setSelection);
  // every change to the items keeps children links, lines and text in step
  const change = (fn: (t: any, d: Doc) => void) => edit((d) => {
    const t = d.layout.find((x) => x.id === el.id) as any;
    if (!t) return;
    fn(t, d);
    afterListChange(d, el.id);
  });
  const candidates = listChildCandidates(doc, el.id);
  const label = (id: number) => {
    const c = doc.layout.find((e) => e.id === id) as any;
    return c ? `#${id} ${prettyType(c.type)}` : `#${id} (missing)`;
  };

  return (
    <Section label={`list items (${el.items.length})`}>
      <select
        value={el.list_style}
        onChange={(e) => edit((d) => {
          const t = d.layout.find((x) => x.id === el.id) as any;
          if (t) t.list_style = e.target.value;
        })}
        className="mb-2 w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
      >
        {["dev", "latin", "roman", "bullet"].map((s) => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>

      <ul className="space-y-2">
        {el.items.map((item: any, i: number) => (
          <li key={i} className="space-y-1">
            <div className="flex gap-1">
              <input
                value={item.marker}
                title="marker"
                onChange={(e) => change((t) => { t.items[i].marker = e.target.value; })}
                className="w-10 rounded bg-slate-800 px-1 text-center font-mono text-[11px] text-slate-100"
              />
              <input
                type="number"
                min={0}
                value={item.level}
                title="level (indent)"
                onChange={(e) => change((t) => { t.items[i].level = Math.max(0, Number(e.target.value)); })}
                className="w-10 rounded bg-slate-800 px-1 text-center text-[11px] text-slate-100"
              />
              <input
                value={item.text}
                onChange={(e) => change((t) => { t.items[i].text = e.target.value; })}
                className="flex-1 rounded bg-slate-800 px-2 py-0.5 font-deva text-[11px] text-slate-100"
              />
              <button
                title="remove item"
                onClick={() => change((t) => { t.items.splice(i, 1); })}
                className="rounded bg-slate-800 px-1 text-[10px] text-slate-400 hover:text-rose-300"
              >
                ×
              </button>
            </div>

            {/* boxes that belong to this item (a note, a figure, …) */}
            <div className="flex flex-wrap items-center gap-1 pl-12">
              {(item.children ?? []).map((cid: number) => (
                <span key={cid} className="flex items-center gap-1 rounded bg-slate-700 px-1.5 py-0.5 text-[10px] text-slate-200">
                  <button
                    title="select this box"
                    onClick={() => setSelection({ kind: "layout", id: cid })}
                    className="hover:underline"
                  >
                    ↳ {label(cid)}
                  </button>
                  <button
                    title="detach from this item"
                    onClick={() => edit((d) => detachFromList(d, el.id, cid))}
                    className="text-slate-400 hover:text-rose-300"
                  >
                    ✕
                  </button>
                </span>
              ))}
              {candidates.some((c) => !(item.children ?? []).includes(c.id)) && (
                <select
                  value=""
                  onChange={(e) => {
                    const cid = Number(e.target.value);
                    if (Number.isFinite(cid)) edit((d) => attachToListItem(d, el.id, i, cid));
                  }}
                  className="rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-400"
                >
                  <option value="">+ attach box…</option>
                  {candidates
                    .filter((c) => !(item.children ?? []).includes(c.id))
                    .map((c) => (
                      <option key={c.id} value={c.id}>
                        {label(c.id)}{c.parent_id != null ? ` (now in item ${c.parent_item})` : ""}
                      </option>
                    ))}
                </select>
              )}
            </div>
          </li>
        ))}
      </ul>

      <button
        onClick={() => change((t) => { t.items.push({ marker: "•", level: 0, text: "" }); })}
        className="mt-2 w-full rounded bg-slate-800 py-0.5 text-[10px] text-slate-400 hover:bg-slate-700"
      >
        + item
      </button>
      {candidates.length > 0 && (
        <button
          onClick={() => edit((d) => { autoAttachListChildren(d, el.id); })}
          title="Each box inside the list goes to the item it sits under"
          className="mt-1 w-full rounded bg-sky-800 py-0.5 text-[10px] text-sky-100 hover:bg-sky-700"
        >
          Attach the {candidates.length} box{candidates.length === 1 ? "" : "es"} inside this list to their items
        </button>
      )}
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Form field editor
// ---------------------------------------------------------------------------

function FormFieldEditor({ el }: { el: any }) {
  const edit = useStore((s) => s.edit);
  const types = useProjectTypes();
  const setSelection = useStore((s) => s.setSelection);

  return (
    <div className="space-y-2">
      <Section label={`fields (${el.fields.length})`}>
        {el.fields.map((f: FormFieldEntry, i: number) => (
          <div key={i} className="mb-2 rounded border border-slate-700 p-2">
            <div className="mb-1 flex items-center justify-between">
              <button
                onClick={() =>
                  setSelection({ kind: "field", elementId: el.id, fieldIndex: i })
                }
                className="font-mono text-[10px] text-sky-400 hover:underline"
              >
                field[{i}]
              </button>
              <button
                onClick={() => edit((d) => {
                  const t = d.layout.find((x) => x.id === el.id) as any;
                  if (t) t.fields.splice(i, 1);
                })}
                className="text-[10px] text-slate-400 hover:text-rose-300"
              >
                remove
              </button>
            </div>

            <div className="space-y-1">
              <Labeled label="label">
                <input
                  value={f.label?.text ?? ""}
                  placeholder={f.label ? "" : "(no question)"}
                  onChange={(e) => edit((d) => {
                    const t = d.layout.find((x) => x.id === el.id) as any;
                    if (!t) return;
                    if (t.fields[i].label) t.fields[i].label.text = e.target.value;
                    else t.fields[i].label = { text: e.target.value };
                  })}
                  className="w-full rounded bg-slate-800 px-2 py-0.5 font-deva text-[11px] text-slate-100"
                />
              </Labeled>

              <Labeled label="field_type">
                <select
                  value={f.field_type}
                  onChange={(e) => edit((d) => {
                    const t = d.layout.find((x) => x.id === el.id) as any;
                    if (t) t.fields[i].field_type = e.target.value;
                  })}
                  className="w-full rounded bg-slate-800 px-2 py-0.5 text-[11px] text-slate-100"
                >
                  {withCurrent(types.field_types, f.field_type).map((t) => (
                    <option key={t} value={t}>{prettyType(t)}</option>
                  ))}
                </select>
              </Labeled>

              {f.field_type !== "checkbox" && !f.value && (
                <div className="text-[10px] text-slate-500">
                  no value area — draw one over the blank / leader
                </div>
              )}

              {f.value && (
                <>
                  <DevanagariTextarea
                    value={f.value.text}
                    onChange={(v) => edit((d) => {
                      const t = d.layout.find((x) => x.id === el.id) as any;
                      t.fields[i].value.text = v;
                      t.fields[i].value.empty = v === "";
                    })}
                    rows={2}
                  />
                  <label className="flex items-center gap-2 text-[10px] text-slate-400">
                    <input
                      type="checkbox"
                      checked={f.value.handwritten}
                      onChange={(e) => edit((d) => {
                        const t = d.layout.find((x) => x.id === el.id) as any;
                        t.fields[i].value.handwritten = e.target.checked;
                      })}
                    />
                    handwritten
                  </label>
                </>
              )}

              {f.field_type === "checkbox" && (
                <CheckboxOptionsEditor elementId={el.id} fieldIndex={i} field={f} />
              )}
            </div>
          </div>
        ))}

        <button
          onClick={() => edit((d) => {
            const t = d.layout.find((x) => x.id === el.id) as any;
            // label text only: its box is drawn or taken from a line, never
            // the whole region's box
            t.fields.push({ label: { text: "" }, field_type: "text", value: null });
          })}
          className="w-full rounded bg-slate-800 py-0.5 text-[10px] text-slate-400 hover:bg-slate-700"
        >
          + field
        </button>
      </Section>
    </div>
  );
}

const NEXT_CHECKED = new Map<boolean | null, boolean | null>([[null, true], [true, false], [false, null]]);
const CHECKED_MARK = (c: boolean | null) => (c === true ? "yes" : c === false ? "no" : "—");

/**
 * The options of a checkbox field. An option's word is a line of the page
 * (edit the line to change it); new options are drawn with the Option tool.
 */
function CheckboxOptionsEditor({
  elementId, fieldIndex, field,
}: { elementId: number; fieldIndex: number; field: FormFieldEntry }) {
  const edit = useStore((s) => s.edit);
  const doc = useStore((s) => s.doc);
  const lineText = (id: number) => doc?.lines.find((ln) => ln.id === id)?.text;
  const options: FieldOption[] = field.options ?? [];
  const fieldOf = (d: Doc) => (d.layout.find((x) => x.id === elementId) as any)?.fields?.[fieldIndex] as
    FormFieldEntry | undefined;
  return (
    <ul className="space-y-1">
      <li>
        <label className="flex items-center gap-2 text-[10px] text-slate-400">
          <input
            type="checkbox"
            checked={!!field.single_choice}
            onChange={(e) => edit((d) => {
              const f = fieldOf(d);
              if (f) f.single_choice = e.target.checked;
            })}
          />
          single choice
        </label>
      </li>
      {options.map((opt, oi) => (
        <li key={`${opt.line_id}-${oi}`} className="flex items-center gap-1">
          <button
            title="checked: — (not marked) → yes → no"
            disabled={opt.tick_cell_id != null}
            onClick={() => edit((d) => {
              const o = fieldOf(d)?.options?.[oi];
              if (o) o.checked = NEXT_CHECKED.get(o.checked ?? null) ?? null;
            })}
            className="w-8 rounded bg-slate-800 px-1 text-[10px] text-slate-300 disabled:opacity-50"
          >
            {CHECKED_MARK(opt.checked ?? null)}
          </button>
          <span className="flex-1 truncate font-deva text-[11px] text-slate-100">
            {lineText(opt.line_id) ?? <span className="text-rose-400">line {opt.line_id} missing</span>}
          </span>
          <span className="text-[10px] text-slate-500">
            {opt.box ? "□" : opt.tick_cell_id != null ? `cell ${opt.tick_cell_id}` : ""}
          </span>
          <button
            onClick={() => edit((d) => {
              const f = fieldOf(d);
              if (!f?.options) return;
              const [gone] = f.options.splice(oi, 1);
              // the word stays a line of the page, but is no longer an option
              const ln = d.lines.find((l) => l.id === gone?.line_id);
              if (ln && ln.role === "option") delete ln.role;
            })}
            className="rounded bg-slate-800 px-1 text-[10px] text-slate-400 hover:text-rose-300"
          >
            ×
          </button>
        </li>
      ))}
      {options.length === 0 && (
        <li className="text-[10px] text-slate-500">no options yet — draw them with the Option tool</li>
      )}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Line editor
// ---------------------------------------------------------------------------

const LINE_ROLES = ["label", "value", "option"] as const;

function LineEditor({ doc, index }: { doc: Doc; index: number }) {
  const edit = useStore((s) => s.edit);
  const ln = doc.lines[index];
  if (!ln) return <div className="text-xs text-rose-400">missing line</div>;

  return (
    <div className="space-y-3 text-xs">
      <Section label="role">
        <select
          value={ln.role ?? ""}
          onChange={(e) => edit((d) => {
            const v = e.target.value as any;
            if (v === "") delete d.lines[index].role;
            else d.lines[index].role = v;
          })}
          className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
        >
          <option value="">— none —</option>
          {LINE_ROLES.map((r) => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
      </Section>

      <Section label="text">
        <DevanagariTextarea
          value={ln.text}
          onChange={(v) => edit((d) => {
            const ln = d.lines[index];
            ln.text = v;
            // a cell's text is made from all of its lines
            if (ln.table_index != null && ln.cell_id != null) syncCellFromLines(d, ln.table_index, ln.cell_id);
            // …and a region's (or table's) text from its lines
            syncRegionTextFromLines(d, ln.block_id);
          })}
          rows={3}
        />
      </Section>

      <Section label="is_heading">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={ln.is_heading}
            onChange={(e) => edit((d) => { d.lines[index].is_heading = e.target.checked; })}
          />
          <span className="text-slate-400">heading line</span>
        </label>
      </Section>

      <Section label="handwritten">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={ln.handwritten === true}
            onChange={(e) => edit((d) => {
              const ln = d.lines[index];
              if (e.target.checked) ln.handwritten = true;
              else delete ln.handwritten;
              // table line: the cell is handwritten if any of its lines is
              if (ln.table_index != null && ln.cell_id != null) syncCellFromLines(d, ln.table_index, ln.cell_id);
            })}
          />
          <span className="text-slate-400">handwritten</span>
        </label>
      </Section>

      <Section label="font_size">
        <input
          type="number"
          step="0.5"
          value={ln.font_size}
          onChange={(e) =>
            edit((d) => { d.lines[index].font_size = Number(e.target.value); })
          }
          className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
        />
      </Section>

      <button
        onClick={() => edit((d) => {
          const [gone] = d.lines.splice(index, 1);
          if (!gone) return;
          if (gone.table_index != null && gone.cell_id != null) syncCellFromLines(d, gone.table_index, gone.cell_id);
          syncRegionTextFromLines(d, gone.block_id);
        })}
        className="w-full rounded bg-rose-900/50 py-1 text-[11px] text-rose-200 hover:bg-rose-800/60"
      >
        Delete line
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cell editor
// ---------------------------------------------------------------------------

function CellEditor({
  doc, tableIndex, cellId,
}: { doc: Doc; tableIndex: number; cellId: number }) {
  const edit = useStore((s) => s.edit);
  const t = doc.tables[tableIndex];
  const cell = t?.cells.find((c) => c.id === cellId);
  // keep the table's lines (and, for span changes, ruling lines) in step
  const syncThis = (d: Doc, ruling = false) => {
    const el = d.layout.find((e: any) => e.type === "table" && e.table_index === tableIndex);
    if (el) syncTable(d, el.id, { ruling });
  };
  if (!cell) return <div className="text-xs text-rose-400">missing cell</div>;

  return (
    <div className="space-y-3 text-xs">
      <Section label="cell role">
        <select
          value={cell.is_header ? "header" : "body"}
          onChange={(e) => edit((d) => {
            const c = d.tables[tableIndex].cells.find((x) => x.id === cellId)!;
            c.is_header = e.target.value === "header";
          })}
          className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
        >
          <option value="body">body</option>
          <option value="header">header</option>
        </select>
      </Section>

      <Section label="text">
        <DevanagariTextarea
          value={cell.text}
          onChange={(v) => edit((d) => {
            const c = d.tables[tableIndex].cells.find((x) => x.id === cellId)!;
            c.text = v;
            c.empty = v.trim() === "";
            syncThis(d);
          })}
          rows={2}
        />
      </Section>

      <Section label="cell flags">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={cell.handwritten}
            onChange={(e) => edit((d) => {
              d.tables[tableIndex].cells.find((x) => x.id === cellId)!.handwritten = e.target.checked;
              syncThis(d);
            })}
          />
          <span>handwritten</span>
        </label>
      </Section>

      <Section label={`span (row ${cell.rowspan} × col ${cell.colspan})`}>
        <div className="flex gap-2">
          <label className="flex-1">
            rowspan
            <input
              type="number"
              min={1}
              value={cell.rowspan}
              onChange={(e) => edit((d) => {
                d.tables[tableIndex].cells.find((x) => x.id === cellId)!.rowspan = Math.max(1, Number(e.target.value));
                syncThis(d, true);
              })}
              className="mt-0.5 w-full rounded bg-slate-800 px-1 text-slate-100"
            />
          </label>
          <label className="flex-1">
            colspan
            <input
              type="number"
              min={1}
              value={cell.colspan}
              onChange={(e) => edit((d) => {
                d.tables[tableIndex].cells.find((x) => x.id === cellId)!.colspan = Math.max(1, Number(e.target.value));
                syncThis(d, true);
              })}
              className="mt-0.5 w-full rounded bg-slate-800 px-1 text-slate-100"
            />
          </label>
        </div>
      </Section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Field editor — direct selection of a form field, shows field_type + role
// ---------------------------------------------------------------------------

function FieldEditor({
  doc, elementId,
}: { doc: Doc; elementId: number }) {
  const edit = useStore((s) => s.edit);
  const types = useProjectTypes();
  const el = doc.layout.find((e) => e.id === elementId);
  const selection = useStore((s) => s.selection);
  if (!el || el.type !== "form_field") return null;

  const fieldIndex =
    selection?.kind === "field" ? selection.fieldIndex : 0;
  const field = el.fields[fieldIndex];
  if (!field) return null;

  return (
    <div className="space-y-3 text-xs">
      <Section label="field_type">
        <select
          value={field.field_type}
          onChange={(e) => edit((d) => {
            const t = d.layout.find((x) => x.id === elementId);
            if (!t || t.type !== "form_field") return;
            t.fields[fieldIndex].field_type = e.target.value;
          })}
          className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
        >
          {withCurrent(types.field_types, field.field_type).map((t) => (
            <option key={t} value={t}>{prettyType(t)}</option>
          ))}
        </select>
      </Section>

      <Section label="label text">
        <input
          value={field.label?.text ?? ""}
          placeholder={field.label ? "" : "(no question)"}
          onChange={(e) => edit((d) => {
            const t = d.layout.find((x) => x.id === elementId);
            if (!t || t.type !== "form_field") return;
            const f = t.fields[fieldIndex];
            if (f.label) f.label.text = e.target.value;
            else f.label = { text: e.target.value };
          })}
          className="w-full rounded bg-slate-800 px-2 py-1 font-deva text-slate-100"
        />
      </Section>

      {field.value && (
        <>
          <Section label="value text">
            <DevanagariTextarea
              value={field.value.text}
              onChange={(v) => edit((d) => {
                const t = d.layout.find((x) => x.id === elementId);
                if (!t || t.type !== "form_field") return;
                const vv = t.fields[fieldIndex].value;
                if (vv) {
                  vv.text = v;
                  vv.empty = v === "";
                }
              })}
              rows={2}
            />
          </Section>
          <label className="flex items-center gap-2 text-[10px] text-slate-400">
            <input
              type="checkbox"
              checked={field.value.handwritten}
              onChange={(e) => edit((d) => {
                const t = d.layout.find((x) => x.id === elementId);
                if (!t || t.type !== "form_field") return;
                const vv = t.fields[fieldIndex].value;
                if (vv) vv.handwritten = e.target.checked;
              })}
            />
            handwritten
          </label>
        </>
      )}

      {field.field_type === "checkbox" && (
        <Section label="options">
          <CheckboxOptionsEditor elementId={elementId} fieldIndex={fieldIndex} field={field} />
        </Section>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Extra region editor
// ---------------------------------------------------------------------------

function ExtraRegionEditor({ doc, index }: { doc: Doc; index: number }) {
  const edit = useStore((s) => s.edit);
  const setSelection = useStore((s) => s.setSelection);
  const types = useProjectTypes();
  const er = doc.extra_regions[index];
  if (!er) return null;

  function changeType(next: string) {
    if (isRegionType(next)) {
      edit((d) => regionToRegion(d, index, next));
      return;
    }
    let id: number | null = null;
    edit((d) => { id = regionToLayout(d, index, next); });
    if (id != null) setSelection({ kind: "layout", id });
  }

  return (
    <div className="space-y-3 text-xs">
      <Section label="type">
        <select
          value={er.type}
          onChange={(e) => changeType(e.target.value)}
          className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
        >
          <TypeOptions layoutTypes={types.layout_types} current={er.type} />
        </select>
        <div className="mt-1 text-[10px] text-slate-500">
          Checkboxes and fingerprints are saved in extra_regions, not layout.
          Pick a layout type to turn this back into a normal box.
        </div>
      </Section>

      {er.type === "checkbox" && (
        <>
          <Section label="checked">
            <label className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={er.checked}
                onChange={(e) => edit((d) => {
                  const x = d.extra_regions[index];
                  if (x && x.type === "checkbox") x.checked = e.target.checked;
                })}
                className="h-4 w-4"
              />
              <span className={er.checked ? "text-emerald-400" : "text-slate-400"}>
                {er.checked ? "☑ ticked" : "☐ not ticked"}
              </span>
            </label>
          </Section>
          <Section label="option (label text)">
            <input
              value={er.option}
              onChange={(e) => edit((d) => {
                const x = d.extra_regions[index];
                if (x && x.type === "checkbox") x.option = e.target.value;
              })}
              placeholder="e.g. पुरुष / Male"
              className="w-full rounded bg-slate-800 px-2 py-1 font-deva text-slate-100"
            />
          </Section>
        </>
      )}

      {er.type === "fingerprint" && (
        <Section label="hand">
          <select
            value={er.hand}
            onChange={(e) => edit((d) => {
              const x = d.extra_regions[index];
              if (x && x.type === "fingerprint") x.hand = e.target.value as any;
            })}
            className="w-full rounded bg-slate-800 px-2 py-1 text-slate-100"
          >
            <option value="left">left</option>
            <option value="right">right</option>
          </select>
        </Section>
      )}

      <button
        onClick={() => {
          edit((d) => { d.extra_regions.splice(index, 1); });
          setSelection(null);
        }}
        className="w-full rounded bg-rose-900/50 py-1 text-[11px] text-rose-200 hover:bg-rose-800/60"
      >
        Delete {er.type}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-0.5 text-[10px] uppercase tracking-wider text-slate-400">
        {label}
      </div>
      {children}
    </div>
  );
}

function Labeled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[10px] text-slate-500">{label}</div>
      {children}
    </div>
  );
}

/**
 * Number inputs for a bbox. Values are kept as a local draft while you
 * type and only applied on Enter / blur, so intermediate keystrokes
 * ("3" -> "35" -> "350") never squash the region and its children.
 */
function BBoxEditor({
  bbox,
  onChange,
}: {
  bbox: BBox;
  onChange: (b: BBox) => void;
}) {
  const fromBBox = (b: BBox) => ({
    x0: String(b[0]), y0: String(b[1]), x1: String(b[2]), y1: String(b[3]),
    w: String(Math.round(b[2] - b[0])), h: String(Math.round(b[3] - b[1])),
  });
  const [draft, setDraft] = useState(() => fromBBox(bbox));
  const key = bbox.join(",");
  useEffect(() => { setDraft(fromBBox(bbox)); /* eslint-disable-next-line */ }, [key]);

  type K = keyof typeof draft;

  function commit(changed: K) {
    const n = (k: K) => Number(draft[k]);
    let next: BBox;
    if (changed === "w" || changed === "h") {
      const w = n("w"), h = n("h");
      if (!Number.isFinite(w) || !Number.isFinite(h) || w < 1 || h < 1) {
        setDraft(fromBBox(bbox));
        return;
      }
      next = [bbox[0], bbox[1], bbox[0] + w, bbox[1] + h];
    } else {
      const vals = [n("x0"), n("y0"), n("x1"), n("y1")];
      if (!vals.every(Number.isFinite)) { setDraft(fromBBox(bbox)); return; }
      next = [
        Math.min(vals[0], vals[2]), Math.min(vals[1], vals[3]),
        Math.max(vals[0], vals[2]), Math.max(vals[1], vals[3]),
      ];
      if (next[2] - next[0] < 1 || next[3] - next[1] < 1) {
        setDraft(fromBBox(bbox));
        return;
      }
    }
    if (next.some((v, i) => v !== bbox[i])) onChange(next);
  }

  const input = (k: K, label: string) => (
    <label key={k} className="flex flex-col">
      <span className="text-[9px] text-slate-500">{label}</span>
      <input
        type="number"
        value={draft[k]}
        onChange={(e) => setDraft((dr) => ({ ...dr, [k]: e.target.value }))}
        onBlur={() => commit(k)}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          if (e.key === "Escape") setDraft(fromBBox(bbox));
        }}
        className="rounded bg-slate-800 px-1 py-0.5 font-mono text-[10px] text-slate-100"
      />
    </label>
  );

  return (
    <>
      <div className="grid grid-cols-4 gap-1">
        {input("x0", "x0")}
        {input("y0", "y0")}
        {input("x1", "x1")}
        {input("y1", "y1")}
      </div>
      <div className="mt-1 grid grid-cols-2 gap-1">
        {input("w", "width")}
        {input("h", "height")}
      </div>
      <div className="mt-0.5 text-[9px] text-slate-500">Enter or click away to apply</div>
    </>
  );
}
