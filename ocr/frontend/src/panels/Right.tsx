import { useEffect, useMemo, useRef, useState } from "react";
import {
  exportCurrentDoc,
  exportProject,
  exportProjectJsonl,
  planJsonlImport,
  applyJsonlImport,
  type JsonlPlanItem,
} from "../editor/exportJson";
import { getDocument } from "../api";
import { normalizeForExport } from "../editor/normalizeDoc";
import { useStore } from "../store";
import type { Doc, LayoutElement } from "../types";
import { typeColor, prettyType } from "../hooks/useProjectTypes";
import { PropertiesEditor } from "../editor/PropertiesEditor";
import { DocTypePicker } from "../editor/DocTypePicker";
import { DetectLinesButton } from "../editor/DetectLinesButton";

const TABS: {
  id: "properties" | "tree" | "readingOrder" | "pageMeta" | "json";
  label: string;
}[] = [
  { id: "properties", label: "Properties" },
  { id: "tree", label: "Tree" },
  { id: "readingOrder", label: "Reading order" },
  { id: "pageMeta", label: "Page meta" },
  { id: "json", label: "JSON" },
];

export function RightPanel() {
  const doc = useStore((s) => s.doc);
  const tab = useStore((s) => s.rightTab);
  const setTab = useStore((s) => s.setRightTab);

  return (
    <aside className="flex h-full w-96 flex-col border-l border-slate-800 bg-slate-900">
      <div className="flex border-b border-slate-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={
              "flex-1 px-2 py-2 text-[11px] font-medium transition-colors " +
              (tab === t.id
                ? "bg-slate-800 text-slate-100"
                : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200")
            }
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {!doc && <div className="text-xs text-slate-500">Loading…</div>}
        {doc && tab === "properties" && (
          <>
            <DetectLinesButton />
            <PropertiesEditor doc={doc} />
          </>
        )}
        {doc && tab === "tree" && <TreePanel doc={doc} />}
        {doc && tab === "readingOrder" && <ReadingOrderPanel doc={doc} />}
        {doc && tab === "pageMeta" && <PageMetaPanel doc={doc} />}
        {doc && tab === "json" && <JsonPanel doc={doc} />}
      </div>
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

function TreePanel({ doc }: { doc: Doc }) {
  const setSelection = useStore((s) => s.setSelection);
  const selection = useStore((s) => s.selection);

  return (
    <ul className="space-y-0.5 text-xs">
      {doc.layout.map((el) => {
        const color = typeColor(el.type);
        const isSelected =
          selection?.kind === "layout" && selection.id === el.id;
        return (
          <li key={el.id}>
            <button
              onClick={() => setSelection({ kind: "layout", id: el.id })}
              className={
                "flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-slate-800 " +
                (isSelected ? "bg-slate-800" : "")
              }
            >
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ background: color }}
              />
              <span className="font-mono text-slate-300">{el.id}</span>
              <span className="text-slate-200">{prettyType(el.type)}</span>
              {el.reading_order != null && (
                <span className="ml-auto text-[10px] text-slate-500">
                  ro={el.reading_order}
                </span>
              )}
            </button>
            <Children el={el} doc={doc} />
          </li>
        );
      })}
    </ul>
  );
}

function Children({ el, doc }: { el: LayoutElement; doc: Doc }) {
  const setSelection = useStore((s) => s.setSelection);

  const lineChildren = doc.lines.filter((ln) => ln.block_id === el.id);
  const lineNodes = lineChildren.map((ln, i) => (
    <li key={`ln-${i}`} className="ml-6">
      <button
        onClick={() =>
          setSelection({ kind: "line", index: doc.lines.indexOf(ln) })
        }
        className="w-full truncate rounded px-2 py-0.5 text-left font-mono text-[11px] text-slate-400 hover:bg-slate-800 hover:text-slate-200"
        title={ln.text}
      >
        {ln.role ? `${ln.role}: ` : ""}
        {ln.text}
      </button>
    </li>
  ));

  if (el.type === "form_field") {
    return (
      <ul>
        {el.fields.map((f, i) => (
          <li key={`f-${i}`} className="ml-6">
            <button
              onClick={() =>
                setSelection({ kind: "field", elementId: el.id, fieldIndex: i })
              }
              className="w-full truncate rounded px-2 py-0.5 text-left text-[11px] text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              title={`${f.label.text} — ${f.value?.text ?? "null"}`}
            >
              <span className="font-mono text-slate-500">{f.field_type}:</span>{" "}
              {f.label.text} {f.value ? `→ ${f.value.text}` : ""}
            </button>
          </li>
        ))}
      </ul>
    );
  }

  if (el.type === "list") {
    return (
      <ul>
        {el.items.map((item, i) => (
          <li
            key={`it-${i}`}
            className="ml-6 truncate px-2 py-0.5 text-[11px] text-slate-400"
          >
            {item.marker} {item.text}
          </li>
        ))}
      </ul>
    );
  }

  if (el.type === "table") {
    const t = doc.tables[el.table_index];
    if (!t) return null;
    return (
      <ul>
        {t.cells.map((c) => (
          <li key={`c-${c.id}`} className="ml-6">
            <button
              onClick={() =>
                setSelection({
                  kind: "cell",
                  tableIndex: el.table_index,
                  cellId: c.id,
                })
              }
              className="w-full truncate rounded px-2 py-0.5 text-left font-mono text-[11px] text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              title={c.text}
            >
              [{c.row},{c.col}] {c.text || "∅"}
            </button>
          </li>
        ))}
      </ul>
    );
  }

  return <ul>{lineNodes}</ul>;
}

// ---------------------------------------------------------------------------
// Reading order
// ---------------------------------------------------------------------------

function ReadingOrderPanel({ doc }: { doc: Doc }) {
  const setSelection = useStore((s) => s.setSelection);
  const ordered = useMemo(
    () =>
      [...doc.layout]
        .filter((e) => e.reading_order != null)
        .sort((a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0)),
    [doc.layout],
  );
  const nulls = useMemo(
    () => doc.layout.filter((e) => e.reading_order == null),
    [doc.layout],
  );

  return (
    <div className="space-y-3 text-xs">
      <ol className="space-y-0.5">
        {ordered.map((el) => (
          <li key={el.id}>
            <button
              onClick={() => setSelection({ kind: "layout", id: el.id })}
              className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-slate-800"
            >
              <span className="w-6 text-right font-mono text-slate-500">
                {el.reading_order}
              </span>
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ background: typeColor(el.type) }}
              />
              <span className="font-mono text-slate-400">{el.id}</span>
              <span className="text-slate-200">{prettyType(el.type)}</span>
            </button>
          </li>
        ))}
      </ol>

      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-400">
          No reading order (null) — locked
        </div>
        <ul className="space-y-0.5">
          {nulls.map((el) => (
            <li key={el.id}>
              <button
                onClick={() => setSelection({ kind: "layout", id: el.id })}
                className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-slate-800"
              >
                <span className="font-mono text-slate-500">{el.id}</span>
                <span
                  className="inline-block h-2 w-2 rounded-sm"
                  style={{ background: typeColor(el.type) }}
                />
                <span className="text-slate-200">{prettyType(el.type)}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page meta (with submit gate)
// ---------------------------------------------------------------------------

function PageMetaPanel({ doc }: { doc: Doc }) {

  return (
    <div className="space-y-3 text-xs">
      <Section title="Required before submit">
        <div>
          <div className="mb-1 text-slate-400">doc_type</div>
          <DocTypePicker />
          {doc.doc_type === "other" && (
            <div className="mt-1 text-[10px] text-rose-400">
              Must be set before submit.
            </div>
          )}
        </div>

        <button
          disabled={doc.doc_type === "other"}
          onClick={async () => {
            const { saveCurrent } = await import("../editor/save");
            if (useStore.getState().dirty && !(await saveCurrent())) return;
            await fetch(`/documents/${doc.document_id}/status`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ target: "submitted" }),
            });
          }}
          className={
            "w-full rounded px-2 py-1 text-[11px] font-medium " +
            (doc.doc_type !== "other"
              ? "bg-emerald-600 text-white hover:bg-emerald-500"
              : "bg-slate-800 text-slate-500")
          }
        >
          Submit for review
        </button>
      </Section>

      <Section title="Language & quality">
        <KV label="language" value={doc.language} />
        <KV label="digit_style" value={doc.digit_style} />
        <KV label="is_grayscale" value={String(doc.quality.is_grayscale)} />
        <KV label="is_binarized" value={String(doc.quality.is_binarized)} />
        <KV label="blur_sigma" value={fmt(doc.quality.blur_sigma)} />
        <KV label="noise_sigma" value={fmt(doc.quality.noise_sigma)} />
        <KV label="skew_deg" value={fmt(doc.quality.skew_deg)} />
        <KV label="jpeg_quality" value={fmt(doc.quality.jpeg_quality)} />
      </Section>

      <Section title="Counts">
        <KV label="line_count" value={String(doc.quality.line_count)} />
        <KV label="table_count" value={String(doc.quality.table_count)} />
        <KV label="figure_count" value={String(doc.quality.figure_count)} />
      </Section>

      <Section title="Full page text">
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-slate-800 p-2 font-mono text-[11px] text-slate-200">
          {doc.text}
        </pre>
      </Section>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5 rounded border border-slate-800 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-400">
        {title}
      </div>
      {children}
    </div>
  );
}

function fmt(v: number | null): string {
  return v == null ? "null" : String(v);
}

function KV({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <div className="text-slate-400">{label}</div>
      <div className={"break-words text-slate-200 " + (mono ? "font-mono" : "")}>
        {value}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// JSON — export / import / edit
// ---------------------------------------------------------------------------

// Keys the JSON paste is never allowed to change (the file cannot move a
// page to a different image, or change the page's identity).
const PROTECTED_KEYS = new Set([
  "image",
  "width",
  "height",
  "document_id",
  "page_number",
  "page_count",
]);

function JsonPanel({ doc }: { doc: Doc }) {
  const dirty = useStore((s) => s.dirty);
  const edit = useStore((s) => s.edit);
  const setSelection = useStore((s) => s.setSelection);
  const requestFit = useStore((s) => s.requestFit);
  const projectId = useStore((s) => s.currentProjectId) ?? 1;

  const [view, setView] = useState<"export" | "raw">("export");
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [plan, setPlan] = useState<{
    name: string;
    items: JsonlPlanItem[];
  } | null>(null);

  // ---- inline JSON editor state -------------------------------------
  const [editMode, setEditMode] = useState(false);
  const [draft, setDraft] = useState<string>("");
  const [applyErr, setApplyErr] = useState<string | null>(null);

  // ---- derived ------------------------------------------------------
  const preview = useMemo(() => normalizeForExport(doc), [doc]);
  const checks = preview.warnings.filter((w) => w.level === "check");
  const fixed = preview.warnings.filter((w) => w.level === "fixed");

  const generatedText = useMemo(
    () => JSON.stringify(view === "export" ? preview.doc : doc, null, 2),
    [view, preview, doc],
  );

  // Keep draft in sync with the document while NOT editing.
  useEffect(() => {
    if (!editMode) setDraft(generatedText);
  }, [generatedText, editMode]);

  const text = editMode ? draft : generatedText;

  // ---- existing async handlers --------------------------------------

  async function allJsonl() {
    setBusy(true);
    setStatus("Exporting JSONL…");
    try {
      const r = await exportProjectJsonl(projectId, (d, t) =>
        setStatus(`Exporting ${d} / ${t}…`),
      );
      const bad = Object.keys(r.warnings);
      setStatus(
        `Exported ${r.count} line${r.count === 1 ? "" : "s"} to JSONL` +
          (r.failed.length ? ` · failed: ${r.failed.join(", ")}` : "") +
          (bad.length
            ? ` · ${bad.length} page(s) need checking: ${bad.join(", ")}`
            : ""),
      );
    } catch (e) {
      setStatus(`Export failed: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  async function pickJsonl(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setStatus(`Reading ${file.name}…`);
    try {
      const items = await planJsonlImport(file, projectId);
      setPlan({ name: file.name, items });
      setStatus(null);
    } catch (e) {
      setStatus(`Could not read ${file.name}: ${String(e)}`);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function runImport() {
    if (!plan) return;
    const n = plan.items.filter((p) => p.status === "update").length;
    if (
      !window.confirm(
        `Overwrite the saved annotation of ${n} page(s) with the contents of ${plan.name}?\n\n` +
          "This replaces what is on the server for those pages.",
      )
    )
      return;
    setBusy(true);
    try {
      const r = await applyJsonlImport(plan.items, (d, t) =>
        setStatus(`Importing ${d} / ${t}…`),
      );
      // refresh the open page if it was part of the import
      if (r.saved.includes(doc.document_id)) {
        const fresh: any = await getDocument(doc.document_id);
        const st: any = useStore.getState();
        if (st.dirty) {
          setStatus(
            `Imported ${r.saved.length} page(s). This page has unsaved edits, so it was NOT reloaded — reopen it to see the imported version.`,
          );
        } else {
          st.reloadIfClean?.(fresh.doc, fresh.revision, fresh.image_url);
        }
      }
      if (
        !(
          r.saved.includes(doc.document_id) &&
          (useStore.getState() as any).dirty
        )
      ) {
        setStatus(
          `Imported ${r.saved.length} page(s)` +
            (r.failed.length
              ? ` · failed: ${r.failed
                  .map((f) => `${f.id} (${f.error})`)
                  .join(", ")}`
              : ""),
        );
      }
      setPlan(null);
    } catch (e) {
      setStatus(`Import failed: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setStatus("Copied to clipboard");
    } catch {
      setStatus("Clipboard blocked by the browser — use Download instead");
    }
  }

  async function all() {
    setBusy(true);
    setStatus("Exporting…");
    try {
      const r = await exportProject(projectId, (d, t) =>
        setStatus(`Exporting ${d} / ${t}…`),
      );
      const bad = Object.keys(r.warnings);
      setStatus(
        `Exported ${r.count} page${r.count === 1 ? "" : "s"}` +
          (r.failed.length ? ` · failed: ${r.failed.join(", ")}` : "") +
          (bad.length
            ? ` · ${bad.length} page(s) need checking: ${bad.join(", ")}`
            : ""),
      );
    } catch (e) {
      setStatus(`Export failed: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  function applyToPage() {
    if (
      !window.confirm(
        "Rewrite this page in the export format?\n\n" +
          "Nested boxes are folded into their parents and ids are renumbered. " +
          "You can Undo this.",
      )
    )
      return;
    const out = preview.doc;
    edit((d: any) => {
      for (const k of Object.keys(out)) d[k] = (out as any)[k];
    });
    setSelection(null);
    setStatus(
      "Page rewritten in export format — fix the remaining items below, then Save.",
    );
    requestFit();
  }

  // ---- inline JSON editor actions -----------------------------------

  /** Enter edit mode on the RAW editor doc (never the derived export view). */
  function enterEditMode() {
    const current = useStore.getState().doc;
    if (!current) return;
    setView("raw");
    setDraft(JSON.stringify(current, null, 2));
    setApplyErr(null);
    setStatus(
      "Editing the raw editor JSON. Apply changes, then press Save to persist.",
    );
    setEditMode(true);
  }

  function exitEditMode() {
    setEditMode(false);
    setApplyErr(null);
    setStatus(null);
  }

  function applyJson() {
    setApplyErr(null);

    let parsed: any;
    try {
      parsed = JSON.parse(draft);
    } catch (e) {
      setApplyErr(`JSON parse error: ${String(e)}`);
      return;
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      setApplyErr("Top level must be a JSON object.");
      return;
    }

    // Guard: pasting an export-v2 page over a nested doc flattens it.
    const nestedNow = doc.layout.some((el) =>
      doc.layout.some(
        (other) =>
          other.id !== el.id &&
          other.bbox[0] >= el.bbox[0] &&
          other.bbox[1] >= el.bbox[1] &&
          other.bbox[2] <= el.bbox[2] &&
          other.bbox[3] <= el.bbox[3],
      ),
    );
    const looksLikeExport =
      parsed.v === 2 &&
      Array.isArray(parsed.blocks) &&
      Array.isArray(parsed.layout);
    if (nestedNow && looksLikeExport) {
      const ok = window.confirm(
        "This JSON looks like the export (v2) shape, and the current page has nested boxes.\n\n" +
          "Applying it will fold nested boxes into their parents, move layout text into lines, and renumber ids 0…n-1.\n\n" +
          "If you only wanted to fix one field, cancel and edit the raw editor JSON instead.\n\n" +
          "Apply anyway?",
      );
      if (!ok) return;
    }

    const appliedKeys: string[] = [];
    const skippedKeys: string[] = [];

    const before = useStore.getState().doc;
    edit((d: any) => {
      for (const k of Object.keys(parsed)) {
        if (PROTECTED_KEYS.has(k)) {
          skippedKeys.push(k);
          continue;
        }
        d[k] = parsed[k];
        appliedKeys.push(k);
      }
    });
    const after = useStore.getState().doc;

    setSelection(null);
    setEditMode(false);

    if (before === after) {
      setStatus(
        appliedKeys.length === 0
          ? `Nothing to apply — the JSON only had protected keys${
              skippedKeys.length ? ` (${skippedKeys.join(", ")})` : ""
            }.`
          : "JSON matches the current page — no change.",
      );
      return;
    }

    const summary =
      appliedKeys.length <= 6
        ? appliedKeys.join(", ")
        : `${appliedKeys.slice(0, 6).join(", ")}, …`;
    setStatus(
      `Applied ${appliedKeys.length} field${
        appliedKeys.length === 1 ? "" : "s"
      }: ${summary}` +
        (skippedKeys.length
          ? ` (skipped: ${skippedKeys.join(", ")})`
          : "") +
        ". Press Save to persist it.",
    );
    // Force the canvas to re-fit so the new geometry shows up right away.
    requestFit();
  }

  // ---- render -------------------------------------------------------

  return (
    <div className="space-y-2">
      {/* toolbar */}
      <div className="flex flex-wrap gap-1">
        <button
          onClick={() => {
            const w = exportCurrentDoc(doc);
            const n = w.filter((x) => x.level === "check").length;
            setStatus(
              `Downloaded ${doc.document_id}.json` +
                (n ? ` · ${n} item(s) to check below` : ""),
            );
          }}
          className="rounded bg-emerald-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-emerald-500"
          title="Download this page in the export format, including unsaved edits"
        >
          Download this page
        </button>

        <button
          onClick={all}
          disabled={busy}
          className={
            "rounded px-2 py-1 text-[11px] font-medium " +
            (busy
              ? "bg-slate-800 text-slate-500"
              : "bg-sky-700 text-white hover:bg-sky-600")
          }
          title="Download every page in the project as one JSON file (saved versions)"
        >
          Download all pages
        </button>

        <button
          onClick={allJsonl}
          disabled={busy}
          className={
            "rounded px-2 py-1 text-[11px] font-medium " +
            (busy
              ? "bg-slate-800 text-slate-500"
              : "bg-sky-800 text-white hover:bg-sky-700")
          }
          title="Download every page as JSONL — one page per line (saved versions)"
        >
          All pages (JSONL)
        </button>

        <button
          onClick={() => fileRef.current?.click()}
          disabled={busy}
          className="rounded bg-violet-800 px-2 py-1 text-[11px] font-medium text-white hover:bg-violet-700 disabled:bg-slate-800 disabled:text-slate-500"
          title="Load an edited .jsonl and update the matching pages"
        >
          Import JSONL…
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".jsonl,.ndjson,.json,application/json"
          className="hidden"
          onChange={(e) => pickJsonl(e.target.files?.[0])}
        />

        <button
          onClick={() => (editMode ? exitEditMode() : enterEditMode())}
          className={
            "rounded px-2 py-1 text-[11px] font-medium " +
            (editMode
              ? "bg-amber-700 text-white hover:bg-amber-600"
              : "bg-slate-800 text-slate-300 hover:bg-slate-700")
          }
          title={
            editMode
              ? "Stop editing"
              : "Edit the raw editor JSON, then apply it back to the page"
          }
        >
          {editMode ? "Stop editing" : "Edit JSON…"}
        </button>

        {editMode && (
          <button
            onClick={applyJson}
            className="rounded bg-emerald-700 px-2 py-1 text-[11px] font-medium text-white hover:bg-emerald-600"
            title="Apply this JSON to the page (Undo works)"
          >
            Apply changes
          </button>
        )}

        <button
          onClick={copy}
          className="rounded bg-slate-800 px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-700"
        >
          Copy
        </button>
      </div>

      {dirty && (
        <div className="text-[10px] text-amber-400">
          This page has unsaved edits. "Download this page" includes them;
          "Download all pages" uses saved versions — press Save first.
        </div>
      )}
      {status && <div className="text-[10px] text-slate-400">{status}</div>}
      {applyErr && (
        <div className="break-words rounded bg-rose-950/60 px-2 py-1 text-[10px] text-rose-300">
          {applyErr}
        </div>
      )}

      {/* import preview */}
      {plan &&
        (() => {
          const upd = plan.items.filter((p) => p.status === "update");
          const unknown = plan.items.filter(
            (p) => p.status === "unknown_page",
          );
          const bad = plan.items.filter(
            (p) => p.status === "bad_json" || p.status === "no_document_id",
          );
          const withChecks = upd.filter((p) =>
            p.warnings.some((w) => w.level === "check"),
          );
          return (
            <div className="space-y-1 rounded border border-violet-700 bg-violet-950/30 p-2 text-[11px]">
              <div className="font-medium text-violet-200">
                Import preview · {plan.name}
              </div>
              <div className="text-slate-300">
                {upd.length} page(s) will be updated
                {unknown.length > 0 && (
                  <> · {unknown.length} not in this project (skipped)</>
                )}
                {bad.length > 0 && (
                  <> · {bad.length} unreadable line(s) (skipped)</>
                )}
              </div>
              <ul className="max-h-40 space-y-0.5 overflow-auto text-[10px]">
                {unknown.map((p) => (
                  <li key={`u${p.line}`} className="text-slate-400">
                    line {p.line}: {p.document_id} — no such page, skipped
                  </li>
                ))}
                {bad.map((p) => (
                  <li key={`b${p.line}`} className="text-rose-300">
                    line {p.line}:{" "}
                    {p.status === "bad_json"
                      ? `not valid JSON (${p.error})`
                      : "no document_id"}
                  </li>
                ))}
                {withChecks.map((p) => (
                  <li key={`w${p.line}`} className="text-amber-200">
                    line {p.line}: {p.document_id} —{" "}
                    {p.warnings.filter((w) => w.level === "check").length}{" "}
                    item(s) to check (imported anyway; open the page to fix)
                  </li>
                ))}
              </ul>
              <div className="flex gap-1">
                <button
                  onClick={runImport}
                  disabled={busy || upd.length === 0}
                  className="rounded bg-violet-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:bg-slate-800 disabled:text-slate-500"
                >
                  Import {upd.length} page(s)
                </button>
                <button
                  onClick={() => setPlan(null)}
                  className="rounded bg-slate-800 px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-700"
                >
                  Cancel
                </button>
              </div>
            </div>
          );
        })()}

      {/* what still needs a human */}
      <div className="rounded border border-slate-700 p-2">
        <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wider">
          <span
            className={
              checks.length ? "text-amber-300" : "text-emerald-400"
            }
          >
            {checks.length ? `${checks.length} to check` : "ready to export"}
          </span>
          {fixed.length > 0 && (
            <span className="text-slate-500">
              {fixed.length} fixed automatically
            </span>
          )}
          {(checks.length > 0 || fixed.length > 0) && (
            <button
              onClick={applyToPage}
              className="ml-auto rounded bg-slate-800 px-1.5 py-0.5 normal-case tracking-normal text-slate-200 hover:bg-slate-700"
              title="Rewrite this page in the export format so you can fix the rest on the canvas"
            >
              Apply fixes to page
            </button>
          )}
        </div>
        <ul className="max-h-48 space-y-0.5 overflow-auto">
          {checks.map((w, i) => (
            <li key={`c${i}`}>
              <button
                disabled={w.id == null}
                onClick={() =>
                  w.id != null && setSelection({ kind: "layout", id: w.id })
                }
                className="w-full rounded px-1 py-0.5 text-left text-[11px] text-amber-200 hover:bg-slate-800 disabled:hover:bg-transparent"
              >
                ⚠ {w.message}
              </button>
            </li>
          ))}
          {fixed.map((w, i) => (
            <li key={`f${i}`} className="px-1 text-[10px] text-slate-500">
              ✓ {w.message}
            </li>
          ))}
        </ul>
      </div>

      {/* view switcher */}
      <div className="flex gap-1 text-[10px]">
        {(["export", "raw"] as const).map((v) => (
          <button
            key={v}
            onClick={() => {
              setView(v);
              setApplyErr(null);
            }}
            disabled={editMode}
            className={
              "rounded px-2 py-0.5 disabled:opacity-50 " +
              (view === v
                ? "bg-slate-700 text-slate-100"
                : "text-slate-400 hover:bg-slate-800")
            }
          >
            {v === "export" ? "Export (v2)" : "Editor data"}
          </button>
        ))}
      </div>

      {/* editor / preview */}
      {editMode ? (
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          spellCheck={false}
          className="h-[60vh] w-full resize-y whitespace-pre overflow-auto rounded bg-slate-950 p-2 font-mono text-[11px] leading-relaxed text-slate-200 outline-none focus:ring-1 focus:ring-sky-500"
        />
      ) : (
        <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-slate-200">
          {text}
        </pre>
      )}
    </div>
  );
}