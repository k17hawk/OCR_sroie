import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { listDocuments } from "../api";
import { useStore } from "../store";
import type { DocumentSummary } from "../types";

// ---------------------------------------------------------------------------
// Status look
// ---------------------------------------------------------------------------

const STATUS: Record<string, { label: string; dot: string; chip: string; bar: string }> = {
  generated:     { label: "new",         dot: "bg-violet-400",  chip: "bg-violet-900/50 text-violet-200",   bar: "bg-violet-500" },
  pre_annotated: { label: "pre-filled",  dot: "bg-indigo-400",  chip: "bg-indigo-900/50 text-indigo-200",   bar: "bg-indigo-500" },
  in_progress:   { label: "in progress", dot: "bg-sky-400",     chip: "bg-sky-900/50 text-sky-200",         bar: "bg-sky-500" },
  submitted:     { label: "submitted",   dot: "bg-amber-400",   chip: "bg-amber-900/50 text-amber-200",     bar: "bg-amber-500" },
  approved:      { label: "approved",    dot: "bg-emerald-400", chip: "bg-emerald-900/50 text-emerald-200", bar: "bg-emerald-500" },
  rejected:      { label: "rejected",    dot: "bg-rose-400",    chip: "bg-rose-900/50 text-rose-200",       bar: "bg-rose-500" },
};
const ORDER = ["generated", "pre_annotated", "in_progress", "rejected", "submitted", "approved"];
const look = (s: string) => STATUS[s] ?? { label: s, dot: "bg-slate-400", chip: "bg-slate-700 text-slate-200", bar: "bg-slate-500" };
/** pages nobody has to touch any more */
const isDone = (s: string) => s === "submitted" || s === "approved";

function ago(iso?: string | null): string {
  if (!iso) return "";
  const s = (Date.now() - Date.parse(iso)) / 1000;
  if (!Number.isFinite(s)) return "";
  if (s < 60) return "now";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

const fileName = (d: DocumentSummary) => d.source_filename || d.document_id;

// ---------------------------------------------------------------------------
// Remembered view settings
// ---------------------------------------------------------------------------

const PREFS_KEY = "devalabel.queue";
type View = "list" | "grid";
interface Prefs { view: View; collapsed: string[] }
function loadPrefs(): Prefs {
  try {
    const v = JSON.parse(localStorage.getItem(PREFS_KEY) ?? "null");
    if (v && (v.view === "list" || v.view === "grid")) return { view: v.view, collapsed: v.collapsed ?? [] };
  } catch { /* storage blocked */ }
  return { view: "list", collapsed: [] };
}
function savePrefs(p: Prefs) {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* storage blocked */ }
}

/** The page in the address bar (not the loaded doc's own id, which can lag or differ). */
function useHashId(): string | null {
  const read = () => window.location.hash.replace(/^#\/?/, "") || null;
  const [id, setId] = useState(read);
  useEffect(() => {
    const on = () => setId(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return id;
}

const go = (id: string) => { window.location.hash = `#/${id}`; };

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------

type Group = { key: string; name: string; pages: DocumentSummary[] };

export function PageQueue({ projectId }: { projectId: number }) {
  const requestFit = useStore((s) => s.requestFit);
  const activeId = useHashId();

  const { data, isLoading } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => listDocuments(projectId),
    refetchInterval: 5000,
  });

  const [query, setQuery] = useState("");
  const [only, setOnly] = useState<string | null>(null);
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [hover, setHover] = useState<{ doc: DocumentSummary; rect: DOMRect } | null>(null);
  const update = (p: Partial<Prefs>) => setPrefs((old) => { const n = { ...old, ...p }; savePrefs(n); return n; });

  const all = useMemo(() => {
    // one row per page, even if the server ever repeats one
    const seen = new Set<string>();
    return (data ?? []).filter((d) => !seen.has(d.document_id) && seen.add(d.document_id));
  }, [data]);

  // same order as N / P navigation
  const ordered = useMemo(
    () => [...all].sort((a, b) => a.document_id.localeCompare(b.document_id) || a.page_number - b.page_number),
    [all],
  );

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const d of all) c[d.status] = (c[d.status] ?? 0) + 1;
    return c;
  }, [all]);
  const done = all.filter((d) => isDone(d.status)).length;

  const groups = useMemo<Group[]>(() => {
    const q = query.trim().toLowerCase();
    const match = (d: DocumentSummary) =>
      (!only || d.status === only) &&
      (!q || fileName(d).toLowerCase().includes(q) || d.document_id.toLowerCase().includes(q));
    const map = new Map<string, DocumentSummary[]>();
    for (const d of ordered) {
      if (!match(d)) continue;
      const key = d.source_file_id ?? d.document_id;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(d);
    }
    return [...map.entries()].map(([key, pages]) => ({
      key, pages, name: fileName(pages[0]),
    }));
  }, [ordered, query, only]);
  const shown = groups.reduce((n, g) => n + g.pages.length, 0);

  // the next page that still needs work, after the open one
  const nextTodo = useMemo(() => {
    const i = ordered.findIndex((d) => d.document_id === activeId);
    const after = [...ordered.slice(i + 1), ...ordered.slice(0, Math.max(0, i))];
    return after.find((d) => !isDone(d.status)) ?? null;
  }, [ordered, activeId]);

  // keep the open page in view
  const listRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-doc="${CSS.escape(activeId ?? "")}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [activeId, groups.length, prefs.view]);

  if (isLoading) return <div className="text-xs text-slate-500">Loading pages…</div>;
  if (all.length === 0) {
    return <div className="rounded border border-dashed border-slate-700 p-3 text-center text-xs text-slate-500">No pages yet — drop images or PDFs above.</div>;
  }

  const open = (d: DocumentSummary) => {
    if (d.document_id === activeId) requestFit();
    else go(d.document_id);
  };

  const rowProps = (d: DocumentSummary) => ({
    "data-doc": d.document_id,
    // blur: page shortcuts (N / P, Tab, Del…) ignore keys while a button has focus
    onClick: (e: ReactMouseEvent<HTMLElement>) => { e.currentTarget.blur(); open(d); },
    onMouseEnter: (e: ReactMouseEvent<HTMLElement>) => setHover({ doc: d, rect: e.currentTarget.getBoundingClientRect() }),
    onMouseLeave: () => setHover(null),
    title: `${fileName(d)} · ${d.document_id}`,
  });

  return (
    <div className="space-y-2">
      {/* progress */}
      <div>
        <div className="mb-1 flex items-baseline justify-between text-[10px] text-slate-400">
          <span><span className="font-semibold text-slate-200">{done}</span> / {all.length} done</span>
          <span>{Math.round((done / all.length) * 100)}%</span>
        </div>
        <div className="flex h-1.5 overflow-hidden rounded-full bg-slate-800">
          {ORDER.filter((s) => counts[s]).map((s) => (
            <div key={s} className={look(s).bar} style={{ width: `${(counts[s] / all.length) * 100}%` }} title={`${look(s).label}: ${counts[s]}`} />
          ))}
        </div>
      </div>

      {/* search + view + next */}
      <div className="flex gap-1">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Escape") setQuery(""); e.stopPropagation(); }}
          placeholder="Search name or id…"
          className="min-w-0 flex-1 rounded bg-slate-800 px-2 py-1 text-[11px] text-slate-100 placeholder:text-slate-500 outline-none focus:ring-1 focus:ring-sky-500"
        />
        <button
          type="button"
          onClick={() => update({ view: prefs.view === "list" ? "grid" : "list" })}
          className="rounded bg-slate-800 px-2 text-[11px] text-slate-300 hover:bg-slate-700"
          title={prefs.view === "list" ? "Show thumbnails" : "Show list"}
        >
          {prefs.view === "list" ? "▦" : "☰"}
        </button>
      </div>

      {/* status filter */}
      <div className="flex flex-wrap gap-1">
        <Chip on={!only} onClick={() => setOnly(null)} cls="bg-slate-700 text-slate-200">all {all.length}</Chip>
        {ORDER.filter((s) => counts[s]).map((s) => (
          <Chip key={s} on={only === s} onClick={() => setOnly(only === s ? null : s)} cls={look(s).chip}>
            <span className={"mr-1 inline-block h-1.5 w-1.5 rounded-full " + look(s).dot} />
            {look(s).label} {counts[s]}
          </Chip>
        ))}
      </div>

      {nextTodo && (
        <button
          type="button"
          onClick={() => go(nextTodo.document_id)}
          className="w-full rounded bg-sky-700/70 py-1 text-[11px] font-medium text-sky-50 hover:bg-sky-600"
          title={`${fileName(nextTodo)} · ${nextTodo.document_id}`}
        >
          Next to do → <span className="font-normal opacity-80">{fileName(nextTodo)}{nextTodo.page_count > 1 ? ` p.${nextTodo.page_number}` : ""}</span>
        </button>
      )}

      {/* pages */}
      <div ref={listRef} className="max-h-[42vh] space-y-2 overflow-y-auto pr-1">
        {shown === 0 && <div className="py-3 text-center text-[11px] text-slate-500">No pages match.</div>}
        {groups.map((g) => {
          const multi = g.pages.length > 1 || (g.pages[0]?.page_count ?? 1) > 1;
          const collapsed = multi && prefs.collapsed.includes(g.key) && !g.pages.some((p) => p.document_id === activeId);
          const gDone = g.pages.filter((p) => isDone(p.status)).length;
          return (
            <div key={g.key}>
              {multi && (
                <button
                  type="button"
                  onClick={() => update({
                    collapsed: prefs.collapsed.includes(g.key)
                      ? prefs.collapsed.filter((k) => k !== g.key)
                      : [...prefs.collapsed, g.key],
                  })}
                  className="mb-1 flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left text-[10px] text-slate-400 hover:bg-slate-800"
                >
                  <span className={"transition-transform " + (collapsed ? "" : "rotate-90")}>▸</span>
                  <span className="truncate font-medium text-slate-300">{g.name}</span>
                  <span className="ml-auto shrink-0 font-mono">{gDone}/{g.pages.length}</span>
                </button>
              )}
              {!collapsed && (prefs.view === "grid" ? (
                <div className={"grid grid-cols-3 gap-1.5 " + (multi ? "pl-2" : "")}>
                  {g.pages.map((d) => {
                    const active = d.document_id === activeId;
                    return (
                      <button key={d.document_id} type="button" {...rowProps(d)}
                        className={"group relative overflow-hidden rounded border text-left transition " +
                          (active ? "border-sky-400 ring-1 ring-sky-400" : "border-slate-700 hover:border-slate-400")}>
                        <img src={d.image_url} alt="" loading="lazy" className="aspect-[3/4] w-full bg-white object-cover object-top" />
                        <span className={"absolute right-1 top-1 h-2 w-2 rounded-full ring-2 ring-slate-900 " + look(d.status).dot} />
                        <span className="absolute inset-x-0 bottom-0 truncate bg-slate-950/80 px-1 text-[9px] text-slate-200">
                          {multi ? `p.${d.page_number}` : fileName(d)}
                        </span>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <ul className={"space-y-0.5 " + (multi ? "border-l border-slate-700 pl-2" : "")}>
                  {g.pages.map((d) => {
                    const active = d.document_id === activeId;
                    return (
                      <li key={d.document_id}>
                        <button type="button" {...rowProps(d)}
                          className={"flex w-full items-center gap-2 rounded px-1.5 py-1 text-left transition " +
                            (active ? "bg-slate-700 ring-1 ring-sky-500/60" : "hover:bg-slate-800")}>
                          <span className={"h-8 w-1 shrink-0 rounded-full " + look(d.status).dot} />
                          <img src={d.image_url} alt="" loading="lazy" className="h-8 w-6 shrink-0 rounded-sm bg-white object-cover object-top" />
                          <span className="min-w-0 flex-1">
                            <span className={"block truncate text-[11px] " + (active ? "text-white" : "text-slate-200")}>
                              {multi ? `Page ${d.page_number}` : fileName(d)}
                            </span>
                            <span className="flex gap-1.5 font-mono text-[9px] text-slate-500">
                              <span>{d.document_id}</span>
                              {d.updated_at && <span title={new Date(d.updated_at).toLocaleString()}>· {ago(d.updated_at)}</span>}
                            </span>
                          </span>
                          <span className={"shrink-0 rounded px-1 text-[9px] " + look(d.status).chip}>{look(d.status).label}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              ))}
            </div>
          );
        })}
      </div>

      {hover && <Preview doc={hover.doc} rect={hover.rect} />}
    </div>
  );
}

function Chip({ on, onClick, cls, children }: { on: boolean; onClick: () => void; cls: string; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={"flex items-center rounded-full px-2 py-0.5 text-[10px] transition " + cls +
        (on ? " ring-1 ring-white/60" : " opacity-60 hover:opacity-100")}
    >
      {children}
    </button>
  );
}

/** Bigger look at a page, beside the panel, while the pointer is on its row. */
function Preview({ doc, rect }: { doc: DocumentSummary; rect: DOMRect }) {
  const W = 220;
  const top = Math.min(Math.max(8, rect.top - 40), window.innerHeight - 330);
  return createPortal(
    <div className="pointer-events-none fixed z-50 rounded-md border border-slate-600 bg-slate-900 p-1.5 shadow-2xl"
      style={{ left: rect.right + 10, top, width: W }}>
      <img src={doc.image_url} alt="" className="max-h-[280px] w-full rounded bg-white object-contain" />
      <div className="mt-1 flex items-center gap-1.5 text-[10px]">
        <span className={"h-1.5 w-1.5 rounded-full " + look(doc.status).dot} />
        <span className="truncate text-slate-200">{fileName(doc)}</span>
        {doc.page_count > 1 && <span className="shrink-0 text-slate-500">p.{doc.page_number}/{doc.page_count}</span>}
      </div>
    </div>,
    document.body,
  );
}
