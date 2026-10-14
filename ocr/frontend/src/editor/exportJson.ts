import { getDocument, listDocuments } from "../api";
import type { Doc } from "../types";
import { normalizeForExport, type ExportWarning } from "./normalizeDoc";

export type { ExportWarning };

/** Trigger a browser download of `data` as a pretty-printed .json file. */
export function downloadJson(filename: string, data: unknown): void {
  const text = JSON.stringify(data, null, 2);
  const blob = new Blob([text], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".json") ? filename : `${filename}.json`;
  // must be in the DOM for Firefox
  document.body.appendChild(a);
  a.click();
  a.remove();
  // give the browser a moment to start the download before revoking
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function safeName(s: string): string {
  return s.replace(/[^\w.-]+/g, "_");
}

/**
 * Export the page currently open in the editor — including edits you have
 * not saved yet — converted to the flat v2 schema (see normalizeDoc.ts).
 * Returns the warnings so the UI can show what still needs fixing.
 */
export function exportCurrentDoc(doc: Doc): ExportWarning[] {
  const { doc: out, warnings } = normalizeForExport(doc);
  downloadJson(`${safeName(doc.document_id)}.json`, out);
  return warnings;
}

/**
 * Export every page in the project as one JSON file:
 *   { project_id, exported_at, count, documents: [Doc, ...] }
 * Pages come from the server, so SAVE first if the open page has edits.
 * `onProgress` is called as pages download.
 */
export async function exportProject(
  projectId: number | string,
  onProgress?: (done: number, total: number) => void,
): Promise<{ count: number; failed: string[]; warnings: Record<string, ExportWarning[]> }> {
  const list = await listDocuments(projectId as any);
  const ids: string[] = (list ?? []).map((d: any) => d.document_id);
  const documents: Doc[] = [];
  const failed: string[] = [];
  const warnings: Record<string, ExportWarning[]> = {};

  for (let i = 0; i < ids.length; i++) {
    try {
      const r: any = await getDocument(ids[i]);
      const n = normalizeForExport(r.doc ?? r);
      documents.push(n.doc);
      const checks = n.warnings.filter((w) => w.level === "check");
      if (checks.length) warnings[ids[i]] = checks;
    } catch {
      failed.push(ids[i]);
    }
    onProgress?.(i + 1, ids.length);
  }

  const payload = {
    project_id: projectId,
    exported_at: new Date().toISOString(),
    count: documents.length,
    documents,
  };
  const stamp = new Date().toISOString().slice(0, 10);
  downloadJson(`project-${projectId}-${stamp}.json`, payload);
  return { count: documents.length, failed, warnings };
}

// ---------------------------------------------------------------------------
// JSONL — one v2 page per line
// ---------------------------------------------------------------------------

function downloadText(filename: string, text: string, type: string): void {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Export every page in the project as JSONL: one exported (v2) page per
 * line, in project order. Pages come from the server — Save first.
 */
export async function exportProjectJsonl(
  projectId: number | string,
  onProgress?: (done: number, total: number) => void,
): Promise<{ count: number; failed: string[]; warnings: Record<string, ExportWarning[]> }> {
  const list = await listDocuments(projectId as any);
  const ids: string[] = (list ?? [])
    .slice()
    .sort((a: any, b: any) =>
      String(a.document_id).localeCompare(String(b.document_id)) ||
      (a.page_number ?? 0) - (b.page_number ?? 0))
    .map((d: any) => d.document_id);
  const out: string[] = [];
  const failed: string[] = [];
  const warnings: Record<string, ExportWarning[]> = {};
  for (let i = 0; i < ids.length; i++) {
    try {
      const r: any = await getDocument(ids[i]);
      const n = normalizeForExport(r.doc ?? r);
      out.push(JSON.stringify(n.doc)); // one line, no pretty-printing
      const checks = n.warnings.filter((w) => w.level === "check");
      if (checks.length) warnings[ids[i]] = checks;
    } catch {
      failed.push(ids[i]);
    }
    onProgress?.(i + 1, ids.length);
  }
  const stamp = new Date().toISOString().slice(0, 10);
  downloadText(`project-${projectId}-${stamp}.jsonl`, out.join("\n") + "\n", "application/x-ndjson;charset=utf-8");
  return { count: out.length, failed, warnings };
}

export interface JsonlPlanItem {
  line: number;                 // 1-based line number in the file
  document_id: string | null;
  status: "update" | "unknown_page" | "bad_json" | "no_document_id";
  doc?: any;                    // normalized page, ready to save
  warnings: ExportWarning[];
  error?: string;
}

/**
 * Read a .jsonl file and work out what an import would do — without
 * changing anything. Each line is normalized (so hand edits that left
 * text / lines / counts out of sync are rebuilt) and matched to a page
 * in the project by document_id.
 */
export async function planJsonlImport(file: File, projectId: number | string): Promise<JsonlPlanItem[]> {
  const text = (await file.text()).replace(/^﻿/, "");
  const list = await listDocuments(projectId as any);
  const known = new Set((list ?? []).map((d: any) => String(d.document_id)));
  const plan: JsonlPlanItem[] = [];
  text.split(/\r?\n/).forEach((raw, i) => {
    if (!raw.trim()) return;
    let obj: any;
    try {
      obj = JSON.parse(raw);
    } catch (e) {
      plan.push({ line: i + 1, document_id: null, status: "bad_json", warnings: [], error: String(e) });
      return;
    }
    // accept both a bare page and an export wrapper { documents: [...] }
    const pages: any[] = Array.isArray(obj?.documents) ? obj.documents : [obj];
    for (const page of pages) {
      const id = page?.document_id != null ? String(page.document_id) : null;
      if (!id) {
        plan.push({ line: i + 1, document_id: null, status: "no_document_id", warnings: [] });
        continue;
      }
      if (!known.has(id)) {
        plan.push({ line: i + 1, document_id: id, status: "unknown_page", warnings: [] });
        continue;
      }
      const n = normalizeForExport(page);
      plan.push({ line: i + 1, document_id: id, status: "update", doc: n.doc, warnings: n.warnings });
    }
  });
  return plan;
}

/**
 * Save every "update" item of a plan to the server. Uses each page's
 * current revision, keeps the server's image path and size (the file
 * cannot move a page to a different image), and returns what happened.
 */
export async function applyJsonlImport(
  plan: JsonlPlanItem[],
  onProgress?: (done: number, total: number) => void,
): Promise<{ saved: string[]; failed: { id: string; error: string }[] }> {
  const { putAnnotation } = await import("../api");
  const todo = plan.filter((p) => p.status === "update" && p.doc);
  const saved: string[] = [];
  const failed: { id: string; error: string }[] = [];
  for (let i = 0; i < todo.length; i++) {
    const item = todo[i];
    const id = item.document_id!;
    try {
      const cur: any = await getDocument(id);
      const server = cur.doc ?? cur;
      const doc = {
        ...item.doc,
        image: server.image ?? item.doc.image,
        width: server.width ?? item.doc.width,
        height: server.height ?? item.doc.height,
      };
      await putAnnotation(id, cur.revision, doc);
      saved.push(id);
    } catch (e) {
      failed.push({ id, error: String(e) });
    }
    onProgress?.(i + 1, todo.length);
  }
  return { saved, failed };
}
