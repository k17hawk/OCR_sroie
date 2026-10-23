import type {
  Doc, Job, UploadResponse, OcrRegionResponse,
  ModelFamily, SystemDevices, DocumentSummary, LoadWarning,
} from "./types";

import type { Project, TableOcrResponse, Table } from "./types";

const API = "";

export async function fetchSample(): Promise<Doc> {
  const r = await fetch("/sample.json");
  if (!r.ok) throw new Error(`sample: ${r.status}`);
  return await r.json();
}

export async function getDocument(id: string): Promise<{
  doc: Doc;
  revision: number;
  image_url: string;
  source_file_id: string | null;
  warnings: LoadWarning[];
}> {
  const r = await fetch(`/documents/${id}`);
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return {
    doc: j.doc as Doc,
    revision: j.revision as number,
    image_url: j.image_url as string,
    source_file_id: (j.source_file_id ?? null) as string | null,
    warnings: (Array.isArray(j.warnings) ? j.warnings : []) as LoadWarning[],
  };
}

export async function putAnnotation(id: string, revision: number, doc: Doc) {
  const r = await fetch(`${API}/documents/${id}/annotation`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision, doc }),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function uploadImages(projectId: number, files: FileList | File[]): Promise<UploadResponse> {
  const fd = new FormData();
  for (const f of Array.from(files)) fd.append("files", f);
  const r = await fetch(`${API}/projects/${projectId}/images`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function preannotate(
  documentId: string,
  opts: { family?: string; device?: string },
): Promise<Job> {
  const r = await fetch(`${API}/documents/${documentId}/preannotate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_config: { family: opts.family ?? "mock" },
      device: opts.device ?? "cpu",
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function getJob(jobId: number): Promise<Job> {
  const r = await fetch(`${API}/jobs/${jobId}`);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function listModels(): Promise<{ families: ModelFamily[] }> {
  const r = await fetch("/system/models");
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function listSystemDevices(): Promise<SystemDevices> {
  const r = await fetch("/system/devices");
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function listDocuments(projectId: number): Promise<DocumentSummary[]> {
  const r = await fetch(`/projects/${projectId}/documents`);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function ocrRegion(
  documentId: string,
  args: {
    quad: [number, number][];
    family: string;
    device: string;
    lang_hint: "ne" | "en" | "mixed" | "hi";},
  ): Promise<OcrRegionResponse> {
  const r = await fetch(`/documents/${documentId}/ocr-region`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}


export async function getProject(projectId: number): Promise<Project> {
  const r = await fetch(`/projects/${projectId}`);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function ocrTable(
  documentId: string,
  args: {
    quad: [number, number][];
    family: string;
    device: string;
    lang_hint: string;
    /** typed table HTML: keep its structure + text, only fit it to the image */
    html?: string;
  },
): Promise<TableOcrResponse> {
  const r = await fetch(`/documents/${documentId}/ocr-table`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}


export interface OcrPageResponse {
  model: string;
  device: string;
  lines: { index: number; text: string; confidence: number }[];
  layout: { id: number; text: string }[];
}

export async function ocrPage(
  documentId: string,
  args: { family: string; device: string; lang_hint: string },
): Promise<OcrPageResponse> {
  const r = await fetch(`/documents/${documentId}/ocr-page`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export type { Table };

/** Result of a geometry fix that reads the scan (detect lines, snap table). */
export interface PageFixResponse {
  record: Doc;
  warnings: string[];
  /** snap-table: one summary per table that was snapped */
  info?: string[];
}

async function pageFix(
  documentId: string,
  action: "detect-lines" | "snap-table",
  args: { doc: Doc; element_id?: number | null; html?: string },
): Promise<PageFixResponse> {
  const r = await fetch(`/documents/${documentId}/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

/** One line record per printed line inside each text region (or just `element_id`). */
export const detectLines = (documentId: string, args: { doc: Doc; element_id?: number | null }) =>
  pageFix(documentId, "detect-lines", args);

/** Put a table's cells on the ruling lines printed on the scan (unchanged + warning on a mismatch). */
export const snapTable = (documentId: string, args: { doc: Doc; element_id: number; html?: string }) =>
  pageFix(documentId, "snap-table", args);

/**
 * Tighten a drawn / resized child box to the text ink inside it (kept inside
 * `clip_to`). Tilted text comes back as a tilted `quad` with its outer `bbox`.
 */
export async function snapBox(
  documentId: string,
  bbox: [number, number, number, number],
  clip_to?: [number, number, number, number] | null,
): Promise<{
  bbox: [number, number, number, number];
  changed: boolean;
  /** set when the text inside is tilted: a tight rotated box (TL, TR, BR, BL) */
  quad?: [number, number][] | null;
}> {
  const r = await fetch(`/documents/${documentId}/snap-box`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bbox, clip_to: clip_to ?? null }),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

/** A table grid suggested from the ruling lines printed inside a drawn box. */
export interface TableSuggestion {
  ok: boolean;
  /** page tilt in degrees (+ = counter-clockwise) */
  skew_deg: number;
  num_rows: number;
  num_cols: number;
  region_bbox: [number, number, number, number] | null;
  region_quad: [number, number][] | null;
  cells: {
    id: number; row: number; col: number; rowspan: number; colspan: number;
    bbox: [number, number, number, number]; quad: [number, number][];
  }[];
  html: string;
  warnings: string[];
}

/** Rows, columns and merged cells read from the scan inside `bbox`; changes nothing. */
export async function suggestTable(
  documentId: string,
  bbox: [number, number, number, number],
): Promise<TableSuggestion> {
  const r = await fetch(`/documents/${documentId}/suggest-table`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bbox }),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
