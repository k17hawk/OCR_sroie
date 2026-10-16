import type { LayoutType } from "./types";

export const TYPE_COLORS: Record<LayoutType, string> = {
  header: "#3b82f6",
  title: "#c026d3",
  heading: "#8b5cf6",
  subheading: "#7c3aed",
  paragraph: "#10b981",
  table_heading: "#f97316",
  figure: "#f97316",
  form_field: "#06b6d4",
  table: "#ef4444",
  list: "#eab308",
  logo: "#94a3b8",
  signature: "#ec4899",
  stamp: "#a16207",
  footer: "#14b8a6",
  caption: "#f472b6",
  handwriting: "#059669",
};

export const LAYER_KEYS = [
  "image",
  "layout",
  "blocks",
  "lines",
  "tables",
  "formFields",
  "checkboxes",
  "extraRegions",
  "readingOrder",
] as const;

export type LayerKey = (typeof LAYER_KEYS)[number];

export const LAYER_LABELS: Record<LayerKey, string> = {
  image: "Image",
  layout: "Layout regions",
  blocks: "Blocks",
  lines: "Lines",
  tables: "Table grid",
  formFields: "Form fields",
  checkboxes: "Checkboxes",
  extraRegions: "Extra regions",
  readingOrder: "Reading order",
};

export const LAYER_DEFAULTS: Record<LayerKey, boolean> = {
  image: true,
  layout: true,
  blocks: false, // derived; hidden by default per §2.4
  lines: true,
  tables: true,
  formFields: true,
  checkboxes: true,
  extraRegions: true,
  readingOrder: false,
};

export function bboxToRect(bbox: [number, number, number, number]) {
  const [x0, y0, x1, y1] = bbox;
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

export function quadPoints(quad: [number, number][]): number[] {
  return quad.flatMap((p) => [p[0], p[1]]);
}

export function formatNumber(n: number, digits = 1): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

export function bboxFromQuad(
  quad: [number, number][],
): [number, number, number, number] {
  const xs = quad.map((p) => p[0]);
  const ys = quad.map((p) => p[1]);
  return [
    Math.floor(Math.min(...xs)),
    Math.floor(Math.min(...ys)),
    Math.ceil(Math.max(...xs)),
    Math.ceil(Math.max(...ys)),
  ];
}

export function roundQuad(
  quad: [number, number][],
): [number, number][] {
  return quad.map((p) => [
    Math.round(p[0] * 10) / 10,
    Math.round(p[1] * 10) / 10,
  ]);
}

/** Describe a selection for the Properties panel header. */
export function selectionLabel(
  kind: string,
  id: string | number,
  doc?: { layout: { id: number; type: string }[] },
): string {
  if (kind === "layout" && doc) {
    const el = doc.layout.find((e) => e.id === id);
    return `layout[${id}] — ${el?.type ?? "?"}`;
  }
  return `${kind}[${id}]`;
}

export const FALLBACK_DOC_TYPES = [
  "bank_account_form", "citizenship", "national_id", "passport",
  "driving_license", "pan_card", "voter_id", "invoice", "receipt",
  "bank_cheque", "bank_statement", "marksheet", "certificate",
  "land_ownership", "application_letter", "government_notice", "other",
] as const;

export const FALLBACK_FIELD_TYPES = [
  "text", "name", "name_m", "name_f", "date", "date_bs", "date_ad",
  "checkbox", "citizenship", "national_id", "pan", "passport_no",
  "district", "municipality", "ward", "address", "phone", "email",
  "money", "account_number", "number", "signature_ref", "other",
] as const;