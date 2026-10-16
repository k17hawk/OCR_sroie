import { useQuery } from "@tanstack/react-query";
import { useStore } from "../store";
import { TYPE_COLORS } from "../lib";

/**
 * Label vocabularies served by the backend at GET /projects/:id/types.
 * Every dropdown in the UI reads from here, so adding a type on the server
 * makes it appear in the annotator without a frontend change.
 */
export interface ProjectTypes {
  doc_types: string[];
  field_types: string[];
  layout_types: string[];
}

// Used until the request resolves, or if the server is unreachable.
export const FALLBACK_TYPES: ProjectTypes = {
  doc_types: [
    "bank_account_form", "citizenship", "national_id", "passport",
    "driving_license", "pan_card", "voter_id", "invoice", "receipt",
    "bank_cheque", "bank_statement", "marksheet", "certificate",
    "land_ownership", "application_letter", "government_notice", "other",
  ],
  field_types: [
    "text", "name", "name_m", "name_f", "date", "date_bs", "date_ad",
    "checkbox", "citizenship", "national_id", "pan", "passport_no",
    "district", "municipality", "ward", "address", "phone", "email",
    "money", "account_number", "number", "signature_ref", "other",
  ],
  // the region types of the v2 schema (older names are mapped on load)
  layout_types: [
    "title", "heading", "subheading", "paragraph", "list", "form_field",
    "table", "table_heading", "header", "footer", "figure", "logo", "stamp",
    "signature", "handwriting", "caption",
  ],
};

async function fetchProjectTypes(projectId: number | string): Promise<ProjectTypes> {
  const res = await fetch(`/projects/${projectId}/types`);
  if (!res.ok) throw new Error(`GET /projects/${projectId}/types → ${res.status}`);
  const j = await res.json();
  return {
    doc_types: Array.isArray(j.doc_types) ? j.doc_types : FALLBACK_TYPES.doc_types,
    field_types: Array.isArray(j.field_types) ? j.field_types : FALLBACK_TYPES.field_types,
    layout_types: Array.isArray(j.layout_types) ? j.layout_types : FALLBACK_TYPES.layout_types,
  };
}

export function useProjectTypes(): ProjectTypes & { loading: boolean; error: unknown } {
  const projectId = useStore((s) => (s as any).currentProjectId) ?? 1;
  const q = useQuery({
    queryKey: ["project-types", projectId],
    queryFn: () => fetchProjectTypes(projectId),
    staleTime: 5 * 60_000,
    retry: 1,
  });
  return { ...(q.data ?? FALLBACK_TYPES), loading: q.isLoading, error: q.error };
}

/**
 * Make sure the element's current value is always in the list, so a type
 * saved by an older schema (or a typo from the model) still shows up
 * instead of the <select> silently snapping to the first option.
 */
export function withCurrent(list: readonly string[], current: string | null | undefined): string[] {
  return current && !list.includes(current) ? [...list, current] : [...list];
}

/** Human-friendly label: "handwritten_number" → "handwritten number" */
export function prettyType(t: string): string {
  return t.replace(/_/g, " ");
}

// Types that carry plain text and are drawn/edited like a paragraph.
export const TEXT_LIKE_TYPES = new Set([
  "header", "title", "heading", "subheading", "paragraph",
  "footer", "caption", "table_heading", "handwriting",
]);

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

/**
 * Bold, saturated colors (Tailwind 500–700) that stay visible on a white
 * scan. Families make types easy to tell apart at a glance:
 *   blues/cyans = running text   purples = headings/titles
 *   oranges/ambers = forms        reds = tables
 *   greens = figures/handwriting  pinks = signatures
 */
export const BOLD_COLORS: Record<string, string> = {
  // running text
  header: "#2563eb",
  paragraph: "#0284c7",
  text_line: "#0891b2",
  footer: "#4f46e5",
  mixed_text: "#6366f1",
  page_number: "#475569",
  // headings
  title: "#c026d3",
  heading: "#9333ea",
  subheading: "#7c3aed",
  // forms
  form_field: "#ea580c",
  form_label: "#0d9488",
  form_value: "#d97706",
  label: "#0f766e",
  options: "#e11d48",
  // tables
  table: "#dc2626",
  table_heading: "#f97316",
  table_footer: "#b45309",
  // lists
  list: "#ca8a04",
  list_item: "#a16207",
  // visual
  figure: "#16a34a",
  logo: "#65a30d",
  signature: "#be185d",
  caption: "#db2777",
  stamp: "#9f1239",
  // handwriting
  handwriting: "#059669",
  handwritten_annotation: "#10b981",
  handwritten_correction: "#f43f5e",
  handwritten_number: "#14b8a6",
};

/** Colors for things that are not layout regions. */
export const KIND_COLORS = {
  line: "#0891b2",
  lineHeading: "#9333ea",
  cell: "#dc2626",
  cellHeader: "#d97706",
  fieldLabel: "#0284c7",
  fieldValue: "#ca8a04",
  checked: "#16a34a",
  unchecked: "#db2777",
  fingerprint: "#7c3aed",
} as const;

function hashColor(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  const hue = h % 360;
  // HSL(hue, 80%, 45%) → hex: saturated and dark enough to read on white
  const l = 0.45, sat = 0.8;
  const a = sat * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + hue / 30) % 12;
    const c = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(c * 255).toString(16).padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

/** Always returns a 6-digit hex color for any layout type (callers append alpha). */
export function typeColor(type: string): string {
  const known =
    BOLD_COLORS[type] ??
    (TYPE_COLORS as Record<string, string | undefined>)[type];
  if (known && /^#[0-9a-fA-F]{6}$/.test(known)) return known;
  return hashColor(type);
}
