export type Point = [number, number];
export type BBox = [number, number, number, number];
export type Quad = [Point, Point, Point, Point];

export type Language = "ne" | "hi" | "en" | "mixed";
export type DigitStyle = "devanagari" | "dev" | "latin" | "mixed";

export type AnnotationSource = "synthetic" | "real";
export type AnnotationStatus =
  | "generated"
  | "pre_annotated"
  | "in_progress"
  | "submitted"
  | "approved"
  | "rejected";

export interface FormFieldLabel {
  text: string;
  bbox: BBox;
  quad: Quad;
}

export interface CheckboxOption {
  text: string;
  checked: boolean;
  box_quad: Quad;
  text_quad: Quad;
}

export interface NormalFieldValue {
  text: string;
  bbox: BBox;
  quad: Quad;
  handwritten: boolean;
}

export interface CheckboxFieldValue {
  text: string;
  options: CheckboxOption[];
}

export type FieldValue = NormalFieldValue | CheckboxFieldValue;

export interface FormFieldEntry {
  label: FormFieldLabel;
  field_type: string;
  value: FieldValue | null;
}

export interface ListItem {
  marker: string;
  level: number;
  text: string;
  /** ids of layout elements that belong to this item (a note, a figure, …) */
  children?: number[];
}

export interface HeaderLayout {
  id: number;
  type: "header";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  quad: Quad;
}
export interface HeadingLayout {
  id: number;
  type: "heading";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  level: number;
  quad: Quad;
}
export interface FigureLayout {
  id: number;
  type: "figure";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  subtype: string;
  quad: Quad;
}
export interface FormFieldLayout {
  id: number;
  type: "form_field";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  fields: FormFieldEntry[];
  quad: Quad;
}
export interface TableLayout {
  id: number;
  type: "table";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  table_index: number;
  quad: Quad;
}
export interface ListLayout {
  id: number;
  type: "list";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  list_style: string;
  items: ListItem[];
  quad: Quad;
}
export interface ParagraphLayout {
  id: number;
  type: "paragraph";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  quad: Quad;
}
export interface SignatureCaptionLayout {
  id: number;
  type: "signature_caption";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  quad: Quad;
}
export interface FooterLayout {
  id: number;
  type: "footer";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  quad: Quad;
}
export interface LogoLayout {
  id: number;
  type: "logo";
  bbox: BBox;
  reading_order: number | null;
  quad: Quad;
}
export interface SignatureLayout {
  id: number;
  type: "signature";
  bbox: BBox;
  reading_order: number | null;
  signer: string;
  quad: Quad;
}
export interface StampLayout {
  id: number;
  type: "stamp";
  bbox: BBox;
  reading_order: number | null;
  stamp_text: string;
  quad: Quad;
}

/** Handwritten region (a note written next to a list item, …). */
export interface HandwritingLayout {
  id: number;
  type: "handwriting";
  bbox: BBox;
  reading_order: number | null;
  text: string;
  subtype?: string;
  handwritten?: boolean;
  quad: Quad;
}

export type LayoutElement = NestedIn & (
  | HeaderLayout
  | HeadingLayout
  | FigureLayout
  | FormFieldLayout
  | TableLayout
  | ListLayout
  | ParagraphLayout
  | SignatureCaptionLayout
  | FooterLayout
  | LogoLayout
  | SignatureLayout
  | StampLayout
  | HandwritingLayout);

/** Set on a box that belongs to a list item (see ListItem.children). */
export interface NestedIn {
  parent_id?: number | null;
  parent_item?: number | null;
}

export type LayoutType = LayoutElement["type"];

export interface Block {
  type: string;
  bbox: BBox;
  block_id: number;
  font_id: number | null;
  quad: Quad;
}

export type LineRole = "label" | "value" | "option";

export interface Line {
  text: string;
  bbox: BBox;
  block_id: number;
  line_in_block: number;
  is_heading: boolean;
  font_size: number;
  font_id: number | null;
  role?: LineRole;
  handwritten?: true;
  table_index?: number;
  cell_id?: number;
  list_level?: number;
  item_index?: number;
  quad: Quad;
}

export interface TableCell {
  id: number;
  row: number;
  col: number;
  rowspan: number;
  colspan: number;
  text: string;
  is_header: boolean;
  handwritten: boolean;
  empty: boolean;
  bbox: BBox;
  quad: Quad;
}

export type RulingStyle = "full" | "horizontal" | "none" | "partial";

export interface Table {
  bbox: BBox;
  quad: Quad;
  num_rows: number;
  num_cols: number;
  header_rows: number;
  ruling_style: RulingStyle;
  theme: string | null;
  col_types: string | null;
  html: string;
  reading_order: number[];
  cells: TableCell[];
}

export interface RulingLine {
  table_index: number;
  orientation: "h" | "v";
  width: number;
  p0: Point;
  p1: Point;
  bbox: BBox;
}

export interface CheckboxExtraRegion {
  type: "checkbox";
  checked: boolean;
  option: string;
  bbox: BBox;
  quad: Quad;
}
export interface FingerprintExtraRegion {
  type: "fingerprint";
  hand: "left" | "right";
  bbox: BBox;
  quad: Quad;
}
export type ExtraRegion = CheckboxExtraRegion | FingerprintExtraRegion;

export interface RawLengths {
  paragraph_chars: number;
  list_items: number;
  form_fields: number;
  table_cells: number;
}

export interface Quality {
  blur_sigma: number | null;
  downscale: number | null;
  noise_sigma: number | null;
  is_grayscale: boolean;
  is_binarized: boolean;
  skew_deg: number | null;
  perspective: boolean;
  ops: string[];
  jpeg_quality: number | null;
  profile: string;
  is_clean: boolean;
  has_stamp: boolean;
  has_signature: boolean;
  has_handwriting: boolean;
  has_fold: boolean;
  has_bleedthrough: boolean;
  line_count: number;
  table_count: number;
  figure_count: number;
  stamp_bboxes: BBox[];
  signature_bboxes: BBox[];
  raw_lengths: RawLengths;
}

export interface Annotation {
  source: AnnotationSource;
  tool: string;
  status: AnnotationStatus;
}

export interface Doc {
  v: number;
  image: string;
  width: number;
  height: number;
  document_id: string;
  page_number: number;
  page_count: number;
  language: Language;
  digit_style: DigitStyle;
  doc_type: string;
  text: string;
  fonts: unknown[];
  font_ids: unknown[];
  layout: LayoutElement[];
  blocks: Block[];
  lines: Line[];
  tables: Table[];
  ruling_lines: RulingLine[];
  extra_regions: ExtraRegion[];
  quality: Quality;
  annotation: Annotation;
}

export interface Job {
  id: number;
  document_id: string;
  kind: "preannotate";
  status: "queued" | "running" | "succeeded" | "failed";
  progress: number;
  device: string;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface FileResult {
  filename: string;
  ok: boolean;
  error: string | null;
  skipped_duplicate: boolean;
  warning?: string | null;
  document_ids: string[];
}

export interface UploadResponse {
  results: FileResult[];
  documents: {
    document_id: string;
    page_number: number;
    page_count: number;
    revision: number;
    status: string;
    image_url: string;
    source_file_id: string | null; 
  }[];
}




export interface OcrRegionResponse {
  text: string;
  confidence: number;
  family: string;
  device: string;
  model: string;
}

export interface ModelFamily {
  family: string;
  requires: string[];
  available: boolean;
  supports_cuda: boolean;
  description: string;
}

export interface SystemDevices {
  devices: {
    id: string;
    kind: "cpu" | "cuda";
    name?: string;
    free_mb: number | null;
    total_mb?: number;
  }[];
  tesseract: boolean;
  version: string;
}

export interface DocumentSummary {
  document_id: string;
  page_number: number;
  page_count: number;
  revision: number;
  status: string;
  image_url: string;
}

export type ConflictState =
  | { kind: "none" }
  | { kind: "remote-newer" }   // server has a higher revision, we have unsaved edits
  | { kind: "remote-changed" }; // server has a higher revision, we're clean

export interface TableOcrResponse {
  table: Table;
  family: string;
  device: string;
  model: string;
}

export interface Project {
  id: number;
  name: string;
  doc_types: string[];
  field_types: string[];
  label_map: Record<string, unknown>;
  model_settings: Record<string, unknown>;   // ← renamed from model_config
  created_at: string;
  updated_at: string;
}