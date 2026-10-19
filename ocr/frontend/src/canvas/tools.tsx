import { useRef, useState } from "react";
import { Rect, Line } from "react-konva";

import { useStore } from "../store";
import type { BBox, Quad } from "../types";
import { bboxFromQuad } from "../lib";
import {
  addFormField, addLayoutElement, addLine, lineParentFor,
} from "../editor/commands";
import { lineParent, snapChildToInk } from "../editor/snapToInk";

type Pt = [number, number];

const quadFromBBox = (b: BBox): Quad =>
  [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]] as Quad;

function iou(a: BBox, b: BBox): number {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const area = (x: BBox) => (x[2] - x[0]) * (x[3] - x[1]);
  const uni = area(a) + area(b) - inter;
  return uni > 0 ? inter / uni : 0;
}

function valueOf(d: any, elementId: number, fieldIndex: number) {
  const t = d.layout.find((e: any) => e.id === elementId);
  const v = t?.type === "form_field" ? t.fields[fieldIndex]?.value : null;
  return v && "bbox" in v ? (v as { bbox: BBox; quad?: Quad }) : undefined;
}

interface DrawPreview {
  mode: "box" | "quad" | "polyline";
  points: Pt[];
  bbox: BBox | null;
}

/**
 * Returns handlers + a preview node for whichever drawing tool is active.
 * The Stage calls these. Nothing here mutates state directly — commits go
 * through ``edit()`` so undo/redo works.
 */
export function useDrawingTools() {
  const tool = useStore((s) => s.tool);
  const defaultType = useStore((s) => s.drawTypeDefault);
  const edit = useStore((s) => s.edit);
  const selection = useStore((s) => s.selection);
  const setSelection = useStore((s) => s.setSelection);
  const valueTarget = useStore((s) => s.valueTarget);
  const setValueTarget = useStore((s) => s.setValueTarget);
  const setTool = useStore((s) => s.setTool);

  const start = useRef<Pt | null>(null);
  const [preview, setPreview] = useState<DrawPreview | null>(null);
  const [quadPointsCollected, setQuadPointsCollected] = useState<Pt[]>([]);
  const [formStage, setFormStage] = useState<"label" | "value">("label");
  const [formLabelBox, setFormLabelBox] = useState<BBox | null>(null);

  const active = tool !== "select" && tool !== "pan" && tool !== "reading-order";

  function onPointerDown(pt: Pt) {
    if (!active) return;

    if (tool === "draw-quad") {
      const next = [...quadPointsCollected, pt];
      if (next.length === 4) {
        const quad = next as unknown as Quad;
        const bbox = bboxFromQuad(quad);
        commitQuad(quad, bbox);
        setQuadPointsCollected([]);
        setPreview(null);
      } else {
        setQuadPointsCollected(next);
        setPreview({ mode: "polyline", points: next, bbox: null });
      }
      return;
    }

    start.current = pt;
    setPreview({ mode: "box", points: [pt, pt], bbox: null });
  }

  function onPointerMove(pt: Pt) {
    if (!active) return;
    if (tool === "draw-quad") return;
    if (!start.current) return;
    const s = start.current;
    const bbox: BBox = [
      Math.min(s[0], pt[0]), Math.min(s[1], pt[1]),
      Math.max(s[0], pt[0]), Math.max(s[1], pt[1]),
    ];
    setPreview({ mode: "box", points: [s, pt], bbox });
  }

  function onPointerUp(pt: Pt, mods: { alt: boolean } = { alt: false }) {
    if (!active) return;
    if (tool === "draw-quad") return;
    if (!start.current) return;
    const s = start.current;
    start.current = null;
    const bbox: BBox = [
      Math.floor(Math.min(s[0], pt[0])), Math.floor(Math.min(s[1], pt[1])),
      Math.ceil(Math.max(s[0], pt[0])), Math.ceil(Math.max(s[1], pt[1])),
    ];
    setPreview(null);

    if (bbox[2] - bbox[0] < 6 || bbox[3] - bbox[1] < 6) return;

    if (tool === "form-field") {
      // "Draw value box" on an existing field: this box is that field's value
      if (valueTarget) {
        const { elementId, fieldIndex } = valueTarget;
        let ok = false;
        edit((d) => {
          const t = d.layout.find((e) => e.id === elementId) as any;
          const f = t?.type === "form_field" ? t.fields[fieldIndex] : undefined;
          if (!f) return;
          f.value = {
            text: "", bbox, quad: quadFromBBox(bbox), handwritten: false, empty: true,
          };
          // the field's region grows to hold its value
          t.bbox = [
            Math.min(t.bbox[0], bbox[0]), Math.min(t.bbox[1], bbox[1]),
            Math.max(t.bbox[2], bbox[2]), Math.max(t.bbox[3], bbox[3]),
          ];
          t.quad = quadFromBBox(t.bbox);
          ok = true;
        });
        setValueTarget(null);
        setTool("select");
        if (ok) {
          setSelection({ kind: "field", elementId, fieldIndex });
          void snapChildToInk({ get: (d) => valueOf(d, elementId, fieldIndex), parent: (d) => d.layout.find((e) => e.id === elementId)?.bbox as BBox | undefined }, bbox, mods.alt);
        }
        return;
      }
      if (formStage === "label") {
        setFormLabelBox(bbox);
        setFormStage("value");
        return;
      }
      // Second box: value box. Commit.
      const labelBox = formLabelBox!;
      let newId = -1;
      edit((d) => {
        newId = addFormField(d, labelBox, bbox);
        // the second box drawn over (almost) the same spot as the label means
        // "no value". Compare the whole box: a value written right under its
        // label has the same left/right edges and must be kept.
        if (iou(labelBox, bbox) > 0.8) {
          const el = d.layout.find((e) => e.id === newId) as any;
          el.fields[0].value = null;
        }
      });
      setFormLabelBox(null);
      setFormStage("label");
      if (newId >= 0) setSelection({ kind: "field", elementId: newId, fieldIndex: 0 });
      return;
    }

    if (tool === "line") {
      let idx = -1;
      edit((d) => {
        // the selected region, the region of the selected line / cell, or the
        // region the box is drawn in — not a new paragraph every time
        idx = addLine(d, bbox, lineParentFor(d, bbox, selection));
        setSelection({ kind: "line", index: idx });
      });
      // tighten the drawn box to the text ink inside it (hold Alt to skip)
      if (idx >= 0) {
        void snapChildToInk({ get: (d) => d.lines[idx], parent: lineParent(idx) }, bbox, mods.alt);
      }
      return;
    }

    // Everything else is "create layout element of tool's type".
    const typeMap: Record<string, string> = {
      "draw-box": defaultType,
      "table": "table",
      "list": "list",
      "stamp": "stamp",
      "signature": "signature",
      "logo": "logo",
      "figure": "figure",
      "checkbox": "form_field",     // creates a form field, empty until options added
      "fingerprint": "figure",      // treat as a figure-shaped extra_region
    };
    const type = typeMap[tool] ?? "paragraph";

    edit((d) => {
      const id = addLayoutElement(d, type, bbox);
      if (tool === "fingerprint") {
        // Also push an extra_region placeholder.
        d.extra_regions.push({
          type: "fingerprint", hand: "right",
          bbox, quad: [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
        });
      }
      setSelection({ kind: "layout", id });
    });
  }

  function commitQuad(quad: Quad, bbox: BBox) {
    const typeMap: Record<string, string> = { "draw-quad": defaultType };
    const type = typeMap[tool] ?? "paragraph";
    edit((d) => {
      const id = addLayoutElement(d, type, bbox);
      const el = d.layout.find((e) => e.id === id);
      if (el) el.quad = quad;
      setSelection({ kind: "layout", id });
    });
  }

  function Preview() {
    if (!preview) {
      if (tool === "draw-quad" && quadPointsCollected.length > 0) {
        const flat = quadPointsCollected.flatMap((p) => [p[0], p[1]]);
        return <Line points={flat} stroke="#f59e0b" strokeWidth={2} dash={[4, 4]} />;
      }
      return null;
    }
    if (preview.mode === "polyline") {
      const flat = preview.points.flatMap((p) => [p[0], p[1]]);
      return <Line points={flat} stroke="#f59e0b" strokeWidth={2} dash={[4, 4]} />;
    }
    if (!preview.bbox) return null;
    const [x0, y0, x1, y1] = preview.bbox;
    return (
      <Rect
        x={x0} y={y0} width={x1 - x0} height={y1 - y0}
        stroke="#f59e0b" strokeWidth={1.5} dash={[6, 4]}
        fill="#f59e0b22"
      />
    );
  }

  return { onPointerDown, onPointerMove, onPointerUp, Preview, active };
}