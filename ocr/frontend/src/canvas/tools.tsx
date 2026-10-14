import { useRef, useState } from "react";
import { Rect, Line } from "react-konva";

import { useStore } from "../store";
import type { BBox, Quad } from "../types";
import { bboxFromQuad } from "../lib";
import {
  addFormField, addLayoutElement, addLine,
} from "../editor/commands";

type Pt = [number, number];

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

  function onPointerUp(pt: Pt) {
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
      if (formStage === "label") {
        setFormLabelBox(bbox);
        setFormStage("value");
        return;
      }
      // Second click: value box. Commit.
      const labelBox = formLabelBox!;
      edit((d) => {
        const id = addFormField(d, labelBox, bbox);
        // If the two boxes overlap heavily, treat as "no value".
        if (
          Math.abs(labelBox[0] - bbox[0]) < 4 &&
          Math.abs(labelBox[2] - bbox[2]) < 4
        ) {
          const el = d.layout.find((e) => e.id === id) as any;
          el.fields[0].value = null;
        }
      });
      setFormLabelBox(null);
      setFormStage("label");
      return;
    }

    if (tool === "line") {
      const parent = selection?.kind === "layout" ? selection.id : null;
      edit((d) => {
        const idx = addLine(d, bbox, parent);
        setSelection({ kind: "line", index: idx });
      });
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