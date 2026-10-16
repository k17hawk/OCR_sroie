import { useRef, useState } from "react";
import { Circle, Group, Line, Rect } from "react-konva";
import type Konva from "konva";
import { useStore } from "../store";
import type { BBox, Doc, Quad } from "../types";
import { resizeLayoutElement } from "../editor/commands";
import { hitBridge } from "./hitBridge";
import { lineParent, snapChildToInk } from "../editor/snapToInk";

/**
 * A click (no drag) on a handle's grab circle is passed on to HitLayer, so
 * the box under it still gets selected. Before, these invisible circles
 * swallowed clicks on any small / nested box near the selected one.
 */
function passClick(e: Konva.KonvaEventObject<MouseEvent>) {
  if (e.evt.button !== 0) return;
  e.cancelBubble = true;
  const pt = e.target.getParent()?.getRelativePointerPosition();
  if (pt && hitBridge.click) {
    hitBridge.click([pt.x, pt.y], { alt: e.evt.altKey, shift: e.evt.shiftKey });
  }
}
function passContextMenu(e: Konva.KonvaEventObject<PointerEvent>) {
  e.evt.preventDefault();
  e.cancelBubble = true;
  const pt = e.target.getParent()?.getRelativePointerPosition();
  if (pt && hitBridge.contextMenu) hitBridge.contextMenu([pt.x, pt.y], e.evt.clientX, e.evt.clientY);
}

function bboxOf(quad: Quad): BBox {
  return [
    Math.floor(Math.min(...quad.map((p) => p[0]))),
    Math.floor(Math.min(...quad.map((p) => p[1]))),
    Math.ceil(Math.max(...quad.map((p) => p[0]))),
    Math.ceil(Math.max(...quad.map((p) => p[1]))),
  ];
}

function quadOf(b: BBox): Quad {
  const [x0, y0, x1, y1] = b;
  return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] as Quad;
}

/** Is this quad an axis-aligned rectangle in TL, TR, BR, BL order? */
function isAxisAligned(q: Quad): boolean {
  const [tl, tr, br, bl] = q;
  const EPS = 0.5;
  return (
    Math.abs(tr[1] - tl[1]) < EPS &&
    Math.abs(br[1] - bl[1]) < EPS &&
    Math.abs(bl[0] - tl[0]) < EPS &&
    Math.abs(br[0] - tr[0]) < EPS
  );
}

const MIN_SIZE = 4;

// Rectangle handles: which bbox edges each one moves.
type Edge = { x0?: true; y0?: true; x1?: true; y1?: true };
const RECT_HANDLES: { id: string; edges: Edge; at: (b: BBox) => [number, number] }[] = [
  { id: "tl", edges: { x0: true, y0: true }, at: (b) => [b[0], b[1]] },
  { id: "t",  edges: { y0: true },           at: (b) => [(b[0] + b[2]) / 2, b[1]] },
  { id: "tr", edges: { x1: true, y0: true }, at: (b) => [b[2], b[1]] },
  { id: "r",  edges: { x1: true },           at: (b) => [b[2], (b[1] + b[3]) / 2] },
  { id: "br", edges: { x1: true, y1: true }, at: (b) => [b[2], b[3]] },
  { id: "b",  edges: { y1: true },           at: (b) => [(b[0] + b[2]) / 2, b[3]] },
  { id: "bl", edges: { x0: true, y1: true }, at: (b) => [b[0], b[3]] },
  { id: "l",  edges: { x0: true },           at: (b) => [b[0], (b[1] + b[3]) / 2] },
];

function applyEdges(start: BBox, edges: Edge, x: number, y: number): BBox {
  let [x0, y0, x1, y1] = start;
  if (edges.x0) x0 = x;
  if (edges.x1) x1 = x;
  if (edges.y0) y0 = y;
  if (edges.y1) y1 = y;
  // allow dragging past the opposite edge (flip), then enforce a min size
  let nx0 = Math.round(Math.min(x0, x1));
  let nx1 = Math.round(Math.max(x0, x1));
  let ny0 = Math.round(Math.min(y0, y1));
  let ny1 = Math.round(Math.max(y0, y1));
  if (nx1 - nx0 < MIN_SIZE) nx1 = nx0 + MIN_SIZE;
  if (ny1 - ny0 < MIN_SIZE) ny1 = ny0 + MIN_SIZE;
  return [nx0, ny0, nx1, ny1];
}

/**
 * Resize handles for any quad.
 *
 * - Axis-aligned quad: 8 handles (4 corners + 4 edge midpoints) that keep
 *   the shape a rectangle — the opposite corner/edge stays put.
 * - Rotated / skewed quad, or Alt held when the drag starts: 4 free corner
 *   handles that move one point at a time.
 */
export function QuadHandlesFor({
  quad,
  onChange,
}: {
  quad: Quad;
  /** `alt`: Alt was held when the drag ended (child boxes: don't snap to ink) */
  onChange: (q: Quad, mods: { alt: boolean }) => void;
}) {
  const scale = useStore((s) => s.scale);
  const [preview, setPreview] = useState<Quad | null>(null);
  // refs so drag handlers never read a stale closure
  const previewRef = useRef<Quad | null>(null);
  const startRef = useRef<{ quad: Quad; bbox: BBox; free: boolean } | null>(null);
  // the handle being dragged keeps its drag-start props so React doesn't
  // fight Konva for its position mid-drag
  const [active, setActive] = useState<{ id: string; x: number; y: number } | null>(null);

  const k = Math.max(scale, 0.05);
  const r = 6 / k;
  const hitR = 9 / k; // smaller grab area: less of the page is covered by handles
  const strokeW = 1.5 / k;

  const shown = preview ?? quad;
  const rectMode = isAxisAligned(quad);

  const update = (q: Quad) => {
    previewRef.current = q;
    setPreview(q);
  };

  const begin = (
    e: Konva.KonvaEventObject<DragEvent>,
    id: string,
    free: boolean,
  ) => {
    e.cancelBubble = true;
    const { x, y } = e.target.position();
    startRef.current = { quad, bbox: bboxOf(quad), free };
    previewRef.current = quad;
    setActive({ id, x, y });
    setPreview(quad);
  };

  const finish = (
    e: Konva.KonvaEventObject<DragEvent>,
    resetTo: (q: Quad) => [number, number],
  ) => {
    e.cancelBubble = true;
    const final = previewRef.current;
    previewRef.current = null;
    startRef.current = null;
    setPreview(null);
    setActive(null);
    if (final) {
      // snap the dragged node onto where the handle belongs now
      const [hx, hy] = resetTo(final);
      e.target.position({ x: hx, y: hy });
      onChange(final, { alt: !!(e.evt as MouseEvent | undefined)?.altKey });
    }
  };

  const outline = (
    <Line
      points={shown.flatMap((p) => [p[0], p[1]])}
      closed
      stroke="#f59e0b"
      strokeWidth={strokeW}
      dash={[6 / k, 4 / k]}
      listening={false}
    />
  );

  const dot = (x: number, y: number, key: string, square = false) =>
    square ? (
      <Rect
        key={key}
        x={x - r * 0.8}
        y={y - r * 0.8}
        width={r * 1.6}
        height={r * 1.6}
        fill="#f59e0b"
        stroke="#ffffff"
        strokeWidth={strokeW}
        listening={false}
      />
    ) : (
      <Circle
        key={key}
        x={x}
        y={y}
        radius={r}
        fill="#f59e0b"
        stroke="#ffffff"
        strokeWidth={strokeW}
        listening={false}
      />
    );

  if (rectMode) {
    const b = bboxOf(shown);
    return (
      <>
        {outline}
        {RECT_HANDLES.filter((hd) => {
          // on a thin box the edge-midpoint handles would cover its middle;
          // corners still resize it
          const wPx = (b[2] - b[0]) * k, hPx = (b[3] - b[1]) * k;
          if ((hd.id === "t" || hd.id === "b") && hPx < 40) return false;
          if ((hd.id === "l" || hd.id === "r") && wPx < 40) return false;
          return true;
        }).map((hd) => {
          const [px, py] = hd.at(b);
          const [hx, hy] =
            active?.id === hd.id ? [active.x, active.y] : [px, py];
          const isCorner = hd.id.length === 2;
          return (
            <Group key={hd.id}>
              <Circle
                x={hx}
                y={hy}
                radius={hitR}
                fill="rgba(0,0,0,0.001)"
                draggable
                dragDistance={3}
                onClick={passClick}
                onContextMenu={passContextMenu}
                onMouseEnter={(e) => {
                  const st = e.target.getStage();
                  if (st) {
                    st.container().style.cursor =
                      hd.id === "t" || hd.id === "b" ? "ns-resize"
                      : hd.id === "l" || hd.id === "r" ? "ew-resize"
                      : hd.id === "tl" || hd.id === "br" ? "nwse-resize"
                      : "nesw-resize";
                  }
                }}
                onMouseLeave={(e) => {
                  const st = e.target.getStage();
                  if (st) st.container().style.cursor = "default";
                }}
                onDragStart={(e) =>
                  begin(e, hd.id, isCorner && !!(e.evt as MouseEvent)?.altKey)
                }
                onDragMove={(e) => {
                  e.cancelBubble = true;
                  const s = startRef.current;
                  if (!s) return;
                  const { x, y } = e.target.position();
                  if (s.free) {
                    const idx = ["tl", "tr", "br", "bl"].indexOf(hd.id);
                    const next = [...s.quad] as Quad;
                    next[idx] = [Math.round(x), Math.round(y)];
                    update(next);
                    return;
                  }
                  // edge handles are constrained to their axis
                  const [sx, sy] = hd.at(s.bbox);
                  const cx = hd.edges.x0 || hd.edges.x1 ? x : sx;
                  const cy = hd.edges.y0 || hd.edges.y1 ? y : sy;
                  if (!(hd.edges.x0 || hd.edges.x1) || !(hd.edges.y0 || hd.edges.y1)) {
                    e.target.position({ x: cx, y: cy });
                  }
                  update(quadOf(applyEdges(s.bbox, hd.edges, cx, cy)));
                }}
                onDragEnd={(e) => finish(e, (q) => hd.at(bboxOf(q)))}
              />
              {dot(px, py, `d-${hd.id}`, !isCorner)}
            </Group>
          );
        })}
      </>
    );
  }

  // Free-corner mode (rotated / skewed quads)
  return (
    <>
      {outline}
      {shown.map((p, i) => (
        <Group key={i}>
          <Circle
            x={active?.id === String(i) ? active.x : p[0]}
            y={active?.id === String(i) ? active.y : p[1]}
            radius={hitR}
            fill="rgba(0,0,0,0.001)"
            draggable
            dragDistance={3}
            onClick={passClick}
            onContextMenu={passContextMenu}
            onDragStart={(e) => begin(e, String(i), true)}
            onDragMove={(e) => {
              e.cancelBubble = true;
              const s = startRef.current;
              if (!s) return;
              const { x, y } = e.target.position();
              const next = [...(previewRef.current ?? s.quad)] as Quad;
              next[i] = [Math.round(x), Math.round(y)];
              update(next);
            }}
            onDragEnd={(e) => finish(e, (q) => q[i] as [number, number])}
          />
          {dot(p[0], p[1], `d-${i}`)}
        </Group>
      ))}
    </>
  );
}

export function QuadHandles({ elementId }: { elementId: number }) {
  const doc = useStore((s) => s.doc);
  const edit = useStore((s) => s.edit);
  const el = doc?.layout.find((e) => e.id === elementId);
  if (!el) return null;

  return (
    <QuadHandlesFor
      quad={el.quad as Quad}
      onChange={(q) => edit((d) => {
        const target = d.layout.find((it) => it.id === elementId);
        if (!target) return;
        // Rectangle in, rectangle out: scale the whole subtree (lines,
        // cells, fields) so children stay inside the region.
        if (isAxisAligned(q)) {
          resizeLayoutElement(d, elementId, bboxOf(q));
        } else {
          target.quad = q;
          target.bbox = bboxOf(q);
        }
      })}
    />
  );
}

export function LineHandles({ lineIndex }: { lineIndex: number }) {
  const doc = useStore((s) => s.doc);
  const edit = useStore((s) => s.edit);
  const ln = doc?.lines[lineIndex];
  if (!ln) return null;

  return (
    <QuadHandlesFor
      quad={ln.quad as Quad}
      onChange={(q, { alt }) => {
        edit((d) => {
          const target = d.lines[lineIndex];
          if (!target) return;
          target.quad = q;
          target.bbox = bboxOf(q);
        });
        // tighten to the text ink inside it (hold Alt to keep it as dragged)
        void snapChildToInk(
          { get: (d) => d.lines[lineIndex], parent: lineParent(lineIndex) },
          bboxOf(q), alt,
        );
      }}
    />
  );
}

export function CellHandles({
  tableIndex, cellId,
}: { tableIndex: number; cellId: number }) {
  const doc = useStore((s) => s.doc);
  const edit = useStore((s) => s.edit);
  const cell = doc?.tables[tableIndex]?.cells.find((c) => c.id === cellId);
  if (!cell) return null;

  return (
    <QuadHandlesFor
      quad={cell.quad as Quad}
      onChange={(q) => edit((d) => {
        const c = d.tables[tableIndex]?.cells.find((x) => x.id === cellId);
        if (!c) return;
        c.quad = q;
        c.bbox = bboxOf(q);
      })}
    />
  );
}

export function ExtraRegionHandles({ index }: { index: number }) {
  const doc = useStore((s) => s.doc);
  const edit = useStore((s) => s.edit);
  const er = doc?.extra_regions[index];
  if (!er) return null;

  return (
    <QuadHandlesFor
      quad={er.quad as Quad}
      onChange={(q) => edit((d) => {
        const target = d.extra_regions[index];
        if (!target) return;
        target.quad = q;
        target.bbox = bboxOf(q);
      })}
    />
  );
}

export function FieldHandles({
  elementId, fieldIndex,
}: { elementId: number; fieldIndex: number }) {
  const doc = useStore((s) => s.doc);
  const edit = useStore((s) => s.edit);
  const el = doc?.layout.find((e) => e.id === elementId);
  if (!el || el.type !== "form_field") return null;
  const field = el.fields[fieldIndex];
  if (!field) return null;

  return (
    <>
      <QuadHandlesFor
        quad={field.label.quad as Quad}
        onChange={(q, { alt }) => {
          edit((d) => {
            const t = d.layout.find((x) => x.id === elementId);
            if (!t || t.type !== "form_field") return;
            t.fields[fieldIndex].label.quad = q;
            t.fields[fieldIndex].label.bbox = bboxOf(q);
          });
          void snapChildToInk({ get: (d) => fieldPart(d, elementId, fieldIndex, "label"), parent: regionBox(elementId) }, bboxOf(q), alt);
        }}
      />
      {field.value && "quad" in field.value && (
        <QuadHandlesFor
          quad={field.value.quad as Quad}
          onChange={(q, { alt }) => {
            edit((d) => {
              const t = d.layout.find((x) => x.id === elementId);
              if (!t || t.type !== "form_field") return;
              const v = t.fields[fieldIndex].value;
              if (!v || !("quad" in v)) return;
              v.quad = q;
              v.bbox = bboxOf(q);
            });
            void snapChildToInk({ get: (d) => fieldPart(d, elementId, fieldIndex, "value"), parent: regionBox(elementId) }, bboxOf(q), alt);
          }}
        />
      )}
    </>
  );
}

/** A form field's label or value box (the value only when it has one). */
function fieldPart(d: Doc, elementId: number, fieldIndex: number, part: "label" | "value") {
  const t = d.layout.find((x) => x.id === elementId);
  if (!t || t.type !== "form_field") return undefined;
  const f = t.fields[fieldIndex];
  if (!f) return undefined;
  const p = part === "label" ? f.label : f.value;
  return p && "bbox" in p ? (p as { bbox: BBox; quad?: Quad }) : undefined;
}

const regionBox = (id: number) => (d: Doc): BBox | undefined =>
  d.layout.find((e) => e.id === id)?.bbox as BBox | undefined;
