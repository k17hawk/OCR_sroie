import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Stage as KStage, Layer, Group, Rect } from "react-konva";
import type Konva from "konva";

import { useStore } from "../store";
import {
  ImageLayer,
  LayoutLayer,
  LineLayer,
  TableLayer,
  FormFieldLayer,
  ExtraRegionLayer,
  ReadingOrderLayer,
  BlockLayer,
} from "./layers";
import { HitLayer } from "./HitLayer";
import { PickMenu } from "./PickMenu";
import { useDrawingTools } from "./tools";
import {
  QuadHandles,
  LineHandles,
  CellHandles,
  ExtraRegionHandles,
  FieldHandles,
} from "./handles";

const MIN_SCALE = 0.05;
const MAX_SCALE = 8;

export function Stage() {
  const doc = useStore((s) => s.doc);
  const layers = useStore((s) => s.layers);
  const overlay = useStore((s) => s.overlay);
  const scale = useStore((s) => s.scale);
  const offset = useStore((s) => s.offset);
  const setViewport = useStore((s) => s.setViewport);
  const fitRequest = useStore((s) => s.fitRequest);
  const selection = useStore((s) => s.selection);
  const setSelection = useStore((s) => s.setSelection);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<Konva.Stage | null>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [spaceDown, setSpaceDown] = useState(false);

  const tools = useDrawingTools();
  const tool = useStore((s) => s.tool);

  // Measure the canvas area. While a page is loading the canvas used to be
  // removed; if the panel around it has no fixed height it collapsed to 0,
  // the new page was fitted into a 0-px-tall canvas (zoom stuck at 0.05) and
  // NOTHING could be clicked. Now 0-size readings are ignored (we fall back
  // to the parent / window size) and the page is refitted once the real
  // size is known.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => {
      const parent = el.parentElement;
      const w = el.clientWidth || parent?.clientWidth || window.innerWidth;
      const h = el.clientHeight || parent?.clientHeight || window.innerHeight;
      if (w < 50 || h < 50) return; // collapsed / hidden: keep the last good size
      setSize((old) => (old.w === w && old.h === h ? old : { w, h }));
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    if (el.parentElement) ro.observe(el.parentElement);
    window.addEventListener("resize", measure);
    measure();
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);

  useEffect(() => {
    if (!doc) return;
    // Pages with width/height 0 or missing: fit to the boxes instead.
    let pw = Number(doc.width) || 0;
    let ph = Number(doc.height) || 0;
    if (!pw || !ph) {
      for (const el of doc.layout ?? []) {
        pw = Math.max(pw, el.bbox[2]);
        ph = Math.max(ph, el.bbox[3]);
      }
      for (const ln of doc.lines ?? []) {
        pw = Math.max(pw, ln.bbox[2]);
        ph = Math.max(ph, ln.bbox[3]);
      }
    }
    if (!pw || !ph) return;
    const pad = 40;
    const sx = (size.w - pad * 2) / pw;
    const sy = (size.h - pad * 2) / ph;
    const fit = Math.min(sx, sy);
    if (!Number.isFinite(fit) || fit <= 0) return; // not measured yet
    const s = Math.max(MIN_SCALE, Math.min(MAX_SCALE, fit));
    setViewport(s, {
      x: (size.w - pw * s) / 2,
      y: (size.h - ph * s) / 2,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitRequest, doc?.document_id, doc?.width, doc?.height, size.w, size.h]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !spaceDown) {
        const t = e.target as HTMLElement | null;
        if (
          t &&
          (t.tagName === "INPUT" ||
            t.tagName === "TEXTAREA" ||
            t.isContentEditable)
        )
          return;
        e.preventDefault();
        setSpaceDown(true);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") setSpaceDown(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [spaceDown]);

  const onWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const oldScale = scale;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const mouseX = (pointer.x - offset.x) / oldScale;
    const mouseY = (pointer.y - offset.y) / oldScale;

    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const factor = 1.1;
    const newScale = Math.max(
      MIN_SCALE,
      Math.min(MAX_SCALE, direction > 0 ? oldScale * factor : oldScale / factor),
    );

    setViewport(newScale, {
      x: pointer.x - mouseX * newScale,
      y: pointer.y - mouseY * newScale,
    });
  };

  const isPanningRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0, ox: 0, oy: 0 });

  const screenToImage = (p: { x: number; y: number }): [number, number] => [
    (p.x - offset.x) / scale,
    (p.y - offset.y) / scale,
  ];

  const onMouseDownStage = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const wantPan = spaceDown || e.evt.button === 1;
    if (wantPan) {
      isPanningRef.current = true;
      const p = stageRef.current?.getPointerPosition();
      panStartRef.current = {
        x: p?.x ?? 0,
        y: p?.y ?? 0,
        ox: offset.x,
        oy: offset.y,
      };
      return;
    }

    if (tools.active && e.evt.button === 0) {
      const p = stageRef.current?.getPointerPosition();
      if (p) tools.onPointerDown(screenToImage(p));
      return;
    }
  };

  const onMouseMoveStage = () => {
    if (isPanningRef.current) {
      const p = stageRef.current?.getPointerPosition();
      if (!p) return;
      const dx = p.x - panStartRef.current.x;
      const dy = p.y - panStartRef.current.y;
      setViewport(scale, {
        x: panStartRef.current.ox + dx,
        y: panStartRef.current.oy + dy,
      });
      return;
    }

    if (tools.active) {
      const p = stageRef.current?.getPointerPosition();
      if (p) tools.onPointerMove(screenToImage(p));
    }
  };

  const onMouseUpStage = (e?: Konva.KonvaEventObject<MouseEvent>) => {
    if (isPanningRef.current) {
      isPanningRef.current = false;
      return;
    }
    if (tools.active) {
      const p = stageRef.current?.getPointerPosition();
      if (p) tools.onPointerUp(screenToImage(p), { alt: !!e?.evt?.altKey });
    }
  };

  const cursor = spaceDown
    ? "grab"
    : tools.active
      ? "crosshair"
      : "default";

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full overflow-hidden bg-slate-950"
      style={{ cursor, minHeight: 200 }}
    >
      {doc && (
        <KStage
          ref={stageRef}
          width={size.w}
          height={size.h}
          onWheel={onWheel}
          onMouseDown={onMouseDownStage}
          onMouseMove={onMouseMoveStage}
          onMouseUp={onMouseUpStage}
          onMouseLeave={onMouseUpStage}
          onContextMenu={(e) => e.evt.preventDefault()}
          onClick={(e) => {
            // Only clear selection on empty-canvas clicks in select-type
            // modes. While a drawing tool is active, the Stage's click fires
            // right after the tool commits a shape — for Quad that commit
            // happens on the 4th mousedown, so this used to wipe the new
            // selection and leave the right panel empty.
            if (!tools.active && e.target === e.target.getStage()) {
              setSelection(null);
            }
          }}
        >
          {/* Decorative layer. NEVER listens: HitLayer does all picking.
              (It used to listen in select mode, which made Konva rebuild a
              hit canvas for every box/text on each change — slow, and its
              click handlers could never fire anyway, HitLayer is on top.) */}
          <Layer listening={false}>
            <Group scaleX={scale} scaleY={scale} x={offset.x} y={offset.y}>
              {layers.image && <ImageLayer doc={doc} />}
              {!layers.image && (
                <Rect x={0} y={0} width={doc.width} height={doc.height} fill="#ffffff" />
              )}
              {/* every annotation drawn at the opacity chosen in Layers, so
                  the scan stays readable; the selected box (HitLayer) is not
                  dimmed */}
              <Group opacity={overlay.opacity} listening={false}>
                {layers.layout && <LayoutLayer doc={doc} />}
                {layers.blocks && <BlockLayer doc={doc} />}
                {layers.lines && <LineLayer doc={doc} />}
                {layers.tables && <TableLayer doc={doc} />}
                {layers.formFields && <FormFieldLayer doc={doc} />}
                {layers.checkboxes && <ExtraRegionLayer doc={doc} kind="checkbox" />}
                {layers.extraRegions && <ExtraRegionLayer doc={doc} kind="fingerprint" />}
              </Group>
              {(layers.readingOrder || tool === "reading-order") && (
                <ReadingOrderLayer doc={doc} />
              )}
              <tools.Preview />
            </Group>
          </Layer>

          {/* Select-mode layer. Konva hit-tests top-down across layers, so
              this must be the LAST <Layer> rendered. Inside it, order also
              matters: handles on top so corner drags win over HitLayer's
              region drag. */}
          <Layer listening={tool === "select" || tool === "reading-order"}>
            <Group scaleX={scale} scaleY={scale} x={offset.x} y={offset.y}>
              <HitLayer doc={doc} />

              {tool === "select" && selection?.kind === "layout" && (
                <QuadHandles elementId={selection.id} />
              )}
              {tool === "select" && selection?.kind === "line" && (
                <LineHandles lineIndex={selection.index} />
              )}
              {tool === "select" && selection?.kind === "cell" && (
                <CellHandles tableIndex={selection.tableIndex} cellId={selection.cellId} />
              )}
              {tool === "select" && selection?.kind === "extraRegion" && (
                <ExtraRegionHandles index={selection.index} />
              )}
              {tool === "select" && selection?.kind === "field" && (
                <FieldHandles elementId={selection.elementId} fieldIndex={selection.fieldIndex} />
              )}
            </Group>
          </Layer>
        </KStage>
      )}
      {!doc && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
          Loading page…
        </div>
      )}
      <PickMenu />
    </div>
  );
}
