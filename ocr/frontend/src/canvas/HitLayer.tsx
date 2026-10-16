import { useEffect, useMemo, useRef, useState } from "react";
import { Rect, Group, Text } from "react-konva";
import type Konva from "konva";
import { useStore, type Selection } from "../store";
import type { BBox, Doc } from "../types";
import {
  translateLayoutElement,
  parentLayoutOf,
  appendReadingOrder,
  removeFromReadingOrder,
  isReadingOrderLocked,
} from "../editor/commands";
import { pickMenu, usePickMenu, describeSelection } from "./pickMenuStore";
import { hitBridge } from "./hitBridge";

type Sel = Exclude<Selection, null>;

type HitItem = {
  key: string;
  bbox: BBox;
  selection: Sel;
  role: "box" | "child";
  layoutId: number | null;
  pad: number;
};

const area = (b: BBox) => Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
const keyOf = (s: Selection) => (s ? JSON.stringify(s) : "");
const HALO = "rgba(255,255,255,0.95)";
/** clicks closer than this (screen px) to the last click count as "same spot" */
const SAME_SPOT_PX = 6;

/**
 * Area the click-catcher must cover: the page, grown to include every box.
 * Pages whose width/height are 0 or missing (or smaller than their boxes)
 * used to get a click-catcher of no size, so nothing could be selected
 * even though the boxes were drawn.
 */
function clickExtent(doc: Doc, items: HitItem[]): [number, number] {
  let w = Number(doc.width) || 0;
  let h = Number(doc.height) || 0;
  for (const it of items) {
    if (Number.isFinite(it.bbox[2])) w = Math.max(w, it.bbox[2]);
    if (Number.isFinite(it.bbox[3])) h = Math.max(h, it.bbox[3]);
  }
  return [w, h];
}

/** Every selectable thing on the page, honouring the layer toggles. */
function collectItems(doc: Doc, layers: Record<string, boolean>): HitItem[] {
  const items: HitItem[] = [];
  const show = (k: string) => layers[k] !== false;
  const layoutIds = new Set(doc.layout.map((e: any) => e.id));

  for (const el of doc.layout) {
    if (show("layout")) {
      items.push({
        key: `layout-${el.id}`,
        bbox: el.bbox,
        role: "box",
        layoutId: el.id,
        selection: { kind: "layout", id: el.id },
        pad: 4,
      });
    }
    if (el.type === "form_field" && show("formFields")) {
      el.fields.forEach((f: any, fi: number) => {
        const sel: Sel = { kind: "field", elementId: el.id, fieldIndex: fi };
        // a label without its own box (typed, or a line / cell / region) has
        // nothing to click here; its value area and option squares still do
        if (f.label?.bbox) {
          items.push({
            key: `field-${el.id}-${fi}`,
            bbox: f.label.bbox as BBox,
            role: "child",
            layoutId: el.id,
            selection: sel,
            pad: 1,
          });
        }
        if (f.value && "bbox" in f.value && f.value.bbox) {
          items.push({
            key: `fieldval-${el.id}-${fi}`,
            bbox: f.value.bbox as BBox,
            role: "child",
            layoutId: el.id,
            selection: sel,
            pad: 1,
          });
        }
        if (Array.isArray(f.options)) {
          f.options.forEach((opt: any, oi: number) => {
            if (!opt.box) return;
            items.push({
              key: `fieldopt-${el.id}-${fi}-${oi}`,
              bbox: opt.box as BBox,
              role: "child",
              layoutId: el.id,
              selection: sel,
              pad: 2,
            });
          });
        }
      });
    }
  }

  if (show("lines")) {
    doc.lines.forEach((ln: any, i: number) => {
      const parent =
        ln.block_id != null && layoutIds.has(ln.block_id) ? ln.block_id : null;
      items.push({
        key: `line-${i}`,
        bbox: ln.bbox,
        role: parent != null ? "child" : "box",
        layoutId: parent,
        selection: { kind: "line", index: i },
        pad: 0,
      });
    });
  }

  doc.extra_regions.forEach((er: any, i: number) => {
    const on =
      er.type === "checkbox" ? show("checkboxes") : show("extraRegions");
    if (!on) return;
    items.push({
      key: `extra-${i}`,
      bbox: er.bbox,
      role: "box",
      layoutId: null,
      selection: { kind: "extraRegion", index: i },
      pad: 2,
    });
  });

  if (show("tables")) {
    const owner = new Map<number, number>();
    for (const el of doc.layout) {
      if (el.type === "table") owner.set((el as any).table_index, el.id);
    }
    doc.tables.forEach((t: any, ti: number) =>
      t.cells.forEach((c: any) => {
        const parent = owner.get(ti) ?? null;
        items.push({
          key: `cell-${ti}-${c.id}`,
          bbox: c.bbox,
          role: parent != null ? "child" : "box",
          layoutId: parent,
          selection: { kind: "cell", tableIndex: ti, cellId: c.id },
          pad: 0,
        });
      }),
    );
  }
  return items;
}

/**
 * Everything under a point, in the order repeated clicks walk through it:
 *   innermost region → its lines / cells / fields under the pointer
 *   → next region out → its children → … → (wraps)
 * `hits` must already be sorted by area, smallest first. A selection that
 * has several boxes (a field's label + value) appears once.
 */
function buildStack(hits: HitItem[]): HitItem[] {
  const boxes = hits.filter((h) => h.role === "box");
  const kids = hits.filter((h) => h.role === "child");
  const used = new Set<HitItem>();
  const stack: HitItem[] = [];
  for (const b of boxes) {
    stack.push(b);
    used.add(b);
    if (b.selection.kind !== "layout") continue;
    for (const c of kids) {
      if (!used.has(c) && c.layoutId === b.layoutId) {
        stack.push(c);
        used.add(c);
      }
    }
  }
  for (const c of kids) if (!used.has(c)) stack.push(c);

  const seen = new Set<string>();
  return stack.filter((it) => {
    const k = keyOf(it.selection);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** Bold outline + optional label chip, drawn in screen px at any zoom. */
function Ring({ bbox, color, k, label }: { bbox: BBox; color: string; k: number; label?: string }) {
  const [x0, y0, x1, y1] = bbox;
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  return (
    <Group listening={false}>
      <Rect
        x={x0} y={y0} width={w} height={h}
        fill={`${color}2e`}
        stroke={HALO} strokeWidth={6} strokeScaleEnabled={false}
        listening={false}
      />
      <Rect
        x={x0} y={y0} width={w} height={h}
        stroke={color} strokeWidth={3} strokeScaleEnabled={false}
        dash={[6 / k, 3 / k]}
        listening={false}
      />
      {label && (
        <Group x={x0} y={y1 + 4 / k} scaleX={1 / k} scaleY={1 / k} listening={false}>
          <Rect
            width={12 + label.length * 6.3} height={19}
            fill="#0f172a" opacity={0.92} cornerRadius={4}
            stroke={color} strokeWidth={2}
          />
          <Text
            x={6} y={4} text={label}
            fontSize={11} fontStyle="bold"
            fontFamily="Inter, system-ui, sans-serif"
            fill="#ffffff"
          />
        </Group>
      )}
    </Group>
  );
}

export function HitLayer({ doc }: { doc: Doc }) {
  const scale = useStore((s) => s.scale);
  const offset = useStore((s) => s.offset);
  const tool = useStore((s) => s.tool);
  const layers = useStore((s) => s.layers) as unknown as Record<string, boolean>;
  const setSelection = useStore((s) => s.setSelection);
  const edit = useStore((s) => s.edit);
  useStore((s) => s.selection); // re-render on selection change
  const { preview: menuPreview } = usePickMenu();

  const [hover, setHover] = useState<{ key: string; pos: number; n: number } | null>(null);
  const [dragPreview, setDragPreview] = useState<{
    bbox: BBox;
    dx: number;
    dy: number;
  } | null>(null);

  const spaceRef = useRef(false);
  const lastClick = useRef<[number, number] | null>(null);
  const down = useRef<{
    sx: number;
    sy: number;
    dragId: number;
    bbox: BBox;
  } | null>(null);

  useEffect(() => {
    const d = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceRef.current = true;
    };
    const u = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceRef.current = false;
    };
    window.addEventListener("keydown", d);
    window.addEventListener("keyup", u);
    return () => {
      window.removeEventListener("keydown", d);
      window.removeEventListener("keyup", u);
    };
  }, []);

  // unregister the handle → HitLayer click bridge when unmounted
  useEffect(() => () => { hitBridge.click = null; hitBridge.contextMenu = null; }, []);

  useEffect(() => {
    lastClick.current = null;
    setHover(null);
    pickMenu.close();
  }, [doc.document_id]);

  const roMode = tool === "reading-order";
  const M = 200;
  const items = useMemo(() => collectItems(doc, layers), [doc, layers]);
  const [extW, extH] = useMemo(() => clickExtent(doc, items), [doc, items]);

  if (tool !== "select" && !roMode) return null;

  const k = Math.max(scale, 0.05);

  const toImage = (
    e: Konva.KonvaEventObject<any>,
  ): [number, number] | null => {
    const parent = e.target.getParent?.();
    const rp = parent?.getRelativePointerPosition?.();
    if (rp && Number.isFinite(rp.x)) return [rp.x, rp.y];
    const p = e.target.getStage()?.getPointerPosition();
    if (!p) return null;
    return [(p.x - offset.x) / k, (p.y - offset.y) / k];
  };

  const under = (pt: [number, number]) =>
    items
      .filter((it) => {
        const pad = it.pad / k;
        const [x0, y0, x1, y1] = it.bbox;
        return (
          pt[0] >= x0 - pad &&
          pt[0] <= x1 + pad &&
          pt[1] >= y0 - pad &&
          pt[1] <= y1 + pad
        );
      })
      .sort((a, b) => area(a.bbox) - area(b.bbox));

  /**
   * What a click at `pt` selects.
   *  - first click: the innermost region under the pointer
   *  - click again on the same spot (or on the selected innermost region):
   *    the next item in the stack — its lines/cells, then the region
   *    AROUND it, and so on, wrapping back to the innermost
   *  - Shift+click: jump straight to the next region outwards
   *  - Alt+click: the smallest thing under the pointer (line / cell / field)
   */
  function resolve(
    pt: [number, number],
    mods: { alt: boolean; shift: boolean },
  ): { target: HitItem | null; stack: HitItem[] } {
    const stack = buildStack(under(pt));
    if (stack.length === 0) return { target: null, stack };
    if (roMode) {
      return { target: stack.find((h) => h.selection.kind === "layout") ?? null, stack };
    }
    if (mods.alt) {
      return { target: [...stack].sort((a, b) => area(a.bbox) - area(b.bbox))[0], stack };
    }

    const cur = keyOf(useStore.getState().selection);
    const curIdx = stack.findIndex((s) => keyOf(s.selection) === cur);

    if (mods.shift) {
      const outer = stack.slice(curIdx + 1).find((s) => s.role === "box");
      return { target: outer ?? stack[0], stack };
    }

    const lc = lastClick.current;
    const sameSpot = !!lc && Math.hypot(pt[0] - lc[0], pt[1] - lc[1]) <= SAME_SPOT_PX / k;
    if (curIdx >= 0 && (sameSpot || curIdx === 0)) {
      return { target: stack[(curIdx + 1) % stack.length], stack };
    }
    return { target: stack[0], stack };
  }

  function updateHover(pt: [number, number], mods: { alt: boolean; shift: boolean }) {
    const { target, stack } = resolve(pt, mods);
    const next = target
      ? { key: target.key, pos: stack.indexOf(target) + 1, n: stack.length }
      : null;
    setHover((h) =>
      h?.key === next?.key && h?.pos === next?.pos && h?.n === next?.n ? h : next,
    );
    return target;
  }

  function apply(target: HitItem | null, shiftKey: boolean, pt: [number, number]) {
    lastClick.current = pt;
    pickMenu.close();
    if (!target) {
      if (!roMode) setSelection(null);
      return;
    }
    if (roMode) {
      if (target.selection.kind !== "layout") return;
      const id = target.selection.id;
      edit((d) =>
        shiftKey ? removeFromReadingOrder(d, id) : appendReadingOrder(d, id),
      );
      setSelection({ kind: "layout", id });
      return;
    }
    setSelection(target.selection);
  }

  // ---- pointer handlers ---------------------------------------------

  function onClick(e: Konva.KonvaEventObject<MouseEvent>) {
    if (e.evt.button !== 0 || spaceRef.current) return;
    e.cancelBubble = true;
    if (dragPreview) {
      setDragPreview(null);
      return;
    }
    const pt = toImage(e);
    if (!pt) return;
    const mods = { alt: e.evt.altKey, shift: e.evt.shiftKey };
    apply(resolve(pt, mods).target, e.evt.shiftKey, pt);
    updateHover(pt, mods); // show what the NEXT click will pick
  }

  function onContextMenu(e: Konva.KonvaEventObject<PointerEvent>) {
    e.evt.preventDefault();
    e.cancelBubble = true;
    const pt = toImage(e);
    if (pt) openMenuAt(pt, e.evt.clientX, e.evt.clientY);
  }

  function openMenuAt(pt: [number, number], clientX: number, clientY: number) {
    if (roMode) return;
    const stack = buildStack(under(pt));
    if (stack.length === 0) {
      pickMenu.close();
      return;
    }
    pickMenu.open(
      clientX,
      clientY,
      stack.map((it) => ({
        key: it.key,
        selection: it.selection,
        depth: it.role === "child" ? 1 : 0,
        ...describeSelection(doc, it.selection),
      })),
    );
  }

  function onDragStart(e: Konva.KonvaEventObject<DragEvent>) {
    e.cancelBubble = true;
    pickMenu.close();
    if (roMode || spaceRef.current) {
      e.target.stopDrag();
      return;
    }
    const pt = toImage(e);
    if (!pt) {
      e.target.stopDrag();
      return;
    }
    const boxes = under(pt).filter((h) => h.selection.kind === "layout");
    const active = parentLayoutOf(doc, useStore.getState().selection);
    const pick = boxes.find((b) => b.layoutId === active) ?? boxes[0];
    if (!pick || pick.layoutId == null) {
      // Nothing draggable here (orphan line, checkbox, fingerprint …).
      // Treat the jittery press as a click instead of swallowing it.
      e.target.stopDrag();
      e.target.position({ x: -M, y: -M });
      apply(resolve(pt, { alt: false, shift: false }).target, false, pt);
      return;
    }
    down.current = {
      sx: e.target.x(),
      sy: e.target.y(),
      dragId: pick.layoutId,
      bbox: pick.bbox,
    };
    setSelection({ kind: "layout", id: pick.layoutId });
  }

  function onDragMove(e: Konva.KonvaEventObject<DragEvent>) {
    e.cancelBubble = true;
    const dn = down.current;
    if (!dn) return;
    const dx = e.target.x() - dn.sx;
    const dy = e.target.y() - dn.sy;
    setDragPreview({ bbox: dn.bbox, dx, dy });
  }

  function onDragEnd(e: Konva.KonvaEventObject<DragEvent>) {
    e.cancelBubble = true;
    const dn = down.current;
    if (!dn) return;
    const dx = Math.round(e.target.x() - dn.sx);
    const dy = Math.round(e.target.y() - dn.sy);
    // put the catch-all rect back
    e.target.position({ x: -M, y: -M });
    down.current = null;
    lastClick.current = null;
    setDragPreview(null);
    if (dx || dy) {
      const id = dn.dragId;
      edit((d) => translateLayoutElement(d, id, dx, dy));
    }
  }

  function onMove(e: Konva.KonvaEventObject<MouseEvent>) {
    if (down.current) return;
    const pt = toImage(e);
    if (!pt) return;
    const t = updateHover(pt, { alt: e.evt.altKey, shift: e.evt.shiftKey });
    const st = e.target.getStage();
    if (st) {
      const locked =
        roMode &&
        t?.selection.kind === "layout" &&
        isReadingOrderLocked(
          doc.layout.find((el: any) => el.id === t.layoutId)?.type ?? "",
        );
      st.container().style.cursor = !t
        ? "default"
        : roMode
          ? locked
            ? "not-allowed"
            : "copy"
          : "pointer";
    }
  }

  // clicks that land on a resize handle (without dragging) come here
  hitBridge.click = (pt, mods) => {
    apply(resolve(pt, mods).target, mods.shift, pt);
    updateHover(pt, mods);
  };
  hitBridge.contextMenu = openMenuAt;

  const hovered = hover ? items.find((it) => it.key === hover.key) : undefined;
  const previewItem = menuPreview ? items.find((it) => it.key === menuPreview) : undefined;

  let hoverRing: JSX.Element | null = null;
  if (hovered && hover && !dragPreview && !previewItem) {
    const d = describeSelection(doc, hovered.selection);
    const color = roMode ? "#0284c7" : d.color;
    const label = roMode
      ? d.label
      : hover.n > 1
        ? `${d.label}   ${hover.pos}/${hover.n} · click again ↻ · right-click: list`
        : d.label;
    hoverRing = <Ring bbox={hovered.bbox} color={color} k={k} label={label} />;
  }

  return (
    <Group>
      {/* Catch-all. dragDistance=5 lets a click through even if the
          pointer jitters a few pixels. */}
      <Rect
        x={-M}
        y={-M}
        width={extW + 2 * M}
        height={extH + 2 * M}
        fill="rgba(0,0,0,0.01)"
        listening={true}
        draggable={!roMode}
        dragDistance={5}
        onClick={onClick}
        onContextMenu={onContextMenu}
        onDragStart={onDragStart}
        onDragMove={onDragMove}
        onDragEnd={onDragEnd}
        onMouseMove={onMove}
        onMouseLeave={(e) => {
          setHover(null);
          const st = e.target.getStage();
          if (st) st.container().style.cursor = "default";
        }}
        onTap={(e) => {
          e.cancelBubble = true;
          const pt = toImage(e as any);
          if (pt) apply(resolve(pt, { alt: false, shift: false }).target, false, pt);
        }}
      />

      {/* hover outline — shows exactly what a click will select */}
      {hoverRing}

      {/* entry hovered in the right-click list */}
      {previewItem && (
        <Ring
          bbox={previewItem.bbox}
          color={describeSelection(doc, previewItem.selection).color}
          k={k}
        />
      )}

      {/* live preview while dragging a region */}
      {dragPreview && (
        <Rect
          x={dragPreview.bbox[0] + dragPreview.dx}
          y={dragPreview.bbox[1] + dragPreview.dy}
          width={dragPreview.bbox[2] - dragPreview.bbox[0]}
          height={dragPreview.bbox[3] - dragPreview.bbox[1]}
          stroke="#f59e0b"
          strokeWidth={2.5}
          strokeScaleEnabled={false}
          dash={[6 / k, 4 / k]}
          fill="#f59e0b33"
          listening={false}
        />
      )}
    </Group>
  );
}
