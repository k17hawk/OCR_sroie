import { useEffect, useMemo, useState } from "react";
import {
  Rect, Text, Line, Arrow, Circle, Group, Image as KImage,
} from "react-konva";

import type { Doc, LayoutElement, BBox } from "../types";
import { quadPoints } from "../lib";
import {
  typeColor, prettyType, TEXT_LIKE_TYPES, KIND_COLORS,
} from "../hooks/useProjectTypes";
import { useStore } from "../store";
import { TextInBox } from "./TextInBox";
import { parentLayoutOf } from "../editor/commands";
import { bboxArea, nestingMap } from "../editor/nesting";

/*
 * Everything here is decorative — HitLayer does all hit-testing — so every
 * node has listening={false}.
 *
 * Coordinates are image pixels, but strokes use strokeScaleEnabled={false}
 * so a "2" is 2 SCREEN pixels at any zoom (before, a 1.5 px image-space
 * stroke shrank to ~0.4 px when the page was zoomed out, which is why the
 * boxes looked pale). Chips, dots and labels are divided by the zoom `k`
 * for the same reason. A white halo under every outline keeps it visible
 * on dark scans too.
 */

const HALO = "rgba(255,255,255,0.95)";
const UI_FONT = "Inter, system-ui, sans-serif";

function useK(): number {
  const s = useStore((st) => st.scale);
  return Math.max(s, 0.05);
}

function Outline({
  bbox, color, width, dash, glow,
}: { bbox: BBox; color: string; width: number; dash?: number[]; glow?: boolean }) {
  const [x0, y0, x1, y1] = bbox;
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  return (
    <>
      <Rect
        x={x0} y={y0} width={w} height={h}
        stroke={HALO} strokeWidth={width + 2.5} strokeScaleEnabled={false}
        listening={false} perfectDrawEnabled={false}
      />
      <Rect
        x={x0} y={y0} width={w} height={h}
        stroke={color} strokeWidth={width} strokeScaleEnabled={false}
        dash={dash}
        shadowColor={glow ? color : undefined}
        shadowBlur={glow ? 12 : 0}
        shadowOpacity={glow ? 1 : 0}
        shadowForStrokeEnabled={!!glow}
        listening={false} perfectDrawEnabled={false}
      />
    </>
  );
}

/** Solid colored tag with white bold text, constant size on screen. */
function Chip({ x, y, k, text, color }: { x: number; y: number; k: number; text: string; color: string }) {
  const h = 16;
  const w = 10 + text.length * 6.4;
  return (
    <Group x={x} y={y} scaleX={1 / k} scaleY={1 / k} listening={false}>
      <Rect width={w} height={h} fill={color} cornerRadius={3} stroke={HALO} strokeWidth={1} />
      <Text
        x={5} y={2.5} text={text}
        fontSize={11} fontStyle="bold" fontFamily={UI_FONT}
        fill="#ffffff"
      />
    </Group>
  );
}

/** White backdrop + dark text inside a box (readable over any scan). */
function BoxText({
  bbox, text, k, maxFont, italic,
}: { bbox: BBox; text: string; k: number; maxFont: number; italic?: boolean }) {
  const [x0, y0, x1, y1] = bbox;
  if (!text || (y1 - y0) * k < 9) return null; // too small to read at this zoom
  return (
    <>
      <Rect
        x={x0} y={y0}
        width={Math.max(1, x1 - x0)} height={Math.max(1, y1 - y0)}
        fill="rgba(255,255,255,0.72)" listening={false}
      />
      <TextInBox
        bbox={bbox}
        text={text}
        color="#0f172a"
        italic={italic}
        padding={2}
        minFontSize={6}
        maxFontSize={maxFont}
        verticalAlign="middle"
      />
    </>
  );
}

// ===========================================================================
// Image
// ===========================================================================

export function ImageLayer({ doc }: { doc: Doc }) {
  const imageUrl = useStore((s) => s.imageUrl);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [failed, setFailed] = useState(false);
  const [loading, setLoading] = useState(true);

  const src = imageUrl ?? `/images/${doc.image.split("/").pop()}`;

  useEffect(() => {
    setImg(null);
    setFailed(false);
    setLoading(true);
    const el = new window.Image();
    el.crossOrigin = "anonymous";
    el.onload = () => { setImg(el); setLoading(false); };
    el.onerror = () => { setFailed(true); setLoading(false); };
    el.src = src;
    return () => { el.onload = null; el.onerror = null; };
  }, [src]);

  return (
    <>
      <Rect x={0} y={0} width={doc.width} height={doc.height} fill="#ffffff" listening={false} />
      {img && (
        <KImage image={img} x={0} y={0} width={doc.width} height={doc.height} listening={false} />
      )}
      {loading && (
        <Text
          x={0} y={doc.height / 2 - 12} width={doc.width} align="center"
          text="Loading image…" fontSize={16}
          fontFamily="Noto Sans Devanagari" fill="#64748b"
          listening={false}
        />
      )}
      {failed && (
        <Text
          x={0} y={doc.height / 2 - 24} width={doc.width} align="center"
          text={`Image not available\n${src}`} fontSize={14}
          fontFamily="Noto Sans Devanagari" fill="#b91c1c"
          listening={false}
        />
      )}
    </>
  );
}

// ===========================================================================
// Layout regions
// ===========================================================================

function isAxisAligned(el: LayoutElement): boolean {
  const [x0, y0, x1, y1] = el.bbox;
  const q = el.quad;
  if (!q || q.length < 4) return true;
  return (
    Math.abs(q[0][0] - x0) <= 1 && Math.abs(q[0][1] - y0) <= 1 &&
    Math.abs(q[2][0] - x1) <= 1 && Math.abs(q[2][1] - y1) <= 1
  );
}

export function LayoutLayer({ doc }: { doc: Doc }) {
  const selection = useStore((s) => s.selection);
  const k = useK();

  const nest = useMemo(() => nestingMap(doc.layout), [doc.layout]);
  // Big (outer) regions first, so nested ones are drawn on top of them.
  const ordered = useMemo(
    () => [...doc.layout].sort((a, b) => bboxArea(b.bbox) - bboxArea(a.bbox)),
    [doc.layout],
  );
  const withLines = useMemo(
    () => new Set(doc.lines.map((l: any) => l.block_id)),
    [doc.lines],
  );

  const selId = selection?.kind === "layout" ? selection.id : null;
  const ctxId = selection && selection.kind !== "layout" ? parentLayoutOf(doc, selection) : null;
  const chipH = 16 / k;

  const view = ordered.map((el) => {
    const info = nest.get(el.id);
    const nested = info?.parent != null;
    const [, y0] = el.bbox;
    // top-level: chip sits just above the box; nested: inside its top-left
    // corner, so it doesn't land on top of the parent's chip
    const chipY = !nested && y0 - chipH >= 0 ? y0 - chipH : y0;
    return {
      el,
      color: typeColor(el.type),
      isSel: el.id === selId,
      isCtx: el.id === ctxId,
      parent: info?.parent ?? null,
      chipY,
      chipInside: chipY === y0,
    };
  });

  return (
    <>
      {/* boxes */}
      {view.map(({ el, color, isSel, isCtx, chipInside }) => {
        const [x0, y0, x1, y1] = el.bbox;
        const w = Math.max(2, x1 - x0);
        const h = Math.max(2, y1 - y0);
        const inside = withLines.has(el.id) ? null : textForLayoutElement(el);
        const textTop = y0 + (chipInside ? chipH : 0) + 2;

        return (
          <Group key={`layout-${el.id}`} listening={false}>
            <Rect
              x={x0} y={y0} width={w} height={h}
              fill={`${color}${isSel ? "40" : isCtx ? "30" : "22"}`}
              listening={false}
            />
            {inside && textTop < y1 - 4 && (
              <Group clipX={x0} clipY={y0} clipWidth={w} clipHeight={h} listening={false}>
                <BoxText
                  bbox={[x0 + 2, textTop, x1 - 2, y1 - 2]}
                  text={inside.text}
                  italic={inside.italic}
                  k={k}
                  maxFont={28}
                />
              </Group>
            )}
            <Outline
              bbox={el.bbox}
              color={color}
              width={isSel ? 3.5 : isCtx ? 2.5 : 2}
              dash={el.reading_order == null ? [7 / k, 4 / k] : undefined}
              glow={isSel}
            />
            {!isAxisAligned(el) && (
              <Line
                points={quadPoints(el.quad)}
                closed
                stroke={color}
                strokeWidth={1.5}
                strokeScaleEnabled={false}
                dash={[4 / k, 3 / k]}
                listening={false}
              />
            )}
          </Group>
        );
      })}

      {/* chips last, so no box ever covers a label */}
      {view.map(({ el, color, parent, chipY }) => {
        const text =
          `${el.id} ${prettyType(el.type)}` +
          (el.reading_order != null ? ` #${el.reading_order}` : "") +
          (parent != null ? ` ⊂${parent}` : "");
        return (
          <Group key={`chip-${el.id}`} listening={false}>
            <Chip x={el.bbox[0]} y={chipY} k={k} text={text} color={color} />
          </Group>
        );
      })}
    </>
  );
}

function textForLayoutElement(
  el: LayoutElement,
): { text: string; italic: boolean } | null {
  if (el.type === "table" || el.type === "form_field" || el.type === "logo") {
    return null;
  }
  if (TEXT_LIKE_TYPES.has(el.type)) {
    const own = "text" in el ? ((el as any).text as string | undefined) : undefined;
    return own && own.trim().length > 0 ? { text: own, italic: false } : null;
  }
  if (el.type === "figure") {
    const sub = (el as any).subtype as string | undefined;
    return sub ? { text: sub, italic: true } : null;
  }
  if (el.type === "signature") {
    const signer = (el as any).signer as string | undefined;
    return signer ? { text: signer, italic: true } : null;
  }
  if (el.type === "stamp") {
    const st = (el as any).stamp_text as string | undefined;
    return st && st.trim().length > 0 ? { text: st, italic: true } : null;
  }
  if (el.type === "list") {
    const items = (el as any).items as { marker: string; text: string }[] | undefined;
    if (!items || items.length === 0) return null;
    return { text: items.map((it) => `${it.marker} ${it.text}`).join("\n"), italic: false };
  }
  return null;
}

// ===========================================================================
// Blocks — decorative only
// ===========================================================================

export function BlockLayer({ doc }: { doc: Doc }) {
  const k = useK();
  return (
    <>
      {doc.blocks.map((b: any) => {
        const [x0, y0, x1, y1] = b.bbox as BBox;
        return (
          <Rect
            key={`block-${b.block_id}`}
            x={x0} y={y0} width={x1 - x0} height={y1 - y0}
            stroke="#475569" strokeWidth={1.25} strokeScaleEnabled={false}
            dash={[4 / k, 4 / k]}
            listening={false}
          />
        );
      })}
    </>
  );
}

// ===========================================================================
// Lines
// ===========================================================================

export function LineLayer({ doc }: { doc: Doc }) {
  const selection = useStore((s) => s.selection);
  const k = useK();

  return (
    <>
      {doc.lines.map((ln: any, i: number) => {
        const [x0, y0, x1, y1] = ln.bbox as BBox;
        const isSel = selection?.kind === "line" && selection.index === i;
        const color = ln.is_heading ? KIND_COLORS.lineHeading : KIND_COLORS.line;
        return (
          <Group key={`line-${i}`} listening={false}>
            <Rect
              x={x0} y={y0}
              width={Math.max(2, x1 - x0)} height={Math.max(2, y1 - y0)}
              fill={`${color}${isSel ? "33" : "14"}`}
              listening={false}
            />
            <BoxText bbox={ln.bbox} text={ln.text} k={k} maxFont={Math.max(10, y1 - y0)} />
            {isSel ? (
              <Outline bbox={ln.bbox} color={color} width={3} glow />
            ) : (
              <Rect
                x={x0} y={y0}
                width={Math.max(2, x1 - x0)} height={Math.max(2, y1 - y0)}
                stroke={color} strokeWidth={1.5} strokeScaleEnabled={false}
                listening={false}
              />
            )}
          </Group>
        );
      })}
    </>
  );
}

// ===========================================================================
// Tables
// ===========================================================================

export function TableLayer({ doc }: { doc: Doc }) {
  const selection = useStore((s) => s.selection);
  const k = useK();

  return (
    <>
      {doc.tables.map((t: any, ti: number) => (
        <Group key={`table-${ti}`} listening={false}>
          {t.cells.map((c: any) => {
            const [x0, y0, x1, y1] = c.bbox as BBox;
            const isSel =
              selection?.kind === "cell" &&
              selection.tableIndex === ti &&
              selection.cellId === c.id;
            const color = c.is_header ? KIND_COLORS.cellHeader : KIND_COLORS.cell;
            return (
              <Group key={`cell-${ti}-${c.id}`} listening={false}>
                <Rect
                  x={x0} y={y0}
                  width={Math.max(2, x1 - x0)} height={Math.max(2, y1 - y0)}
                  fill={`${color}${isSel ? "38" : c.empty ? "0d" : "1c"}`}
                  listening={false}
                />
                <BoxText bbox={c.bbox} text={c.text} k={k} maxFont={Math.max(10, (y1 - y0) * 0.8)} />
                {isSel ? (
                  <Outline bbox={c.bbox} color={color} width={3} glow />
                ) : (
                  <Rect
                    x={x0} y={y0}
                    width={Math.max(2, x1 - x0)} height={Math.max(2, y1 - y0)}
                    stroke={color} strokeWidth={1.5} strokeScaleEnabled={false}
                    listening={false}
                  />
                )}
              </Group>
            );
          })}
          {doc.ruling_lines
            .filter((rl: any) => rl.table_index === ti)
            .map((rl: any, i: number) => (
              <Line
                key={`rl-${ti}-${i}`}
                points={[rl.p0[0], rl.p0[1], rl.p1[0], rl.p1[1]]}
                stroke="#b45309"
                strokeWidth={Math.max(1.5, rl.width ?? 1)}
                strokeScaleEnabled={false}
                opacity={0.8}
                listening={false}
              />
            ))}
        </Group>
      ))}
    </>
  );
}

// ===========================================================================
// Form fields
// ===========================================================================

export function FormFieldLayer({ doc }: { doc: Doc }) {
  const selection = useStore((s) => s.selection);
  const k = useK();

  const fields = useMemo(() => {
    const out: {
      elementId: number;
      fieldIndex: number;
      labelText: string;
      labelBox: BBox;
      valueText: string | null;
      valueBox: BBox | null;
      options: { text: string; box: BBox; checked: boolean }[];
    }[] = [];

    for (const el of doc.layout) {
      if (el.type !== "form_field") continue;
      el.fields.forEach((f: any, fi: number) => {
        let valueBox: BBox | null = null;
        let valueText: string | null = null;
        const options: { text: string; box: BBox; checked: boolean }[] = [];
        if (f.value && "bbox" in f.value) {
          valueBox = f.value.bbox;
          valueText = f.value.text;
        } else if (f.value && "options" in f.value) {
          valueText = f.value.text;
          for (const opt of f.value.options) {
            const xs = opt.box_quad.map((p: number[]) => p[0]);
            const ys = opt.box_quad.map((p: number[]) => p[1]);
            options.push({
              text: opt.text,
              box: [
                Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)),
                Math.ceil(Math.max(...xs)), Math.ceil(Math.max(...ys)),
              ],
              checked: opt.checked,
            });
          }
        }
        out.push({
          elementId: el.id, fieldIndex: fi,
          labelText: f.label.text, labelBox: f.label.bbox,
          valueBox, valueText, options,
        });
      });
    }
    return out;
  }, [doc]);

  return (
    <>
      {fields.map((f, i) => {
        const isSel =
          selection?.kind === "field" &&
          selection.elementId === f.elementId &&
          selection.fieldIndex === f.fieldIndex;
        const [lx0, ly0, lx1, ly1] = f.labelBox;
        const lw = Math.max(2, lx1 - lx0), lh = Math.max(2, ly1 - ly0);

        return (
          <Group key={`field-${i}`} listening={false}>
            <Rect
              x={lx0} y={ly0} width={lw} height={lh}
              fill={`${KIND_COLORS.fieldLabel}${isSel ? "38" : "1c"}`}
              listening={false}
            />
            <BoxText bbox={f.labelBox} text={f.labelText} k={k} maxFont={Math.max(10, lh * 0.85)} />
            <Outline bbox={f.labelBox} color={KIND_COLORS.fieldLabel} width={isSel ? 3 : 1.75} glow={isSel} />

            {f.valueBox && (() => {
              const [vx0, vy0, vx1, vy1] = f.valueBox;
              const vw = Math.max(2, vx1 - vx0), vh = Math.max(2, vy1 - vy0);
              return (
                <>
                  <Rect
                    x={vx0} y={vy0} width={vw} height={vh}
                    fill={`${KIND_COLORS.fieldValue}${isSel ? "38" : "1c"}`}
                    listening={false}
                  />
                  <BoxText bbox={f.valueBox} text={f.valueText ?? ""} k={k} maxFont={Math.max(10, vh * 0.85)} />
                  <Outline bbox={f.valueBox} color={KIND_COLORS.fieldValue} width={isSel ? 3 : 1.75} glow={isSel} />
                  <Arrow
                    points={[lx0 + lw, ly0 + lh / 2, vx0, vy0 + vh / 2]}
                    stroke={KIND_COLORS.fieldLabel}
                    fill={KIND_COLORS.fieldLabel}
                    strokeWidth={1.75}
                    strokeScaleEnabled={false}
                    pointerLength={7 / k}
                    pointerWidth={7 / k}
                    listening={false}
                  />
                </>
              );
            })()}

            {f.options.map((opt, oi) => {
              const [ox0, oy0, ox1, oy1] = opt.box;
              const color = opt.checked ? KIND_COLORS.checked : KIND_COLORS.unchecked;
              return (
                <Group key={`opt-${i}-${oi}`} listening={false}>
                  <Rect
                    x={ox0} y={oy0}
                    width={Math.max(2, ox1 - ox0)} height={Math.max(2, oy1 - oy0)}
                    fill={opt.checked ? `${color}55` : undefined}
                    listening={false}
                  />
                  <Outline bbox={opt.box} color={color} width={2} />
                  <Text
                    x={ox1 + 3 / k}
                    y={oy0}
                    text={opt.text}
                    fontSize={11 / k}
                    fontStyle="bold"
                    fontFamily="Noto Sans Devanagari"
                    fill={color}
                    stroke={HALO}
                    strokeWidth={3}
                    strokeScaleEnabled={false}
                    fillAfterStrokeEnabled
                    listening={false}
                  />
                </Group>
              );
            })}
          </Group>
        );
      })}
    </>
  );
}

// ===========================================================================
// Extra regions
// ===========================================================================

export function ExtraRegionLayer({
  doc, kind,
}: { doc: Doc; kind: "checkbox" | "fingerprint" }) {
  const selection = useStore((s) => s.selection);
  const k = useK();

  return (
    <>
      {doc.extra_regions.map((er: any, i: number) => {
        if (er.type !== kind) return null;
        const [x0, y0, x1, y1] = er.bbox as BBox;
        const isSel = selection?.kind === "extraRegion" && selection.index === i;
        const isChecked = er.type === "checkbox" ? er.checked : false;
        const color = kind === "checkbox"
          ? (isChecked ? KIND_COLORS.checked : KIND_COLORS.unchecked)
          : KIND_COLORS.fingerprint;

        return (
          <Group key={`extra-${kind}-${i}`} listening={false}>
            <Rect
              x={x0} y={y0}
              width={Math.max(2, x1 - x0)} height={Math.max(2, y1 - y0)}
              fill={isChecked ? `${color}55` : `${color}1a`}
              listening={false}
            />
            <Outline bbox={er.bbox} color={color} width={isSel ? 3 : 2} glow={isSel} />
            {kind === "checkbox" && er.option && (
              <Text
                x={x0}
                y={y0 - 14 / k}
                text={er.option}
                fontSize={11 / k}
                fontStyle="bold"
                fontFamily="Noto Sans Devanagari"
                fill={color}
                stroke={HALO}
                strokeWidth={3}
                strokeScaleEnabled={false}
                fillAfterStrokeEnabled
                listening={false}
              />
            )}
            {kind === "fingerprint" && (
              <Text
                x={x0 + 3 / k}
                y={y0 + 3 / k}
                text={String(er.hand ?? "?")[0].toUpperCase()}
                fontSize={13 / k}
                fontStyle="bold"
                fontFamily={UI_FONT}
                fill={color}
                listening={false}
              />
            )}
          </Group>
        );
      })}
    </>
  );
}

// ===========================================================================
// Reading order
// ===========================================================================

export function ReadingOrderLayer({ doc }: { doc: Doc }) {
  const k = useK();
  const ordered = useMemo(
    () =>
      [...doc.layout]
        .filter((el) => el.reading_order != null)
        .sort((a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0)),
    [doc.layout],
  );
  const r = 11 / k;
  const center = (el: LayoutElement): [number, number] => [
    (el.bbox[0] + el.bbox[2]) / 2,
    (el.bbox[1] + el.bbox[3]) / 2,
  ];

  return (
    <>
      {/* arrows first, so the numbered dots sit on top */}
      {ordered.slice(0, -1).map((el, i) => {
        const [ax, ay] = center(el);
        const [bx, by] = center(ordered[i + 1]);
        const len = Math.hypot(bx - ax, by - ay);
        if (len < 2 * r + 1) return null;
        const ux = (bx - ax) / len, uy = (by - ay) / len;
        return (
          <Arrow
            key={`ro-arrow-${el.id}`}
            points={[ax + ux * r, ay + uy * r, bx - ux * r, by - uy * r]}
            stroke="#0369a1" fill="#0369a1"
            strokeWidth={2} strokeScaleEnabled={false}
            dash={[5 / k, 4 / k]}
            pointerLength={8 / k} pointerWidth={8 / k}
            listening={false}
          />
        );
      })}
      {ordered.map((el) => {
        const [cx, cy] = center(el);
        return (
          <Group key={`ro-${el.id}`} listening={false}>
            <Circle
              x={cx} y={cy} radius={r}
              fill="#0284c7" stroke={HALO} strokeWidth={2} strokeScaleEnabled={false}
              listening={false}
            />
            <Text
              x={cx - r} y={cy - 6.5 / k} width={2 * r} align="center"
              text={String(el.reading_order)}
              fontSize={12 / k} fontStyle="bold" fontFamily={UI_FONT}
              fill="#ffffff"
              listening={false}
            />
          </Group>
        );
      })}
    </>
  );
}
