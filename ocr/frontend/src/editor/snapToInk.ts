import { snapBox } from "../api";
import { useStore } from "../store";
import type { BBox, Doc, Quad } from "../types";
import { quadFromBBox } from "./commands";

/** A child box inside the page: where it lives and which region it belongs to. */
export interface ChildRef {
  /** the box object (line, field label / value) in the given doc, if it still exists */
  get: (d: Doc) => { bbox: BBox; quad?: Quad } | undefined;
  /** its parent region's box: the snapped result never leaves it */
  parent?: (d: Doc) => BBox | undefined;
}

const same = (a: BBox, b: BBox) => a.every((v, i) => Math.round(v) === Math.round(b[i]));

/**
 * After a child box was drawn or resized (already committed as drawn), ask
 * the server to tighten it to the text ink inside it and apply that as one
 * more edit. Tilted text comes back as a tight tilted quad, so a loose
 * straight box drawn around a slanted line becomes a slanted box. Skipped
 * when Alt is held, for quads that are already tilted (adjusted by hand),
 * and when the box was changed again before the answer came back.
 */
export async function snapChildToInk(ref: ChildRef, drawn: BBox, alt: boolean): Promise<void> {
  if (alt) return;
  const st = useStore.getState();
  const doc = st.doc;
  if (!doc) return;
  const obj = ref.get(doc);
  if (!obj || (obj.quad && !isRect(obj.quad))) return; // a tilted quad stays as drawn
  try {
    const r = await snapBox(doc.document_id, drawn, ref.parent?.(doc) ?? null);
    if (!r.changed) return;
    useStore.getState().edit((d) => {
      const o = ref.get(d);
      if (!o || !same(o.bbox, drawn)) return; // moved again meanwhile: keep the newer box
      o.bbox = r.bbox;
      // tilted text: a tight rotated box; straight text: the rectangle
      o.quad = r.quad && r.quad.length === 4 ? (r.quad as unknown as Quad) : quadFromBBox(r.bbox);
    });
  } catch {
    // snapping is a convenience: the box stays as drawn
  }
}

function isRect(q: Quad): boolean {
  const [tl, tr, br, bl] = q;
  return tl[1] === tr[1] && bl[1] === br[1] && tl[0] === bl[0] && tr[0] === br[0];
}

/** The parent region box of a line (by its block_id). */
export const lineParent = (index: number) => (d: Doc): BBox | undefined => {
  const ln = d.lines[index];
  return ln ? (d.layout.find((e) => e.id === ln.block_id)?.bbox as BBox | undefined) : undefined;
};
