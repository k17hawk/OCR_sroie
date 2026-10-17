/**
 * Viewport math for the page canvas: zoom around a point, scroll, keep the
 * page on screen, fit. Pure functions (no React / Konva / store), so they can
 * be unit tested.
 *
 * A viewport is `scale` (screen px per image px) and `offset` (screen
 * position of the image's top-left corner): screen = image * scale + offset.
 */

export interface Pt { x: number; y: number }
export interface Size { w: number; h: number }
export interface View { scale: number; offset: Pt }

export const MIN_SCALE = 0.05;
export const MAX_SCALE = 8;
/** one zoom step (buttons, keys, one wheel notch with Ctrl) */
export const ZOOM_STEP = 1.2;
/** at least this many screen px of the page stay visible when scrolling */
export const KEEP_VISIBLE = 80;
/** margin around the page when fitting */
export const FIT_PAD = 40;

export const clampScale = (s: number) => Math.max(MIN_SCALE, Math.min(MAX_SCALE, s));

/** The view that shows the whole page, centred. */
export function fitView(page: Size, view: Size, pad = FIT_PAD): View | null {
  if (!(page.w > 0 && page.h > 0 && view.w > 0 && view.h > 0)) return null;
  const fit = Math.min((view.w - pad * 2) / page.w, (view.h - pad * 2) / page.h);
  if (!Number.isFinite(fit) || fit <= 0) return null;
  const scale = clampScale(fit);
  return { scale, offset: { x: (view.w - page.w * scale) / 2, y: (view.h - page.h * scale) / 2 } };
}

/**
 * Keep part of the page on screen: the page may be scrolled until only
 * `keep` px of it remain visible on any side, never further. A page smaller
 * than the screen can still be moved around inside it.
 */
export function clampOffset(offset: Pt, scale: number, page: Size, view: Size,
                            keep = KEEP_VISIBLE): Pt {
  const clampAxis = (o: number, pageLen: number, viewLen: number) => {
    const len = pageLen * scale;
    const k = Math.min(keep, len, viewLen);
    const lo = k - len;          // page's right / bottom edge at `k`
    const hi = viewLen - k;      // page's left / top edge at viewLen - k
    return Math.max(lo, Math.min(hi, o));
  };
  return { x: clampAxis(offset.x, page.w, view.w), y: clampAxis(offset.y, page.h, view.h) };
}

/** Zoom by `factor` keeping the image point under `at` (screen px) still. */
export function zoomAt(v: View, factor: number, at: Pt, page: Size, view: Size): View {
  const scale = clampScale(v.scale * factor);
  const ix = (at.x - v.offset.x) / v.scale;
  const iy = (at.y - v.offset.y) / v.scale;
  const offset = { x: at.x - ix * scale, y: at.y - iy * scale };
  return { scale, offset: clampOffset(offset, scale, page, view) };
}

/** Zoom to an exact scale (e.g. 1 = 100 %) around `at`. */
export function zoomTo(v: View, scale: number, at: Pt, page: Size, view: Size): View {
  return zoomAt(v, clampScale(scale) / v.scale, at, page, view);
}

/** Scroll by a screen delta (wheel / trackpad): the page moves opposite to it. */
export function scrollBy(v: View, dx: number, dy: number, page: Size, view: Size): View {
  return { scale: v.scale, offset: clampOffset({ x: v.offset.x - dx, y: v.offset.y - dy }, v.scale, page, view) };
}

/**
 * What a wheel event does. Ctrl / ⌘ + wheel — and a trackpad pinch, which
 * browsers report as a wheel event with ctrlKey — zooms; Shift + wheel
 * scrolls sideways; a plain wheel / two-finger swipe scrolls.
 */
export function wheelAction(e: { deltaX: number; deltaY: number; deltaMode: number;
                                 ctrlKey: boolean; metaKey: boolean; shiftKey: boolean },
): { kind: "zoom"; factor: number } | { kind: "scroll"; dx: number; dy: number } {
  // deltaMode 1 = lines, 2 = pages (Firefox mouse wheels): convert to px
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
  const dx = e.deltaX * unit;
  const dy = e.deltaY * unit;
  if (e.ctrlKey || e.metaKey) {
    // smooth for trackpads (small deltas), one step per mouse notch
    const f = Math.exp(-dy * 0.0025);
    const factor = Math.abs(dy) >= 50 ? (dy > 0 ? 1 / ZOOM_STEP : ZOOM_STEP) : f;
    return { kind: "zoom", factor };
  }
  if (e.shiftKey && dx === 0) return { kind: "scroll", dx: dy, dy: 0 };
  return { kind: "scroll", dx, dy };
}

/** The page size to fit: its width/height, or the boxes when those are missing. */
export function pageSize(doc: { width?: number; height?: number;
                                layout?: { bbox: number[] }[]; lines?: { bbox: number[] }[] }): Size {
  let w = Number(doc.width) || 0;
  let h = Number(doc.height) || 0;
  if (!w || !h) {
    for (const el of [...(doc.layout ?? []), ...(doc.lines ?? [])]) {
      w = Math.max(w, el.bbox[2]);
      h = Math.max(h, el.bbox[3]);
    }
  }
  return { w, h };
}
