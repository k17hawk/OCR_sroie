import { describe, expect, it } from "vitest";

import {
  clampOffset, fitView, KEEP_VISIBLE, MAX_SCALE, MIN_SCALE, pageSize, scrollBy,
  wheelAction, zoomAt, zoomTo, ZOOM_STEP, type View,
} from "./viewport";

// gov_005 scans: 2481 x 3508 px, shown in a 1200 x 800 canvas
const page = { w: 2481, h: 3508 };
const view = { w: 1200, h: 800 };
const wheel = (o: Partial<Parameters<typeof wheelAction>[0]>) =>
  wheelAction({ deltaX: 0, deltaY: 0, deltaMode: 0, ctrlKey: false, metaKey: false, shiftKey: false, ...o });
const toImage = (v: View, p: { x: number; y: number }) =>
  [(p.x - v.offset.x) / v.scale, (p.y - v.offset.y) / v.scale];

describe("fit (back to size)", () => {
  it("shows the whole page, centred, inside the margin", () => {
    const v = fitView(page, view)!;
    expect(v.scale).toBeCloseTo((800 - 80) / 3508, 6);
    expect(v.offset.y).toBeCloseTo(40, 6);
    expect(v.offset.x).toBeCloseTo((1200 - 2481 * v.scale) / 2, 6);
  });

  it("returns null before the canvas is measured", () => {
    expect(fitView(page, { w: 0, h: 0 })).toBeNull();
    expect(fitView({ w: 0, h: 0 }, view)).toBeNull();
  });

  it("after zooming and scrolling, fit gives the same view again", () => {
    const start = fitView(page, view)!;
    let v = zoomAt(start, 3, { x: 300, y: 200 }, page, view);
    v = scrollBy(v, 400, 900, page, view);
    expect(fitView(page, view)).toEqual(start);
  });
});

describe("zoom", () => {
  it("keeps the image point under the pointer still", () => {
    const v0 = fitView(page, view)!;
    const at = { x: 500, y: 300 };
    const v1 = zoomAt(v0, ZOOM_STEP, at, page, view);
    expect(v1.scale).toBeCloseTo(v0.scale * ZOOM_STEP, 9);
    const [ax, ay] = toImage(v0, at);
    const [bx, by] = toImage(v1, at);
    expect(bx).toBeCloseTo(ax, 6);
    expect(by).toBeCloseTo(ay, 6);
  });

  it("stays within the scale limits", () => {
    const v0 = fitView(page, view)!;
    expect(zoomAt(v0, 1e6, { x: 0, y: 0 }, page, view).scale).toBe(MAX_SCALE);
    expect(zoomAt(v0, 1e-6, { x: 0, y: 0 }, page, view).scale).toBe(MIN_SCALE);
  });

  it("100 % = one scan pixel per screen pixel", () => {
    expect(zoomTo(fitView(page, view)!, 1, { x: 600, y: 400 }, page, view).scale).toBe(1);
  });
});

describe("scroll", () => {
  it("moves the page opposite to the wheel delta", () => {
    const v0 = zoomTo(fitView(page, view)!, 1, { x: 600, y: 400 }, page, view);
    const v1 = scrollBy(v0, 0, 120, page, view);
    expect(v1.scale).toBe(v0.scale);
    expect(v1.offset.y).toBe(v0.offset.y - 120);
    expect(v1.offset.x).toBe(v0.offset.x);
  });

  it("never scrolls the page off screen", () => {
    const v0: View = { scale: 1, offset: { x: 0, y: 0 } };
    const down = scrollBy(v0, 0, 1e6, page, view);       // far down
    expect(down.offset.y + page.h).toBe(KEEP_VISIBLE);    // bottom edge still on screen
    const right = scrollBy(v0, -1e6, 0, page, view);     // far left
    expect(right.offset.x).toBe(view.w - KEEP_VISIBLE);
  });

  it("clampOffset leaves an in-range offset alone", () => {
    expect(clampOffset({ x: -100, y: -200 }, 1, page, view)).toEqual({ x: -100, y: -200 });
  });
});

describe("wheel", () => {
  it("plain wheel scrolls, Shift + wheel scrolls sideways", () => {
    expect(wheel({ deltaY: 100 })).toEqual({ kind: "scroll", dx: 0, dy: 100 });
    expect(wheel({ deltaY: 100, shiftKey: true })).toEqual({ kind: "scroll", dx: 100, dy: 0 });
    expect(wheel({ deltaX: 30, deltaY: 10 })).toEqual({ kind: "scroll", dx: 30, dy: 10 });
  });

  it("line-mode wheels (Firefox) scroll in pixels", () => {
    expect(wheel({ deltaY: 3, deltaMode: 1 })).toEqual({ kind: "scroll", dx: 0, dy: 48 });
  });

  it("Ctrl / ⌘ + wheel and pinch zoom; wheel up zooms in", () => {
    const notch = wheel({ deltaY: -100, ctrlKey: true });
    expect(notch).toEqual({ kind: "zoom", factor: ZOOM_STEP });
    expect(wheel({ deltaY: 100, metaKey: true })).toEqual({ kind: "zoom", factor: 1 / ZOOM_STEP });
    const pinch = wheel({ deltaY: -4, ctrlKey: true });     // trackpad: small, smooth steps
    expect(pinch.kind).toBe("zoom");
    if (pinch.kind === "zoom") expect(pinch.factor).toBeGreaterThan(1);
  });
});

describe("pageSize", () => {
  it("uses width/height, or the boxes when they are missing", () => {
    expect(pageSize({ width: 2481, height: 3508 })).toEqual(page);
    expect(pageSize({ width: 0, height: 0, layout: [{ bbox: [0, 0, 900, 1200] }],
                      lines: [{ bbox: [10, 10, 950, 40] }] })).toEqual({ w: 950, h: 1200 });
  });
});
