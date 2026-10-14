/**
 * Lets the resize handles pass a plain click (no drag) on to HitLayer.
 * Handles sit above HitLayer so their drags win, but a click on a handle's
 * invisible grab circle used to be swallowed — so small or nested boxes
 * next to the selected box could not be selected. HitLayer registers its
 * click / right-click logic here; handles.tsx forwards to it.
 */
type Pt = [number, number];
type Mods = { alt: boolean; shift: boolean };

export const hitBridge: {
  click: ((pt: Pt, mods: Mods) => void) | null;
  contextMenu: ((pt: Pt, clientX: number, clientY: number) => void) | null;
} = { click: null, contextMenu: null };
