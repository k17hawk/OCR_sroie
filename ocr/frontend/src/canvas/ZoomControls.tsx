/**
 * Zoom bar in the bottom-right corner of the canvas:  −  [ 42% ]  +  Fit  100%
 *
 * "Fit" brings the page back to size (the whole page on screen, as when it
 * opens). Shortcuts: Ctrl/⌘ + "+" / "−", Ctrl/⌘ + 0 (fit), Ctrl/⌘ + 1 (100 %).
 * Wheel scrolls the page, Ctrl/⌘ + wheel or a trackpad pinch zooms.
 */
interface Props {
  scale: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  onActual: () => void;
}

const btn =
  "h-7 min-w-7 rounded px-2 text-xs font-medium text-slate-200 hover:bg-slate-700 " +
  "focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400";

export function ZoomControls({ scale, onZoomIn, onZoomOut, onFit, onActual }: Props) {
  const pct = Math.round(scale * 100);
  // keep keyboard focus off the buttons so Space / tool keys still reach the canvas
  const noFocus = (e: React.MouseEvent) => e.preventDefault();
  return (
    <div
      className="absolute bottom-3 right-3 z-10 flex items-center gap-0.5 rounded-md border border-slate-700 bg-slate-900/90 p-0.5 shadow-lg"
      onWheel={(e) => e.stopPropagation()}
    >
      <button type="button" className={btn} onMouseDown={noFocus} onClick={onZoomOut}
              title="Zoom out (Ctrl/⌘ −)" aria-label="Zoom out">−</button>
      <span className="w-12 select-none text-center text-xs tabular-nums text-slate-300"
            title="Ctrl/⌘ + wheel or pinch to zoom; wheel to scroll" aria-live="polite">
        {pct}%
      </span>
      <button type="button" className={btn} onMouseDown={noFocus} onClick={onZoomIn}
              title="Zoom in (Ctrl/⌘ +)" aria-label="Zoom in">+</button>
      <span className="mx-0.5 h-4 w-px bg-slate-700" aria-hidden />
      <button type="button" className={btn} onMouseDown={noFocus} onClick={onFit}
              title="Fit the whole page (Ctrl/⌘ 0)">Fit</button>
      <button type="button" className={btn} onMouseDown={noFocus} onClick={onActual}
              title="Actual size: 1 scan pixel = 1 screen pixel (Ctrl/⌘ 1)">100%</button>
    </div>
  );
}
