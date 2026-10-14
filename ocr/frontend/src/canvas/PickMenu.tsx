import { useEffect, useRef } from "react";
import { useStore } from "../store";
import { pickMenu, usePickMenu } from "./pickMenuStore";

/**
 * Right-click on the canvas → a list of EVERY box under the cursor
 * (innermost first). Hover an entry to outline it, click to select it.
 * This is the sure way to reach a box that is hidden under / inside others.
 */
export function PickMenu() {
  const { menu } = usePickMenu();
  const selection = useStore((s) => s.selection);
  const setSelection = useStore((s) => s.setSelection);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menu) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) pickMenu.close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation(); // don't also clear the selection
        pickMenu.close();
      }
    };
    const onWheel = () => pickMenu.close();
    window.addEventListener("mousedown", onDown, true);
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("wheel", onWheel, true);
    return () => {
      window.removeEventListener("mousedown", onDown, true);
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("wheel", onWheel, true);
    };
  }, [menu]);

  if (!menu) return null;

  const W = 280;
  const estH = Math.min(340, 34 + menu.entries.length * 30);
  const left = Math.max(8, Math.min(menu.x, window.innerWidth - W - 8));
  const top = Math.max(8, Math.min(menu.y, window.innerHeight - estH - 8));
  const curKey = selection ? JSON.stringify(selection) : "";

  return (
    <div
      ref={ref}
      style={{ position: "fixed", left, top, width: W, zIndex: 50 }}
      className="max-h-[340px] overflow-y-auto rounded-md border border-slate-600 bg-slate-900/95 p-1 text-xs shadow-2xl backdrop-blur"
      onContextMenu={(e) => e.preventDefault()}
    >
      <div className="px-2 pb-1 pt-0.5 text-[10px] uppercase tracking-wider text-slate-400">
        {menu.entries.length} under cursor · inner → outer
      </div>
      {menu.entries.map((en) => {
        const current = JSON.stringify(en.selection) === curKey;
        return (
          <button
            key={en.key}
            type="button"
            onMouseEnter={() => pickMenu.setPreview(en.key)}
            onMouseLeave={() => pickMenu.setPreview(null)}
            onClick={() => {
              setSelection(en.selection);
              pickMenu.close();
            }}
            className={
              "flex w-full items-center gap-2 rounded py-1 pr-2 text-left hover:bg-slate-700 " +
              (current ? "bg-slate-800" : "")
            }
            style={{ paddingLeft: 8 + en.depth * 14 }}
          >
            <span
              className="inline-block h-3 w-3 shrink-0 rounded-sm ring-1 ring-white/70"
              style={{ background: en.color }}
            />
            <span className="shrink-0 font-mono text-[11px] font-semibold text-slate-100">
              {en.label}
            </span>
            <span className="truncate font-deva text-[11px] text-slate-400">{en.sub}</span>
            {current && <span className="ml-auto text-[10px] text-sky-400">●</span>}
          </button>
        );
      })}
    </div>
  );
}
