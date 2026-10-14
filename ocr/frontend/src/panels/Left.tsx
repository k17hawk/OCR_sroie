import { useStore } from "../store";
import { LAYER_KEYS, LAYER_LABELS } from "../lib";
import { DropZone } from "../upload/DropZone";
import { PageQueue } from "./PageQueue";

interface Props {
  projectId?: number;
}

export function LeftPanel({ projectId = 1 }: Props) {
  const layers = useStore((s) => s.layers);
  const toggleLayer = useStore((s) => s.toggleLayer);
  const selection = useStore((s) => s.selection);

  return (
    <aside className="flex h-full w-72 flex-col border-r border-slate-800 bg-slate-900">
      <div className="border-b border-slate-800 px-4 py-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          DevaLabel
        </div>
        <div className="text-sm text-slate-200">Phase 4 · annotator</div>
      </div>

      <div className="space-y-3 border-b border-slate-800 p-3">
        <DropZone
          projectId={projectId}
          onUploaded={(ids) => {
            if (ids[0]) window.location.hash = `#/${ids[0]}`;
          }}
        />

        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Page queue
          </div>
          <PageQueue projectId={projectId} />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
          Layers
        </div>
        <ul className="space-y-1">
          {LAYER_KEYS.map((k) => (
            <li key={k}>
              <label className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-xs hover:bg-slate-800">
                <input
                  type="checkbox"
                  checked={layers[k]}
                  onChange={() => toggleLayer(k)}
                  className="h-3.5 w-3.5 accent-sky-500"
                />
                <span className="text-slate-300">{LAYER_LABELS[k]}</span>
                {k === "blocks" && (
                  <span className="ml-auto text-[10px] text-slate-500">
                    derived
                  </span>
                )}
              </label>
            </li>
          ))}
        </ul>

        <div className="mt-4 mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
          Selection
        </div>
        <div className="rounded bg-slate-800 p-2 font-mono text-[11px] text-slate-300">
          {selection ? JSON.stringify(selection) : "—"}
        </div>
      </div>

      <div className="border-t border-slate-800 p-3 text-[10px] leading-relaxed text-slate-500">
        <div>
          <kbd className="rounded bg-slate-800 px-1">Space</kbd> + drag to pan ·{" "}
          <kbd className="rounded bg-slate-800 px-1">wheel</kbd> to zoom ·{" "}
        
        </div>
          <div>
    <kbd className="rounded bg-slate-800 px-1">N</kbd> /{" "}
    <kbd className="rounded bg-slate-800 px-1">P</kbd> next / prev page ·{" "}
    <kbd className="rounded bg-slate-800 px-1">←</kbd>{" "}
    <kbd className="rounded bg-slate-800 px-1">→</kbd>

    <kbd className="rounded bg-slate-800 px-1">Tab</kbd> cycle ·{" "}
    <kbd className="rounded bg-slate-800 px-1">↑↓←→</kbd> nudge ·{" "}
    <kbd className="rounded bg-slate-800 px-1">Shift</kbd>+nudge = 10px ·{" "}
    <kbd className="rounded bg-slate-800 px-1">Del</kbd> remove
  </div>
  </div>
    </aside>
  );
}