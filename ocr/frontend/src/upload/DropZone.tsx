import { useRef, useState } from "react";
import { uploadImages } from "../api";

interface Props {
  projectId: number;
  onUploaded: (ids: string[]) => void;
}

export function DropZone({ projectId, onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(
    null,
  );

  async function doUpload(files: FileList | File[]) {
    const list = Array.from(files);
    if (list.length === 0) return;

    setBusy(true);
    setErrors([]);
    setProgress({ done: 0, total: list.length });

    try {
      // The API accepts many files per request; the counter here is
      // per-batch (0 → 1) rather than per-file. If you want true per-file
      // progress, split the batch and call uploadImages once per file.
      const res = await uploadImages(projectId, list);

      setProgress({ done: list.length, total: list.length });

      const failures = res.results.filter((r) => !r.ok);
      const warnings = res.results.filter(
        (r) => r.ok && r.skipped_duplicate,
      );

      const messages: string[] = [];
      for (const f of failures) {
        messages.push(`✗ ${f.filename}: ${f.error ?? "failed"}`);
      }
      for (const w of warnings) {
        if (!w.warning) {
          messages.push(`↷ ${w.filename}: already uploaded — opened the saved annotations`);
        }
      }
      for (const r of res.results) {
        if (r.ok && r.warning) messages.push(`⚠ ${r.warning}`);
      }
      setErrors(messages);

      onUploaded(res.documents.map((d) => d.document_id));
    } catch (e) {
      setErrors([`✗ ${String(e)}`]);
    } finally {
      setBusy(false);
      // Keep the summary visible for a moment.
      setTimeout(() => setProgress(null), 800);
    }
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!busy) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (busy) return;
        if (e.dataTransfer.files.length) void doUpload(e.dataTransfer.files);
      }}
      className={
        "rounded border-2 border-dashed p-3 text-center text-xs transition-colors " +
        (dragging
          ? "border-sky-500 bg-sky-500/10 text-sky-200"
          : "border-slate-700 text-slate-400")
      }
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,.webp,.pdf,.zip"
        className="hidden"
        onChange={(e) => {
          if (e.target.files) void doUpload(e.target.files);
          e.target.value = ""; // allow re-uploading the same file
        }}
      />

      <div className="mb-2">Drop images, PDFs, or a ZIP here</div>

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={busy}
        className={
          "rounded px-2 py-1 text-[11px] " +
          (busy
            ? "bg-slate-800 text-slate-500"
            : "bg-slate-800 text-slate-200 hover:bg-slate-700")
        }
      >
        {busy ? "Uploading…" : "Choose files"}
      </button>

      {progress && (
        <div className="mt-2 text-[10px] text-slate-500">
          {progress.done} / {progress.total}
        </div>
      )}

      {errors.length > 0 && (
        <ul className="mt-2 space-y-0.5 text-left text-[10px]">
          {errors.map((e, i) => (
            <li
              key={i}
              className={
                e.startsWith("✗")
                  ? "text-rose-400"
                  : e.startsWith("↷")
                    ? "text-amber-400"
                    : "text-slate-400"
              }
            >
              {e}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}