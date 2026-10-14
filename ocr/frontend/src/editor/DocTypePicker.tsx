import { useStore } from "../store";
import { useProjectTypes, withCurrent, prettyType } from "../hooks/useProjectTypes";

export function DocTypePicker() {
  const doc = useStore((s) => s.doc);
  const project = useStore((s) => s.project);
  const edit = useStore((s) => s.edit);
  const types = useProjectTypes();

  if (!doc) return null;

  const options = withCurrent(
    types.error ? ((project as any)?.doc_types ?? types.doc_types) : types.doc_types,
    doc.doc_type,
  );
  const invalid = doc.doc_type === "other";

  return (
    <select
      value={doc.doc_type}
      onChange={(e) => edit((d) => { d.doc_type = e.target.value; })}
      className={
        "w-full rounded bg-slate-800 px-2 py-1 text-[12px] text-slate-100 " +
        (invalid ? "ring-1 ring-rose-500" : "")
      }
      title="Document type"
    >
      <option value="other">— choose document type —</option>
      {options.filter((t) => t !== "other").map((t) => (
        <option key={t} value={t}>{prettyType(t)}</option>
      ))}
      <option value="other">other</option>
    </select>
  );
}