import { useQuery } from "@tanstack/react-query";
import { listDocuments } from "../api";
import { useStore } from "../store";

type Page = {
  document_id: string;
  page_number: number;
  page_count: number;
  status: "generated" | "in_progress" | "submitted" | "approved" | string;
  source_file_id?: string | null;
};

export function PageQueue({ projectId }: { projectId: number }) {
  const currentDoc = useStore((s) => s.doc);
  const requestFit = useStore((s) => s.requestFit);

  const { data } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => listDocuments(projectId),
    refetchInterval: 5000,
  });

  if (!data || data.length === 0) {
    return <div className="text-xs text-slate-500">No documents yet.</div>;
  }

  const items = data as Page[];

  // Group by source_file_id (PDF) or fall back to document_id (single image).
  type Group = { key: string; pages: Page[] };
  const groups = new Map<string, Page[]>();
  for (const d of items) {
    const key = d.source_file_id ?? d.document_id;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(d);
  }
  const ordered: Group[] = [...groups.entries()]
    .map(([key, pages]) => ({
      key,
      pages: [...pages].sort((a, b) => a.page_number - b.page_number),
    }))
    .sort((a, b) =>
      a.pages[0].document_id.localeCompare(b.pages[0].document_id),
    );

  return (
    <ul className="space-y-2">
      {ordered.map((g) => (
        <li key={g.key}>
          {g.pages.length > 1 && (
            <div className="mb-1 flex items-center gap-1 text-[10px] uppercase tracking-wider text-slate-500">
              <span>PDF</span>
              <span className="font-mono text-slate-400">{g.pages.length}</span>
              <span>pages</span>
            </div>
          )}
          <ul className="space-y-0.5 border-l border-slate-700 pl-2">
            {g.pages.map((d) => {
              const active = currentDoc?.document_id === d.document_id;
              return (
                <li key={d.document_id}>
                  <a
                    href={`#/${d.document_id}`}
                    onClick={(e) => {
                      if (active) {
                        e.preventDefault();
                        requestFit();
                      }
                    }}
                    className={
                      "block rounded px-2 py-1 text-xs " +
                      (active ? "bg-slate-700" : "hover:bg-slate-800")
                    }
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-slate-300">
                        {d.document_id}
                      </span>
                      {d.page_count > 1 && (
                        <span className="text-[10px] text-slate-500">
                          p.{d.page_number}/{d.page_count}
                        </span>
                      )}
                      <span
                        className={
                          "ml-auto rounded px-1 text-[10px] " +
                          (d.status === "generated"
                            ? "bg-violet-900/50 text-violet-200"
                            : d.status === "in_progress"
                              ? "bg-sky-900/50 text-sky-200"
                              : d.status === "submitted"
                                ? "bg-amber-900/50 text-amber-200"
                                : "bg-emerald-900/50 text-emerald-200")
                        }
                      >
                        {d.status}
                      </span>
                    </div>
                  </a>
                </li>
              );
            })}
          </ul>
        </li>
      ))}
    </ul>
  );
}