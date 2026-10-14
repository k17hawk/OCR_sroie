import { useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { listDocuments } from "../api";
import { useStore } from "../store";
import type { DocumentSummary } from "../types";

interface PageNav {
  prev: DocumentSummary | null;
  next: DocumentSummary | null;
  position: string;           // "3 / 12"
  total: number;
  goPrev: () => void;
  goNext: () => void;
  goFirst: () => void;
  goLast: () => void;
}

/**
 * Navigate across every page of every document in the project.
 * Ordering: by document_id then by page_number, so a 3-page PDF is
 * traversed as page 1 -> page 2 -> page 3, then the next document.
 */
export function usePageNav(currentDocumentId: string | null): PageNav {
  const projectId = useStore((s) => s.currentProjectId);

  const { data } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => listDocuments(projectId),
    // Match the queue's polling so navigation reflects new uploads.
    refetchInterval: 5000,
  });

  const ordered = useMemo<DocumentSummary[]>(() => {
    if (!data) return [];
    return [...data].sort(
      (a, b) =>
        a.document_id.localeCompare(b.document_id) ||
        a.page_number - b.page_number,
    );
  }, [data]);

  const idx = useMemo(
    () => ordered.findIndex((d) => d.document_id === currentDocumentId),
    [ordered, currentDocumentId],
  );

  const prev = idx > 0 ? ordered[idx - 1] : null;
  const next = idx >= 0 && idx < ordered.length - 1 ? ordered[idx + 1] : null;
  const position = idx >= 0 && ordered.length > 0
    ? `${idx + 1} / ${ordered.length}`
    : "—";

  const go = useCallback((id: string | null | undefined) => {
    if (!id) return;
    if (window.location.hash !== `#/${id}`) {
      window.location.hash = `#/${id}`;
    }
  }, []);

  return {
    prev,
    next,
    position,
    total: ordered.length,
    goPrev: () => go(prev?.document_id),
    goNext: () => go(next?.document_id),
    goFirst: () => go(ordered[0]?.document_id),
    goLast: () => go(ordered[ordered.length - 1]?.document_id),
  };
}