import { useEffect, useState } from "react";
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";

import { Stage } from "./canvas/Stage";
import { LeftPanel } from "./panels/Left";
import { RightPanel } from "./panels/Right";
import { Toolbar } from "./panels/Toolbar";
import { fetchSample, getDocument, getProject } from "./api";
import { useStore } from "./store";
import { usePageNav } from "./hooks/usePageNav";
import { useElementKeyboard } from "./hooks/useElementKeyboard";

const qc = new QueryClient();

function useDocumentIdFromHash(): string | null {
  const [id, setId] = useState<string | null>(() =>
    window.location.hash.replace(/^#\/?/, "") || null,
  );
  useEffect(() => {
    const onChange = () =>
      setId(window.location.hash.replace(/^#\/?/, "") || null);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return id;
}

function Shell() {
  const load = useStore((s) => s.load);
  const reloadIfClean = useStore((s) => s.reloadIfClean);
  const requestFit = useStore((s) => s.requestFit);
  const conflict = useStore((s) => s.conflict);
  const setConflict = useStore((s) => s.setConflict);
  const setProject = useStore((s) => s.setProject);
  const currentProjectId = useStore((s) => s.currentProjectId);
  const documentId = useDocumentIdFromHash();
  const nav = usePageNav(documentId);
  useElementKeyboard();
  // Load the active project once (and whenever currentProjectId changes) so
  // pickers/selects can read real doc_types / field_types from the backend.
  useEffect(() => {
    getProject(currentProjectId).then(setProject).catch(() => {});
  }, [currentProjectId, setProject]);

  const { data, error, isLoading } = useQuery({
    queryKey: ["document", documentId ?? "sample"],
    queryFn: async () => {
      if (documentId) {
        const r = await getDocument(documentId);
        return { doc: r.doc, revision: r.revision, image_url: r.image_url };
      }
      const doc = await fetchSample();
      return { doc, revision: 1, image_url: null as string | null };
    },
    // Poll only when a real document is loaded.
    refetchInterval: documentId ? 3000 : false,
    refetchOnWindowFocus: true,
  });

  // Initial load: seed the store.
  useEffect(() => {
    if (data) {
      const s = useStore.getState();
      if (!s.doc || s.doc.document_id !== data.doc.document_id) {
        load(data.doc, data.revision, data.image_url);
        setTimeout(() => requestFit(), 50);
      }
    }
  }, [data, load, requestFit]);

  // Subsequent polls: only reload if the server revision moved and we're clean.
  useEffect(() => {
    if (!data) return;
    const s = useStore.getState();
    if (!s.doc || s.doc.document_id !== data.doc.document_id) return;
    if (data.revision > s.serverRevision) {
      reloadIfClean(data.doc, data.revision, data.image_url);
    }
  }, [data, reloadIfClean]);

  // Page navigation keyboard shortcuts (N/P, ←/→, Home/End).
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      ) {
        return;
      }
      if (e.metaKey || e.ctrlKey) return;

      if (e.key === "n" || e.key === "N" || e.key === "ArrowRight") {
        e.preventDefault();
        nav.goNext();
      } else if (e.key === "p" || e.key === "P" || e.key === "ArrowLeft") {
        e.preventDefault();
        nav.goPrev();
      } else if (e.key === "Home") {
        e.preventDefault();
        nav.goFirst();
      } else if (e.key === "End") {
        e.preventDefault();
        nav.goLast();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nav]);

  // Wait for Noto Sans Devanagari so Konva measures text correctly.
  const [fontsReady] = useStateReady();

  if (isLoading || !fontsReady) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-400">
        Loading document…
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-rose-400">
        {String(error)}
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <Toolbar />

      {conflict === "remote-newer" && (
        <div className="flex items-center gap-2 border-b border-amber-800 bg-amber-950/60 px-3 py-1.5 text-xs text-amber-200">
          <span>
            Server has newer edits. Reload and discard your local changes?
          </span>
          <button
            onClick={() => {
              // Force reload: bypass the dirty check
              if (data) {
                useStore.setState({
                  doc: data.doc,
                  revision: data.revision,
                  serverRevision: data.revision,
                  imageUrl: data.image_url,
                  past: [],
                  future: [],
                  dirty: false,
                  conflict: "none",
                });
              }
            }}
            className="rounded bg-amber-800 px-2 py-0.5 text-[11px] hover:bg-amber-700"
          >
            Reload
          </button>
          <button
            onClick={() => setConflict("none")}
            className="ml-auto text-[11px] text-amber-300 hover:underline"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <LeftPanel projectId={1} />
        <main className="relative flex-1">
          <Stage />
        </main>
        <RightPanel />
      </div>
    </div>
  );
}

function useStateReady() {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let cancelled = false;
    document.fonts?.ready?.then(() => {
      if (!cancelled) setReady(true);
    }) ?? setReady(true);
    return () => {
      cancelled = true;
    };
  }, []);
  return [ready, setReady] as const;
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <Shell />
    </QueryClientProvider>
  );
}