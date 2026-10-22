/**
 * Change the open page's workflow status (submit / approve / reject).
 *
 * Status only changes through POST /documents/{id}/status — saving never
 * changes it. After a change the editor's own copy (doc.annotation.status
 * and the revision number) is updated, so the next Save doesn't hit a
 * revision conflict and doesn't send a stale status back.
 */
import { useStore } from "../store";
import { describe, saveCurrent } from "./save";

export type StatusTarget = "in_progress" | "submitted" | "approved" | "rejected";

export async function changeStatus(
  target: StatusTarget,
  comment?: string,
): Promise<boolean> {
  // unsaved edits go to the server first, so they are part of what is submitted/approved
  if (useStore.getState().dirty && !(await saveCurrent())) return false;

  const doc = useStore.getState().doc;
  if (!doc) return false;

  try {
    const r = await fetch(`/documents/${doc.document_id}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target, comment: comment ?? null }),
    });
    const text = await r.text();

    if (!r.ok) {
      useStore.getState().setSaveStatus({
        state: "error",
        message: `Status not changed to "${target}" (HTTP ${r.status}):\n${describe(text)}`,
      });
      return false;
    }

    const body = JSON.parse(text) as { revision: number; status: string };
    useStore.setState((s) => ({
      revision: body.revision,
      serverRevision: body.revision,
      doc:
        s.doc && s.doc.document_id === doc.document_id
          ? {
              ...s.doc,
              annotation: { ...s.doc.annotation, status: body.status as Doc["annotation"]["status"] },
            }
          : s.doc,
    }));
    useStore.getState().setSaveStatus({ state: "saved", at: Date.now() });
    return true;
  } catch (e) {
    useStore.getState().setSaveStatus({
      state: "error",
      message: `Status not changed to "${target}": ${String(e)}`,
    });
    return false;
  }
}

type Doc = NonNullable<ReturnType<typeof useStore.getState>["doc"]>;
