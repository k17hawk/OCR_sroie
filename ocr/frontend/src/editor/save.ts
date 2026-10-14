/**
 * Save the open page — used by the Save button, Ctrl/⌘+S and "Submit".
 *
 * Before, a failed save threw inside the click handler: nothing was shown,
 * the page stayed "dirty", and it looked like the tool would not save.
 * Now every outcome is reported through `saveStatus` in the store:
 *   409  the server has a newer revision (auto-annotate finished, another
 *        tab saved) → ask, then overwrite with the server's revision number
 *   422  the server could not read the page → the first problems are listed
 *   else network / server error text
 * Edits made while the request is in flight stay "unsaved".
 */
import { useStore } from "../store";

type Detail = { where?: string; loc?: unknown[]; msg?: string; got?: unknown };

function describe(body: string): string {
  try {
    const j = JSON.parse(body);
    const d = j?.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      const parts = (d as Detail[]).slice(0, 4).map((x) => {
        const where = x.where ?? (Array.isArray(x.loc) ? x.loc.slice(2).join(".") : "");
        return `${where}: ${x.msg ?? ""}${x.got !== undefined ? ` (got ${JSON.stringify(x.got)})` : ""}`;
      });
      return parts.join("\n") + (d.length > 4 ? `\n… and ${d.length - 4} more` : "");
    }
  } catch {
    /* not JSON */
  }
  return body.slice(0, 400) || "unknown error";
}

async function put(id: string, revision: number, doc: unknown) {
  const r = await fetch(`/documents/${id}/annotation`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision, doc }),
  });
  const text = await r.text();
  return { status: r.status, text };
}

export async function saveCurrent(): Promise<boolean> {
  const st = useStore.getState();
  const doc = st.doc;
  if (!doc) return false;
  if (st.saveStatus.state === "saving") return false;
  st.setSaveStatus({ state: "saving" });

  try {
    let res = await put(doc.document_id, st.revision, doc);

    if (res.status === 409) {
      // someone/something saved this page after we loaded it
      const latest = await fetch(`/documents/${doc.document_id}`).then((r) => r.json()).catch(() => null);
      const serverRev = latest?.revision;
      const ok =
        typeof serverRev === "number" &&
        window.confirm(
          `This page was changed on the server after you opened it ` +
            `(server revision ${serverRev}, yours ${st.revision}).\n\n` +
            `OK = overwrite it with what you see now\nCancel = keep editing (nothing is saved)`,
        );
      if (!ok) {
        useStore.getState().setSaveStatus({
          state: "error",
          message: "Not saved: the server has a newer version of this page. Save again to overwrite it, or reload the page to take the server's version.",
        });
        return false;
      }
      res = await put(doc.document_id, serverRev, doc);
    }

    if (res.status < 200 || res.status >= 300) {
      useStore.getState().setSaveStatus({
        state: "error",
        message: `Not saved (HTTP ${res.status}):\n${describe(res.text)}`,
      });
      return false;
    }

    const body = JSON.parse(res.text);
    const now = useStore.getState();
    if (now.doc === doc) {
      now.markSaved(body.revision);
    } else {
      // edited while saving: keep those edits marked as unsaved
      useStore.setState({ revision: body.revision, serverRevision: body.revision });
    }
    useStore.getState().setSaveStatus({ state: "saved", at: Date.now() });
    return true;
  } catch (e) {
    useStore.getState().setSaveStatus({ state: "error", message: `Not saved: ${String(e)}` });
    return false;
  }
}
