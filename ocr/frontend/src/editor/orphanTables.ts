import type { Doc } from "../types";

/**
 * Remove tables that no table region points at any more (in place, inside an
 * immer recipe).
 *
 * Deleting a table region (Delete key, the Delete button, changing its type,
 * turning it into a checkbox …) only removed the layout entry. Its entry in
 * `doc.tables` stayed - with its cells, ruling lines and cell lines - so the
 * canvas kept drawing and hover-highlighting a table that is not in the
 * layout tree, and the saved JSON kept it.
 *
 * Kept tables are re-indexed in order and `table_index` on table regions,
 * lines and ruling lines follows. Lines of a dropped table go too, unless
 * their region still exists (then they only lose their cell link).
 * The backend's finalize() does the same on save (drop_orphan_tables).
 *
 * Returns the old → new index map for the kept tables, or null when nothing
 * changed (so immer sees no change and the edit stays a no-op).
 */
export function dropOrphanTables(d: Doc): Map<number, number> | null {
  const tables = d.tables ?? [];
  const n = tables.length;
  const used = new Set<number>();
  for (const el of d.layout) {
    if (el.type !== "table") continue;
    const ti = (el as { table_index?: number }).table_index;
    if (typeof ti === "number" && ti >= 0 && ti < n) used.add(ti);
  }
  if (used.size === n) return null;

  const keep = [...used].sort((a, b) => a - b);
  const remap = new Map<number, number>(keep.map((old, i) => [old, i]));
  d.tables = keep.map((i) => tables[i]);

  for (const el of d.layout) {
    if (el.type !== "table") continue;
    const t = el as { table_index: number };
    const ni = remap.get(t.table_index);
    if (ni !== undefined) t.table_index = ni;
  }

  const regionIds = new Set(d.layout.map((e) => e.id));
  d.lines = d.lines.filter((ln) => {
    if (ln.table_index == null) return true;
    const ni = remap.get(ln.table_index);
    if (ni !== undefined) {
      ln.table_index = ni;
      return true;
    }
    if (!regionIds.has(ln.block_id)) return false; // belonged to the deleted table only
    delete ln.table_index;
    delete ln.cell_id;
    return true;
  });

  d.ruling_lines = (d.ruling_lines ?? []).filter((r) => {
    const ni = remap.get(r.table_index);
    if (ni === undefined) return false;
    r.table_index = ni;
    return true;
  });
  return remap;
}
