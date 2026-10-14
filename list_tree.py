"""
Nested list detection for ocr_tool.py

Builds a TREE from the text lines inside one list block:

    list (style "num.")                      <- parent of everything, one box
      item 1  "1. ..."
      item 2  "2. ..."
      item 3  "3. ... (wrapped line joined)"
      item 4  "4. ..."
      item 5  "5. ..."                       <- box = its own lines + all children
        heading   "Local varieties"
        list (style "dev)")                  <- sub-list, its own box
          item "क) ..."
          item "ख) ..."

How a line is identified
  * marker line      -> new item. Its LEVEL comes from the marker style:
                        "1."  "(1)"  "क)"  "a)"  "i."  "5.1"  "•" are all different styles.
                        A style already open on the stack = sibling at that level (pop back),
                        a new style = child list of the last item (push).
                        "5.1" / "5.1.2" get their depth straight from the dots.
  * outdented line   -> pops back to the level whose marker x matches (handles a sub-list
                        written with the SAME style as the parent).
  * unmarked line    -> heading   if taller than body text, or short + no end punctuation
                                  and the next line opens a new sub-list
                        continuation of the current item otherwise
  * "i" / "v" / "x"  -> alpha if it continues an open a), b) ... sequence, roman otherwise.

Use:
    from list_tree import build_list_tree, tree_text, line_paths, absorb_nested_blocks

Custom labels (e.g. "options"):
    from list_tree import add_options_label, add_custom_label
    add_options_label()                       # "options" -> list
    add_custom_label("question_title", "heading")
"""
import re

_MARK_RE = re.compile(
    r"^\s*(?P<open>\()?"
    r"(?:(?P<num>[0-9\u0966-\u096f]+(?:\.[0-9\u0966-\u096f]+)*)"
    r"|(?P<roman>[ivxlcIVXLC]{1,6})"
    r"|(?P<alpha>[A-Za-z])"
    r"|(?P<dev>[\u0915-\u0939]))"
    r"(?P<close>\)\.?|[.)\u0964])?\s+(?P<rest>.*)$"
    r"|^\s*(?P<bullet>[\u2022\u25cf\u25cb\u25e6\u25aa\u25a0\u25a1\u25c6\u25c7\u25ba\u25b8\u27a2\u2713\u2714\u2756*\-\u2013])\s+(?P<brest>.*)$")

_END_PUNCT = ("\u0964", ".", ",", ";", ":", "?", "!")


def parse_marker(text):
    """-> dict(kind, style, value, depth, marker, rest) or None."""
    m = _MARK_RE.match(text or "")
    if not m:
        return None
    if m.group("bullet"):
        b = m.group("bullet")
        return {"kind": "bullet", "style": "bullet" + b, "value": None, "depth": 1,
                "marker": b, "rest": m.group("brest").strip()}
    close = m.group("close") or ""
    if not close and not (m.group("num") and "." in m.group("num")):
        return None                          # only "5.1 text" may omit the closing mark
    shape = ("(" if m.group("open") else "") + close
    for kind in ("num", "roman", "alpha", "dev"):
        v = m.group(kind)
        if v:
            depth = v.count(".") + 1 if kind == "num" else 1
            marker = text.strip()[:len(text.strip()) - len(m.group("rest"))].strip()
            return {"kind": kind, "style": f"{kind}{depth if kind == 'num' else ''}{shape}",
                    "value": v, "depth": depth, "marker": marker, "rest": m.group("rest").strip()}
    return None


def _letter_index(v):
    return ord(v.lower()) - ord("a") if len(v) == 1 and v.isascii() and v.isalpha() else None


def _resolve_roman_alpha(mk, stack):
    """'i', 'v', 'x', 'c', 'l' can be roman or alpha - decide from the open lists."""
    if mk["kind"] not in ("roman", "alpha"):
        return mk
    v = mk["value"]
    shape = mk["style"][len(mk["kind"]):]
    alpha_style, roman_style = "alpha" + shape, "roman" + shape
    open_alpha = next((l for l in reversed(stack) if l["style"] == alpha_style), None)
    open_roman = next((l for l in reversed(stack) if l["style"] == roman_style), None)
    if len(v) == 1 and open_alpha and open_alpha["items"]:
        prev = _letter_index(open_alpha["items"][-1]["value"] or "")
        if prev is not None and _letter_index(v) == prev + 1:
            return dict(mk, kind="alpha", style=alpha_style)
    if mk["kind"] == "roman" and len(v) == 1 and v.lower() not in "iv" and not open_roman:
        return dict(mk, kind="alpha", style=alpha_style)      # "c)" "l)" "x)" -> letters
    if mk["kind"] == "alpha" and open_roman:
        return dict(mk, kind="roman", style=roman_style)
    return mk


def _x0(L):
    return float(L["bbox"][0])


def _h(L):
    return float(L.get("fh") or L["bbox"][3])


def _union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _new_list(style, indent):
    return {"type": "list", "style": style, "indent": indent, "items": [], "bbox": None}


def _merge_row_splits(lines, hmed):
    """OCR often cuts one printed line into several boxes on the same row
    ("Required from" + "the ward office"). Join them left to right; the
    original boxes are kept in "_parts" so each still gets its item_path."""
    out = []
    for L in sorted(lines, key=lambda L: (L["bbox"][1], L["bbox"][0])):
        x, y, w, h = L["bbox"]
        hit = None
        for M in reversed(out[-3:]):
            mx, my, mw, mh = M["bbox"]
            ov = min(my + mh, y + h) - max(my, y)
            gap = x - (mx + mw)
            if ov >= 0.5 * min(h, mh) and -0.5 * hmed <= gap <= 3.0 * hmed \
                    and not parse_marker(L["text"]):
                hit = M
                break
        if hit is None:
            out.append(dict(L, _parts=[L]))
            continue
        hit["text"] = hit["text"].rstrip() + " " + L["text"].strip()
        hit["bbox"] = _union([tuple(hit["bbox"]), tuple(L["bbox"])])
        if L.get("fh") or hit.get("fh"):
            hit["fh"] = max(v for v in (hit.get("fh"), L.get("fh")) if v)
        hit["_parts"].append(L)
    return sorted(out, key=lambda L: (L["bbox"][1], L["bbox"][0]))


def build_list_tree(lines, hmed=None):
    """
    lines: dicts with "text" and "bbox" (x, y, w, h); optional "fh".
    Returns the root list node. Every node gets a "bbox" covering its whole subtree,
    every item gets "path" (e.g. [4, 1] = 5th item -> its first child).
    """
    lines = [L for L in lines if (L.get("text") or "").strip()]
    if not lines:
        return _new_list(None, 0)
    hs = sorted(_h(L) for L in lines)
    hmed = float(hmed or hs[len(hs) // 2])
    lines = _merge_row_splits(lines, hmed)
    maxw = max(L["bbox"][2] for L in lines)
    # right margin = the furthest edge that at least TWO lines reach, so one
    # stray wide line (a table row, a long title) can't hide every wrap
    rights = sorted((L["bbox"][0] + L["bbox"][2] for L in lines), reverse=True)
    right = next((v for v in rights
                  if sum(1 for r in rights if r >= v - hmed) >= 2), rights[0])
    left = min(L["bbox"][0] for L in lines)

    def full(L):        # does this line run to the right margin (so it can wrap)?
        return L["bbox"][0] + L["bbox"][2] >= right - 0.15 * (right - left)

    root = _new_list(None, None)
    root["intro"] = []                    # unmarked lines before the first item
    stack = [root]                        # open list nodes, outermost first
    last_item = None                      # deepest item seen so far

    def open_child_list(style, indent):
        parent = stack[-1]["items"][-1] if stack[-1]["items"] else None
        lst = _new_list(style, indent)
        if parent is None:                # nothing to hang it on: treat as root
            stack[-1].update(style=style, indent=indent)
            return stack[-1]
        parent["children"].append(lst)
        stack.append(lst)
        return lst

    def opens_sublist(j):
        if j >= len(lines):
            return False
        mk = parse_marker(lines[j]["text"])
        if not mk:
            return False
        st = _resolve_roman_alpha(mk, stack)["style"]
        return all(l["style"] != st for l in stack)

    def same_size(a, b, tol=0.25):
        return abs(_h(a) - _h(b)) <= tol * max(_h(a), _h(b))

    def starts_new_paragraph(prev_l, cur):
        """A body-size line that is wide although the line above stopped short:
        the text above ended, a new block of text begins here."""
        return (_h(cur) < 1.2 * hmed and not full(prev_l)
                and cur["bbox"][2] > 0.8 * maxw)

    def continues_run(prev_l, cur):
        return (not parse_marker(cur["text"])
                and same_size(prev_l, cur, tol=0.15)
                and near_below(prev_l, cur)
                and not starts_new_paragraph(prev_l, cur))

    def near_below(a, b):
        return b["bbox"][1] - (a["bbox"][1] + a["bbox"][3]) <= 1.0 * max(_h(a), _h(b))

    def add_text(node_lines, L):        # where an unmarked continuation goes
        tail = last_item["children"][-1] if last_item["children"] else None
        if tail and tail["type"] == "paragraph":
            tail["text"] += " " + L["text"].strip()
            tail["lines"].append(L)
        elif tail:
            last_item["children"].append({"type": "paragraph", "text": L["text"].strip(),
                                          "lines": [L]})
        else:
            last_item["text"] = (last_item["text"] + " " + L["text"].strip()).strip()
            last_item["lines"].append(L)

    k = 0
    while k < len(lines):
        L = lines[k]
        text = L["text"].strip()
        mk = parse_marker(text)
        x0 = _x0(L)

        if mk:
            mk = _resolve_roman_alpha(mk, stack)
            style = mk["style"]
            same = [i for i, l in enumerate(stack) if l["style"] == style]
            if root["style"] is None:
                root.update(style=style, indent=x0)
            elif mk["kind"] == "num" and mk["depth"] > 1 and not same:
                while len(stack) > 1 and stack[-1]["style"].startswith("num") and \
                        int(re.match(r"num(\d+)", stack[-1]["style"]).group(1)) >= mk["depth"]:
                    stack.pop()
                open_child_list(style, x0)
            elif same:
                del stack[same[-1] + 1:]
            elif x0 < stack[-1]["indent"] - 0.6 * hmed and len(stack) > 1:
                i = min(range(len(stack)), key=lambda i: abs(stack[i]["indent"] - x0))
                del stack[i + 1:]
            else:
                open_child_list(style, x0)
            item = {"type": "item", "marker": mk["marker"], "value": mk["value"],
                    "text": mk["rest"], "lines": [L], "children": [],
                    "heading": _h(L) >= 1.2 * hmed}
            stack[-1]["items"].append(item)
            last_item = item
            k += 1
            continue

        # ---- run of unmarked lines --------------------------------------------
        prev = lines[k - 1] if k else None
        # a real wrap: the line above is full width and this line is body size
        if last_item is not None and prev is not None and full(prev) and same_size(prev, L) \
                and near_below(prev, L) and _h(L) < 1.2 * hmed:
            add_text(None, L)
            k += 1
            continue
        e = k + 1
        while e < len(lines) and continues_run(lines[e - 1], lines[e]):
            e += 1
        run = lines[k:e]
        joined = " ".join(r["text"].strip() for r in run)
        tall = all(_h(r) >= 1.2 * hmed for r in run)
        heading_like = (len(joined) <= 120 and not joined.endswith(("\u0964", ".", ";", ","))
                        and len(run) <= 3)
        is_heading = heading_like and (tall or opens_sublist(e))

        if is_heading:
            node = {"type": "heading", "text": joined, "lines": list(run)}
            (root["intro"] if last_item is None else last_item["children"]).append(node)
            k = e
            continue
        if last_item is None:
            root["intro"].append({"type": "paragraph", "text": joined, "lines": list(run)})
            k = e
            continue
        # body text that is not a wrap of the line above: its own paragraph child
        last_item["children"].append({"type": "paragraph", "text": joined, "lines": list(run)})
        k = e

    _finish(root, [])
    return root


def _finish(node, path):
    """Fill bbox (union of subtree) and path; returns the node's bbox."""
    boxes = [tuple(L["bbox"]) for L in node.get("lines", [])]
    for e in node.get("intro", []):
        e["bbox"] = _union([tuple(L["bbox"]) for L in e["lines"]])
        boxes.append(e["bbox"])
    if node["type"] == "list":
        for i, it in enumerate(node["items"]):
            boxes.append(_finish(it, path + [i]))
    elif node["type"] == "item":
        node["path"] = path
        for ch in node["children"]:
            if ch["type"] == "list":
                boxes.append(_finish(ch, path))
            else:
                ch["bbox"] = _union([tuple(L["bbox"]) for L in ch["lines"]])
                boxes.append(ch["bbox"])
    node["bbox"] = _union(boxes)
    return node["bbox"]


def tree_text(node, indent="", _depth=0):
    """One item / heading per line (synthetic v2 convention). With indent="  "
    each nesting level is indented, e.g. for a readable preview."""
    pad = indent * _depth
    out = [pad + e["text"] for e in node.get("intro", [])]
    for it in node["items"]:
        out.append(pad + (it["marker"] + " " + it["text"]).strip())
        for ch in it["children"]:
            out.append(tree_text(ch, indent, _depth + 1) if ch["type"] == "list"
                       else indent * (_depth + 1) + ch["text"])
    return "\n".join(t for t in out if t.strip())


def line_paths(node):
    """id(line) -> item path, to write "item_path" on each exported line."""
    res = {}

    def walk(n):
        for it in n["items"]:
            for L in it["lines"]:
                for P in L.get("_parts", [L]):
                    res[id(P)] = it["path"]
            for ch in it["children"]:
                if ch["type"] == "list":
                    walk(ch)
                else:
                    for L in ch["lines"]:
                        for P in L.get("_parts", [L]):
                            res[id(P)] = it["path"]
    walk(node)
    return res


def tree_for_export(node):
    """JSON-safe copy (drops line dicts, keeps boxes as xyxy)."""
    def box(b):
        return None if b is None else [round(b[0], 1), round(b[1], 1),
                                       round(b[0] + b[2], 1), round(b[1] + b[3], 1)]
    if node["type"] == "list":
        return {"type": "list", "style": node["style"], "bbox": box(node["bbox"]),
                "items": [tree_for_export(i) for i in node["items"]]}
    if node["type"] == "item":
        return {"type": "item", "path": node["path"], "marker": node["marker"],
                "text": node["text"], "bbox": box(node["bbox"]), "is_heading": node["heading"],
                "children": [tree_for_export(c) for c in node["children"]]}
    return {"type": node["type"], "text": node["text"],
            "bbox": box(_union([tuple(L["bbox"]) for L in node["lines"]]))}


# ----------------------------------------------------------------------------
# Block-level pass for auto_annotate_page: glue a heading / sub-list that the
# line chainer split off back into the list it belongs to.
# ----------------------------------------------------------------------------
_LABEL_ALIASES = {"list_item": "list", "paragraph_title": "heading", "doc_title": "title",
                  "text": "paragraph", "content": "paragraph", "reference_content": "list"}

# canonical targets the pipeline understands end-to-end
_CANONICAL_LABELS = {"list", "heading", "subheading", "paragraph", "title",
                     "text_line", "form_label", "form_value"}


def _canon(label):
    label = (label or "paragraph").strip()
    return _LABEL_ALIASES.get(label, label)


# ----------------------------------------------------------------------------
# Custom label registration (e.g. "options" for MCQ choice lists).
#
#   Both absorb_nested_blocks() and merge_list_detections() run every incoming
#   label through _canon() BEFORE comparing it, so mapping an alias to "list"
#   is enough to make the whole pipeline (build_list_tree, glue logic, flat
#   export, line_paths) treat it exactly like a numbered list.
# ----------------------------------------------------------------------------
def add_options_label(*aliases):
    """
    Register an "options" label (and common spellings) as a list.

        from list_tree import add_options_label
        add_options_label()                      # -> "options"
        add_options_label("options", "choices")  # -> both

    Every detection whose "label" is one of these will be:
      * parsed by build_list_tree (क)/ख)/a)/i)/• markers become list items)
      * absorbed / merged by absorb_nested_blocks & merge_list_detections
      * exported with items[], item_path, list_level, item_index like any list
    """
    names = aliases or ("options",)
    for a in names:
        a = (a or "").strip()
        if a:
            _LABEL_ALIASES[a] = "list"
    for a in ("option", "choice", "choices",
              "mcq_option", "mcq_options",
              "answer_option", "answer_options"):
        _LABEL_ALIASES[a] = "list"
    return "list"


def add_custom_label(alias, canonical="list"):
    """
    Generic version of add_options_label — alias -> any canonical label.

        add_custom_label("question_title", "heading")
        add_custom_label("inline",         "text_line")
        add_custom_label("myopts",         "list")
    """
    alias = (alias or "").strip()
    if not alias:
        raise ValueError("alias must be a non-empty string")
    canonical = _canon(canonical)
    if canonical not in _CANONICAL_LABELS:
        raise ValueError(
            f"canonical label {canonical!r} unknown; "
            f"expected one of {sorted(_CANONICAL_LABELS)}"
        )
    _LABEL_ALIASES[alias] = canonical
    return canonical


def absorb_nested_blocks(blocks, hmed):
    """
    blocks: [{"lines": [...], "label": ...}, ...] as built in auto_annotate_page.
    A block directly below a list joins it when it is
      * another list (sibling items "6." or an indented sub-list), or
      * a heading / subheading / paragraph that is indented inside the list and
        is followed by an indented list (the "5. -> heading -> क) ख)" case).
    """
    def bb(b):
        xs0 = min(L["bbox"][0] for L in b["lines"])
        ys0 = min(L["bbox"][1] for L in b["lines"])
        xs1 = max(L["bbox"][0] + L["bbox"][2] for L in b["lines"])
        ys1 = max(L["bbox"][1] + L["bbox"][3] for L in b["lines"])
        return xs0, ys0, xs1, ys1

    blocks = [dict(b, label=_canon(b.get("label"))) for b in blocks]
    order = sorted(blocks, key=lambda b: bb(b)[1])
    out, i = [], 0
    while i < len(order):
        b = order[i]
        if b["label"] != "list":
            out.append(b)
            i += 1
            continue
        group = dict(b, lines=list(b["lines"]))
        lx0, _ly0, lx1, ly1 = bb(group)
        j = i + 1
        while j < len(order):
            c = order[j]
            cx0, cy0, cx1, cy1 = bb(c)
            close = cy0 - ly1 <= 2.5 * hmed
            overlaps = min(lx1, cx1) - max(lx0, cx0) > 0
            inside = cx0 >= lx0 - 0.6 * hmed
            if not (close and overlaps and inside):
                break
            if c["label"] == "list":
                ok = True
            elif c["label"] in ("heading", "subheading", "paragraph"):
                nxt = order[j + 1] if j + 1 < len(order) else None
                ok = (cx0 > lx0 + 0.6 * hmed or c["label"] != "paragraph") and \
                     nxt is not None and nxt["label"] == "list" and \
                     bb(nxt)[0] > lx0 + 0.6 * hmed and bb(nxt)[1] - cy1 <= 2.5 * hmed
            else:
                ok = False
            if not ok:
                break
            group["lines"] += c["lines"]
            lx0, _ly0, lx1, ly1 = bb(group)
            j += 1
        out.append(group)
        i = j
    return out


# ----------------------------------------------------------------------------
# Parent text from children (bottom-up) - every node knows all text below it
# ----------------------------------------------------------------------------
def node_full_text(node):
    """Text of a node INCLUDING every child and grandchild, one item/heading per line."""
    if node["type"] == "list":
        return tree_text(node)
    if node["type"] == "item":
        out = [(node["marker"] + " " + node["text"]).strip()]
        for ch in node["children"]:
            out.append(node_full_text(ch))
        return "\n".join(t for t in out if t)
    return node["text"]


def flatten_tree(tree):
    """
    Flat node list, each parent naming its children:
      {"id", "parent_id", "type", "path", "marker", "text" (own words only),
       "full_text" (own + all descendants), "children": [ids], "bbox": xyxy}
    """
    nodes = []

    def box(b):
        return None if b is None else [round(b[0], 1), round(b[1], 1),
                                       round(b[0] + b[2], 1), round(b[1] + b[3], 1)]

    def add(node, parent_id):
        nid = len(nodes)
        if node["type"] == "list":
            bb = node["bbox"]
        elif node["type"] == "item":
            bb = node["bbox"]
        else:
            bb = _union([tuple(L["bbox"]) for L in node["lines"]])
        rec = {"id": nid, "parent_id": parent_id, "type": node["type"],
               "path": node.get("path"), "marker": node.get("marker"),
               "style": node.get("style"),
               "text": node.get("text", "") if node["type"] != "list" else "",
               "full_text": node_full_text(node), "children": [], "bbox": box(bb)}
        nodes.append(rec)
        kids = node["items"] if node["type"] == "list" else node.get("children", [])
        if node["type"] == "list":
            for e in node.get("intro", []):
                rec["children"].append(add(e, nid))
        for k in kids:
            rec["children"].append(add(k, nid))
        return nid

    add(tree, None)
    return nodes


# ----------------------------------------------------------------------------
# Parent text -> children (top-down): correct the block text ONCE and push the
# words down to every line box, so lines never have to be edited one by one.
# ----------------------------------------------------------------------------
def _tok(w):
    return re.sub(r"[^\wऀ-ॿ]", "", w).lower()


def distribute_text_to_lines(new_text, lines, min_ratio=0.35):
    """
    new_text: the corrected text of the parent block (any line breaks).
    lines:    the child line dicts ("text", "bbox"); they are NOT modified.
    Returns (list_of_new_line_texts in the order of `lines`, match_ratio),
    or (None, ratio) when the OCR is too different to align safely.

    Words are aligned with difflib against the OCR words of each line, so a
    corrected word lands on the line box it was read from; added words stick to
    the previous word's line; removed words simply disappear.
    """
    import difflib
    order = sorted(range(len(lines)), key=lambda i: (lines[i]["bbox"][1], lines[i]["bbox"][0]))
    ocr_words, owner = [], []
    for i in order:
        for w in (lines[i].get("text") or "").split():
            ocr_words.append(w)
            owner.append(i)
    gt_rows = [r.split() for r in (new_text or "").split("\n") if r.strip()]
    gt_words = [w for r in gt_rows for w in r]
    out = {i: [] for i in range(len(lines))}

    if not ocr_words or not gt_words:
        return None, 0.0
    sm = difflib.SequenceMatcher(None, [_tok(w) for w in ocr_words],
                                 [_tok(w) for w in gt_words], autojunk=False)
    ratio = sm.ratio()
    if ratio < min_ratio:
        # alignment unreliable: only safe if the typed rows match the line count
        if len(gt_rows) == len(lines):
            res = [""] * len(lines)
            for i, r in zip(order, gt_rows):
                res[i] = " ".join(r)
            return res, ratio
        return None, ratio

    last = owner[0]
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "delete":
            last = owner[a1 - 1]
            continue
        n = b1 - b0
        for k in range(n):
            if tag == "insert":
                tgt = last          # words added at the very start go to the first line
            else:  # equal / replace: spread the new words over the old ones
                src = a0 + min(a1 - a0 - 1, (k * (a1 - a0)) // n)
                tgt = owner[src]
            out[tgt].append(gt_words[b0 + k])
            last = tgt
        if a1 > a0:
            last = owner[a1 - 1]
    return [" ".join(out[i]) for i in range(len(lines))], ratio


# ----------------------------------------------------------------------------
# Merge one-item list BLOCKS into one list (works on ocr_tool detections).
# Layout models (PPStructureV3) give one box per table-of-contents line, so
# "१." "२." ... "५." and "५.१)" "५.२)" arrive as separate list blocks.  They are
# joined when the numbering continues: ३. after २., ५.१) after ५., ६. after ५.३),
# a new style starting at 1 / क / a / i (a sub-list), or a short heading that
# sits directly before such a sub-list.
# ----------------------------------------------------------------------------
_DEV_DIGITS = "०१२३४५६७८९"
_DEV_LETTERS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
_LINE_LABELS = {"text_line", "form_label", "form_value"}
_TEXT_BLOCKS = {"list", "heading", "subheading", "paragraph"}


def _roman_val(s):
    s = s.lower()
    tot = 0
    for i, ch in enumerate(s):
        v = _ROMAN.get(ch, 0)
        tot += -v if i + 1 < len(s) and _ROMAN.get(s[i + 1], 0) > v else v
    return tot


def marker_values(mk):
    """Possible positions of a marker as tuples, e.g. '५.१)' -> [(5, 1)].
    'i' can be roman 1 or the letter i (9), so it returns both."""
    if not mk or mk["kind"] == "bullet":
        return []
    v = mk["value"]
    if mk["kind"] == "num":
        return [tuple(int("".join(str(_DEV_DIGITS.index(c)) if c in _DEV_DIGITS else c
                                  for c in part)) for part in v.split("."))]
    if mk["kind"] == "dev":
        return [(_DEV_LETTERS.index(v) + 1,)] if v in _DEV_LETTERS else []
    out = []
    if all(c.lower() in _ROMAN for c in v):
        out.append((_roman_val(v),))
    if len(v) == 1:
        out.append((ord(v.lower()) - 96,))
    return out


def _shape(mk):
    return mk["style"].lstrip("numromanalphadevbullet0123456789")


def follows(prev, mk):
    """Does marker `mk` continue the list whose markers so far are `prev`?
    Returns "sibling", "child" or None."""
    if mk is None:
        return None
    if mk["kind"] == "bullet":
        return "sibling" if any(p["style"] == mk["style"] for p in prev) else None
    vals = marker_values(mk)
    for p in reversed(prev):                       # same style seen: next number?
        if p["kind"] == "bullet" or _shape(p) != _shape(mk):
            continue
        same_kind = p["kind"] == mk["kind"] or {p["kind"], mk["kind"]} <= {"roman", "alpha"}
        if not same_kind or (mk["kind"] == "num" and p["depth"] != mk["depth"]):
            continue
        for pv in marker_values(p):
            for v in vals:
                if v[:-1] == pv[:-1] and v[-1] == pv[-1] + 1:
                    return "sibling"
        break
    if mk["kind"] == "num" and mk["depth"] > 1:    # ५.१ after ५ / 5.2.1 after 5.2
        for v in vals:
            if v[-1] == 1 and any(v[:-1] in marker_values(p) for p in prev if p["kind"] == "num"):
                return "child"
        return None
    if any(v[-1] == 1 for v in vals) and all(p["style"] != mk["style"] for p in prev):
        return "child"                             # new style starting at 1 / क / a / i
    return None


def merge_list_detections(dets, hmed=None):
    """
    dets: ocr_tool detections (blocks + text_line boxes, bbox = x, y, w, h).
    Returns (new_dets, number_of_blocks_merged_away). Merged list blocks get the
    union box and their text rebuilt from the tree; the rest is untouched.
    """
    lab = lambda d: _canon(d.get("label"))
    lines = [d for d in dets if lab(d) in _LINE_LABELS]
    blocks = [d for d in dets if lab(d) in _TEXT_BLOCKS]
    if not lines or len(blocks) < 2:
        return dets, 0

    def center_in(L, b):
        x, y, w, h = b["bbox"]
        cx = L["bbox"][0] + L["bbox"][2] / 2.0
        cy = L["bbox"][1] + L["bbox"][3] / 2.0
        return x <= cx <= x + w and y <= cy <= y + h

    owned = {id(b): [] for b in blocks}
    for L in lines:
        inside = [b for b in blocks if center_in(L, b)]
        if inside:
            b = min(inside, key=lambda b: b["bbox"][2] * b["bbox"][3])
            owned[id(b)].append(L)
    hs = sorted(L["bbox"][3] for L in lines)
    hmed = float(hmed or hs[len(hs) // 2])

    def ext(b):
        ls = owned[id(b)]
        if not ls:
            x, y, w, h = b["bbox"]
            return x, y, x + w, y + h
        return (min(L["bbox"][0] for L in ls), min(L["bbox"][1] for L in ls),
                max(L["bbox"][0] + L["bbox"][2] for L in ls),
                max(L["bbox"][1] + L["bbox"][3] for L in ls))

    def markers(b):
        ls = sorted(owned[id(b)], key=lambda L: (L["bbox"][1], L["bbox"][0]))
        return [m for m in (parse_marker(L.get("text")) for L in ls) if m]

    def first_marker(b):
        ls = sorted(owned[id(b)], key=lambda L: (L["bbox"][1], L["bbox"][0]))
        return parse_marker(ls[0].get("text")) if ls else None

    def near(a_ext, b_ext):
        return (b_ext[1] - a_ext[3] <= 2.5 * hmed
                and min(a_ext[2], b_ext[2]) - max(a_ext[0], b_ext[0]) > 0)

    def listish(b):
        return lab(b) == "list" or first_marker(b) is not None

    order = sorted((b for b in blocks if owned[id(b)]), key=lambda b: ext(b)[1])
    gone, merged_away = set(), 0
    i = 0
    while i < len(order):
        g = order[i]
        if not listish(g) or id(g) in gone:
            i += 1
            continue
        seen, g_ext, members = markers(g), ext(g), [g]
        j = i + 1
        while j < len(order):
            c = order[j]
            c_ext = ext(c)
            if not near(g_ext, c_ext):
                break
            take = []
            if listish(c) and follows(seen, first_marker(c)):
                take = [c]
            elif lab(c) in ("heading", "subheading", "paragraph") and len(owned[id(c)]) <= 3 \
                    and first_marker(c) is None \
                    and j + 1 < len(order) and listish(order[j + 1]) \
                    and near(c_ext, ext(order[j + 1])) \
                    and follows(seen, first_marker(order[j + 1])) == "child":
                take = [c, order[j + 1]]
            if not take:
                break
            for t in take:
                members.append(t)
                seen += markers(t)
                e = ext(t)
                g_ext = (min(g_ext[0], e[0]), min(g_ext[1], e[1]),
                         max(g_ext[2], e[2]), max(g_ext[3], e[3]))
            j += len(take)
        if len(members) > 1:
            all_lines = [L for m in members for L in owned[id(m)]]
            boxes = [tuple(m["bbox"]) for m in members] + [tuple(L["bbox"]) for L in all_lines]
            g["bbox"] = tuple(int(round(v)) for v in _union(boxes))
            g["label"] = "list"
            g["text"] = list_block_text(all_lines)
            if not all(m.get("status") == "reviewed" for m in members):
                g["status"] = g.get("status") if g.get("status") != "reviewed" else "auto"
            for m in members[1:]:
                gone.add(id(m))
                owned[id(g)] += owned[id(m)]
            merged_away += len(members) - 1
        i = j
    return [d for d in dets if id(d) not in gone], merged_away


# ----------------------------------------------------------------------------
# Flat export format (synthetic v2 sample):
#   layout list  -> "items": [{"marker": "१.", "level": 0, "text": ...}, ...]
#                   "text":  one item per line, 2 spaces per level, bullets as "-"
#   each line    -> "list_level", "item_index"
# A heading / paragraph that sits inside an item (no marker of its own) becomes
# an item with "marker": null one level below that item.
# ----------------------------------------------------------------------------
def flat_items(tree):
    """-> (items, line_map) where line_map[id(line)] = (item_index, level)."""
    items, line_map = [], {}

    def own(lines_, idx, level):
        for L in lines_:
            for P in L.get("_parts", [L]):
                line_map[id(P)] = (idx, level)

    def walk(lst, level):
        for it in lst["items"]:
            idx = len(items)
            items.append({"marker": it["marker"], "level": level, "text": it["text"]})
            own(it["lines"], idx, level)
            for ch in it["children"]:
                if ch["type"] == "list":
                    walk(ch, level + 1)
                else:
                    cidx = len(items)
                    items.append({"marker": None, "level": level + 1, "text": ch["text"]})
                    own(ch["lines"], cidx, level + 1)

    for e in tree.get("intro", []):
        idx = len(items)
        items.append({"marker": None, "level": 0, "text": e["text"]})
        own(e["lines"], idx, 0)
    walk(tree, 0)
    return items, line_map


def items_text(items):
    """Block text in the sample's style: '  ' per level, bullets written as '-'."""
    out = []
    for it in items:
        m = it.get("marker")
        if m and parse_marker(m + " x") and parse_marker(m + " x")["kind"] == "bullet":
            m = "-"
        out.append("  " * int(it.get("level") or 0) + (f"{m} " if m else "") + (it.get("text") or ""))
    return "\n".join(o.rstrip() for o in out if o.strip())


def list_block_text(lines):
    """Text for a list block straight from its line boxes."""
    return items_text(flat_items(build_list_tree(lines))[0])
