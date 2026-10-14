"""
OCR GUI Tool — Images + PDF, bounding boxes, image info, page navigation,
preprocessing, regex field extraction, pluggable OCR backends, table
recognition, line detection, underline/bold/italic style detection, and
optional Ollama-assisted correction.

Four-stage pipeline:
    Layout detector      — block boxes on the page (PaddleStructureV3 or manual)
    Line detector        — per-line boxes inside each block (Tesseract psm 6 grouping,
                           with newline-proportional fallback)
    Line recognizer      — text per line (Tesseract/EasyOCR/Surya/PaddleOCR)
    Table structure      — grid of each table (PaddleStructureV3 + HTML cell parser)

Text style attributes (per block and per line):
    is_underlined, is_bold, is_italic
  * is_underlined can be auto-detected from a line crop (dark horizontal run
    just below the ink) or set manually with the "Toggle Underline" button /
    Ctrl+U.
  * All three are exported in every JSON format and preserved on import.

Backends:
    tesseract          — baseline, needs tesseract-ocr-nep on Ubuntu
    easyocr            — native Nepali/Hindi, ~2GB torch install
    surya              — best open Devanagari quality, GPU recommended
    paddleocr          — lightweight dedicated devanagari_PP-OCRv5_server_rec
    paddlestructurev3  — layout + table recognition (returns HTML tables)
    auto               — tries surya -> paddlestructurev3 -> paddleocr -> easyocr -> tesseract

Annotation workflow:
    * Opening an image/PDF auto-loads an existing sidecar annotation file from
      the same folder or "annotations/" subfolder.
    * Opening a .json/.jsonl directly finds and opens the referenced image.
    * "Page folders (PDF + images)" (on by default): opening a PDF writes
          <name>_pages/<name>_pageNNN.png + manifest.json
      Ctrl+S writes one <name>_pageNNN.json per page next to its image.
    * Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) undo / redo box edits (per page).
    * Ctrl+A selects all text in any text field.
    * Ctrl+U toggles is_underlined on the selected detection.
    * Delete / BackSpace on the canvas or list deletes the selected box.

Ubuntu setup:
    sudo apt install tesseract-ocr tesseract-ocr-nep fonts-noto-devanagari libraqm0
    pip install --upgrade --force-reinstall pillow
    pip install opencv-python pytesseract pillow numpy pymupdf tkinterdnd2 requests
    pip install easyocr
    pip install surya-ocr
    pip install "paddleocr>=3.0.2" paddlepaddle
    ollama pull gemma3:4b
"""

import base64
import copy
import csv
import glob
import inspect
import json
import os
import platform
import re
import threading
import time
import tkinter as tk
import unicodedata
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

OLLAMA_URL = "http://localhost:11434"
DEFAULT_TEXT_MODEL = "gemma3:4b"
DEFAULT_VISION_MODEL = "gemma3:4b"
PREFERRED_MODEL_PREFIXES = ["gemma3", "qwen2.5vl", "qwen3", "qwen2.5"]

OLLAMA_OPTIONS = {"temperature": 0, "num_ctx": 2048, "num_predict": 768}
OLLAMA_KEEP_ALIVE = "30m"
BATCH_CHUNK_SIZE = 10

DEVANAGARI_REC_MODEL = "devanagari_PP-OCRv5_server_rec"

FORCE_DEVANAGARI_FONT = os.path.expanduser("~/.local/share/fonts/Nirmala.ttf")

MAX_UNDO = 200
IMAGE_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

# ----------------------------------------------------------------------
# OCR backend registry
# ----------------------------------------------------------------------
OCR_BACKENDS = [
    "Auto (best available)",
    "Surya (best Devanagari)",
    "PaddleStructureV3 (layout + tables)",
    "PaddleOCR Devanagari",
    "EasyOCR",
    "Tesseract",
]
DEFAULT_BACKEND = "Auto (best available)"

AUTO_ORDER = ["surya", "paddlestructurev3", "paddleocr", "easyocr", "tesseract"]

LANG_HINTS = {
    "eng":     ["en"],
    "nep":     ["ne"],
    "eng+nep": ["en", "ne"],
}


def wants_nepali(lang_str):
    return "ne" in LANG_HINTS.get(lang_str, ["en"])


# ----------------------------------------------------------------------
# Export types: label -> (method, extension, filename suffix)
# ----------------------------------------------------------------------
EXPORT_TYPES = {
    "Ground truth v2 JSONL (benchmark)": ("export_gt_v2_jsonl",      ".jsonl", "_gt"),
    "Rich page JSON (synthetic-style)":  ("export_rich_json",        ".json",  "_rich"),
    "Rich page JSONL (per-page)":        ("export_rich_jsonl",       ".jsonl", "_rich"),
    "Annotations JSONL (per-page)":      ("export_jsonl",            ".jsonl", "_annotations"),
    "Annotations JSON (per-page)":       ("export_annotations_json", ".json",  "_annotations"),
    "Training JSONL (ERNIEKit)":         ("export_training_jsonl",   ".jsonl", "_train"),
    "PaddleOCR-VL JSON":                 ("export_paddle_json",      ".json",  "_paddle"),
    "PaddleOCR-VL JSONL (per-page)":     ("export_paddle_jsonl",     ".jsonl", "_paddle"),
    "CSV (detections)":                  ("export_csv",              ".csv",   "_detections"),
    "JSON (simple)":                     ("export_json",             ".json",  "_simple"),
    "Searchable PDF":                    ("export_searchable_pdf",   ".pdf",   "_searchable"),
    "CSV (extracted fields)":            ("export_fields_csv",       ".csv",   "_fields"),
}
DEFAULT_EXPORT_TYPE = "Annotations JSONL (per-page)"

# ----------------------------------------------------------------------
# Block labels (custom set)
# ----------------------------------------------------------------------
DEFAULT_LABEL = "paragraph"

PADDLE_LABELS = [
    "title", "heading", "subheading", "paragraph", "text_line", "list_item",
    "label", "header", "footer", "page_number",
    "table", "table_heading", "table_footer",
    "form", "form_field", "checkbox", "radio_button",
    "handwriting", "signature", "handwritten_annotation",
    "handwritten_correction", "handwritten_number",
    "logo", "stamp", "seal", "letterhead", "watermark",
    "image", "photo", "diagram", "chart", "map", "illustration",
    "crossed_out", "redacted", "blurred", "unclear", "mixed_text",
]

LEGACY_LABEL_MAP = {
    "text": "paragraph", "content": "paragraph", "abstract": "paragraph",
    "doc_title": "title", "paragraph_title": "heading",
    "table_title": "table_heading", "figure_title": "label",
    "vision_footnote": "table_footer",
    "table_body": "table", "table_cell": "table",
    "number": "page_number", "figure": "image",
    "header_image": "logo", "footer_image": "image",
    "vertical_text": "text_line", "reference_content": "list_item",
}

EMPTY_CONTENT_LABELS = {
    "image", "photo", "diagram", "chart", "map", "illustration",
    "logo", "seal", "signature", "redacted",
}
TEXT_REQUIRED_LABELS = {
    "title", "heading", "subheading", "paragraph", "text_line", "list_item",
    "label", "header", "footer", "page_number",
    "table_heading", "table_footer", "form_field",
    "handwriting", "handwritten_annotation", "handwritten_correction",
    "handwritten_number", "mixed_text",
}
NON_FLOW_LABELS = {
    "header", "footer", "page_number", "letterhead", "watermark",
    "logo", "stamp", "seal", "signature", "handwritten_annotation",
    "image", "photo", "diagram", "chart", "map", "illustration",
    "crossed_out", "redacted",
}


def canonical_label(label):
    label = (label or DEFAULT_LABEL).strip()
    return LEGACY_LABEL_MAP.get(label, label)


# BGR colors
LABEL_COLORS = {
    "title": (0, 140, 255), "heading": (0, 200, 255), "subheading": (0, 230, 230),
    "paragraph": (200, 200, 200), "text_line": (170, 170, 170),
    "list_item": (180, 200, 140), "label": (140, 180, 220),
    "header": (255, 120, 100), "footer": (255, 120, 100),
    "page_number": (100, 100, 255),
    "table": (255, 100, 200), "table_heading": (255, 140, 220),
    "table_footer": (220, 160, 240),
    "form": (120, 200, 0), "form_field": (80, 220, 120),
    "checkbox": (40, 180, 40), "radio_button": (40, 160, 90),
    "handwriting": (200, 0, 200), "signature": (160, 0, 255),
    "handwritten_annotation": (230, 80, 230),
    "handwritten_correction": (130, 0, 200), "handwritten_number": (255, 90, 200),
    "logo": (0, 120, 200), "stamp": (60, 60, 230), "seal": (0, 0, 255),
    "letterhead": (0, 90, 160), "watermark": (190, 190, 120),
    "image": (0, 200, 100), "photo": (0, 170, 80), "diagram": (200, 200, 0),
    "chart": (230, 180, 60), "map": (160, 140, 40), "illustration": (60, 220, 160),
    "crossed_out": (0, 0, 160), "redacted": (40, 40, 40), "blurred": (150, 150, 150),
    "unclear": (0, 255, 255), "mixed_text": (255, 255, 0),
}

DEFAULT_MODEL_SETTINGS = {
    "use_doc_preprocessor": False,
    "use_layout_detection": True,
    "use_chart_recognition": False,
    "use_seal_recognition": False,
    "use_ocr_for_image_block": False,
    "format_block_content": False,
    "merge_layout_blocks": True,
    "markdown_ignore_labels": [
        "number", "footnote", "header", "header_image",
        "footer", "footer_image", "aside_text",
    ],
    "return_layout_polygon_points": True,
}

TEXT_FIX_PROMPT = (
    "You are correcting OCR output that may contain Nepali (Devanagari) "
    "and/or English text. Fix spelling, spacing, and garbled characters "
    "caused by OCR mistakes, but preserve the original language, meaning, "
    "and line breaks as closely as possible. Do not translate. Return ONLY "
    "the corrected text with no explanation, preamble, or markdown.\n\n"
    "OCR text:\n{text}"
)

BATCH_FIX_PROMPT = (
    "Correct OCR errors in these Nepali (Devanagari) and/or English strings. "
    "Fix spelling, spacing and garbled characters only; do not translate or "
    "rephrase. If a string already looks correct, return it unchanged. "
    "Input is a JSON object mapping ids to strings. Return ONLY a JSON object "
    "with exactly the same ids mapping to the corrected strings.\n\n{payload}"
)

VISION_READ_PROMPT = (
    "This image is a cropped region from a scanned document. It may contain "
    "Nepali (Devanagari script) and/or English text that standard OCR "
    "handles poorly. Carefully read and transcribe the text exactly as it "
    "appears, preserving the original script and line breaks. Do not "
    "translate or summarize. Return ONLY the transcribed text, no "
    "explanation, preamble, or markdown."
)

DEFAULT_FIELD_PATTERNS = [
    ("Invoice / Policy #", r"(?:invoice|policy)\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Za-z0-9\-\/]+)"),
    ("Date", r"\b(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})\b"),
    ("Amount", r"(?:rs\.?|npr|usd|\$)\s?([\d,]+(?:\.\d{1,2})?)"),
    ("Email", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    ("Phone", r"\b(?:\+?\d{1,3}[-.\s]?)?\d{9,10}\b"),
]

# ----------------------------------------------------------------------
# Font discovery
# ----------------------------------------------------------------------
FONT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ocr_tool", "fonts")

NOTO_DEVANAGARI_URLS = [
    "https://cdn.jsdelivr.net/gh/notofonts/notofonts.github.io@main/"
    "fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
    "https://github.com/notofonts/notofonts.github.io/raw/main/"
    "fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
]

DEVANAGARI_FONT_PATTERNS = [
    "*Nirmala*", "*nirmala*", "*Mangal*", "*mangal*",
    "*NotoSansDevanagari*", "*NotoSerifDevanagari*",
    "*Devanagari*", "*devanagari*",
    "*Lohit*", "*Kohinoor*", "*Sangam*",
    "*Mukta*", "*Kalimati*", "*Gargi*",
    "*Annapurna*", "*Himalaya*", "*FreeSans*",
]

UNICODE_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "/usr/share/fonts/truetype/Gargi/Gargi.ttf",
    "/usr/share/fonts/truetype/samanata/Samanata.ttf",
    "/usr/share/fonts/truetype/mukti/Mukti-Regular.ttf",
    "/usr/share/fonts/truetype/sahadeva/Sahadeva.ttf",
    "/usr/share/fonts/truetype/sarai/Sarai.ttf",
    r"C:\Windows\Fonts\Nirmala.ttf",
    r"C:\Windows\Fonts\NirmalaUI.ttf",
    r"C:\Windows\Fonts\nirmala.ttf",
    r"C:\Windows\Fonts\mangal.ttf",
    r"C:\Windows\Fonts\Mangal.ttf",
    "/System/Library/Fonts/Supplemental/DevanagariSangamMN.ttc",
    "/System/Library/Fonts/Supplemental/KohinoorDevanagari.ttc",
    "/Library/Fonts/KohinoorDevanagari.ttc",
]


def _system_font_dirs():
    system = platform.system()
    dirs = []
    if system == "Windows":
        win = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(win, "Fonts"))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    elif system == "Darwin":
        dirs += ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
                 "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts",
                 os.path.expanduser("~/.fonts"),
                 os.path.expanduser("~/.local/share/fonts")]
    return [d for d in dirs if os.path.isdir(d)]


def _open_font(path, size):
    if not path or not os.path.isfile(path):
        return None
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        pass
    for idx in range(1, 4):
        try:
            return ImageFont.truetype(path, size, index=idx)
        except Exception:
            continue
    return None


def _font_renders_devanagari(path, size=20):
    font = _open_font(path, size)
    if font is None:
        return False
    try:
        b1 = font.getbbox("\u0928")
        b2 = font.getbbox("\uE000")
        if b1 is None or b2 is None:
            return False
        if b1 == b2:
            return False
        return (b1[2] - b1[0]) > 0 and (b1[3] - b1[1]) > 0
    except Exception:
        return False


def _download_devanagari_font():
    try:
        os.makedirs(FONT_CACHE_DIR, exist_ok=True)
    except Exception:
        return None
    target = os.path.join(FONT_CACHE_DIR, "NotoSansDevanagari-Regular.ttf")
    if os.path.isfile(target) and os.path.getsize(target) > 10_000:
        if _font_renders_devanagari(target):
            return target
    for url in NOTO_DEVANAGARI_URLS:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = r.read()
            if len(data) < 10_000:
                continue
            with open(target, "wb") as f:
                f.write(data)
            if _font_renders_devanagari(target):
                return target
        except Exception:
            continue
    return None


def find_devanagari_font():
    if FORCE_DEVANAGARI_FONT and os.path.isfile(FORCE_DEVANAGARI_FONT):
        if _font_renders_devanagari(FORCE_DEVANAGARI_FONT):
            return FORCE_DEVANAGARI_FONT
        print(f"[ocr-tool] FORCE_DEVANAGARI_FONT={FORCE_DEVANAGARI_FONT} "
              f"failed Devanagari glyph check; falling back.")
    for p in UNICODE_FONT_CANDIDATES:
        if os.path.isfile(p) and _font_renders_devanagari(p):
            return p
    for d in _system_font_dirs():
        for pat in DEVANAGARI_FONT_PATTERNS:
            for path in glob.glob(os.path.join(d, "**", pat), recursive=True):
                if path.lower().endswith((".ttf", ".ttc", ".otf")):
                    if _font_renders_devanagari(path):
                        return path
    return _download_devanagari_font()


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def conf_color(conf):
    if conf < 50:
        return (0, 0, 220)
    elif conf < 80:
        return (0, 200, 220)
    return (0, 200, 0)


def label_color(label):
    return LABEL_COLORS.get(label, (200, 200, 200))


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


def xyxy_to_xywh(bbox):
    x0, y0, x1, y1 = bbox
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def quad_from_bbox(bbox):
    """4-point polygon (clockwise from top-left) from a (x, y, w, h) bbox."""
    x, y, w, h = bbox
    return [[float(x), float(y)], [float(x + w), float(y)],
            [float(x + w), float(y + h)], [float(x), float(y + h)]]


def bbox_from_quad(quad):
    """(x, y, w, h) bbox enclosing a 4-point polygon."""
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")


def _fix_double_escaped(s):
    """Turn literal '\\u0915' text (double-escaped JSON, pasted escapes) into real characters."""
    if not isinstance(s, str) or ("\\u" not in s and "\\U" not in s):
        return s

    def sub(m):
        try:
            return chr(int(m.group(1) or m.group(2), 16))
        except ValueError:
            return m.group(0)
    return _ESCAPE_RE.sub(sub, s)


# Tk font families that can draw Devanagari, best first.
TK_DEVANAGARI_FAMILIES = [
    "Nirmala UI", "Noto Sans Devanagari", "Noto Sans Devanagari UI",
    "Noto Serif Devanagari", "Mangal", "Lohit Devanagari", "Kalimati",
    "Mukta", "Gargi", "Annapurna SIL", "Kohinoor Devanagari",
    "Devanagari Sangam MN", "FreeSans",
]


def _strip_html_tags(html_str):
    if not html_str:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html_str)
    text = re.sub(r"</t[dh]>", "\t", text)
    text = re.sub(r"</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _pad_white(img, pad):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _looks_like_html(text):
    if not text or not isinstance(text, str):
        return False
    return bool(re.search(r"</?\w+[^>]*>", text))


def normalize_text(s):
    """Unicode NFC so hand-typed, OCR and LLM text compare byte-for-byte."""
    if not isinstance(s, str):
        return s
    return unicodedata.normalize("NFC", s)


# ----------------------------------------------------------------------
# Ground-truth cleaning + QA (applied on every save)
# ----------------------------------------------------------------------
_NUKTA_RE = re.compile("[\u093c\u0958-\u095f]")
_ELLIPSIS_IN_LEADER_RE = re.compile("[.\u2026]*\u2026[.\u2026]*")
_ZW_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


def _expand_leader(m):
    s = m.group(0)
    return "." * (s.count(".") + 3 * s.count("\u2026"))


def clean_block_content(text, label):
    """Safe, loss-free fixes for ground truth. Returns (text, label, fixes)."""
    fixes = []
    label = canonical_label(label)
    text = normalize_text(_fix_double_escaped(text or ""))
    if text.lstrip().lower().startswith("<table") and label != "table":
        fixes.append(f"label '{label}' -> 'table' (content is table HTML)")
        label = "table"
    if label == "table":
        if text.strip():
            fixed = repair_table_html(text)
            if fixed != text:
                fixes.append("table HTML normalized")
            text = fixed
        return text, label, fixes
    if label in EMPTY_CONTENT_LABELS:
        if text.strip():
            fixes.append(f"content removed ({label} blocks have no text)")
        return "", label, fixes
    if "\t" in text:
        text = re.sub(r"[ \t]*\t[ \t]*", " ", text)
        fixes.append("TAB -> space")
    if "\u2026" in text:
        new = _ELLIPSIS_IN_LEADER_RE.sub(
            lambda m: _expand_leader(m) if "." in m.group(0) or len(m.group(0)) > 1 else m.group(0),
            text)
        if new != text:
            fixes.append("'\u2026' in dot leader -> dots")
            text = new
    stripped = "\n".join(line.rstrip() for line in text.strip().split("\n"))
    return stripped, label, fixes


def qa_block_warnings(d):
    w = []
    t = d.get("text") or ""
    label = canonical_label(d.get("label"))
    if label == "table":
        for p in validate_table_html(t)[:3]:
            w.append(f"table: {p}")
        return w
    nk = _NUKTA_RE.findall(t)
    if nk:
        w.append(f"nukta {''.join(sorted(set(nk)))} - rare in Nepali, confirm it is printed")
    if "\u2026" in t:
        w.append("contains '\u2026' - use '...' unless an ellipsis is really printed")
    if _ZW_RE.search(t):
        w.append("zero-width character (ZWJ/ZWNJ) - keep only if the half-form is visible")
    if label in TEXT_REQUIRED_LABELS and not t.strip():
        w.append(f"empty {label} block")
    if re.search(r"[0-9]", t) and re.search("[\u0966-\u096f]", t):
        w.append("mixes 0-9 and \u0966-\u096f - confirm each digit is as printed")
    return w


def _box_xyxy(d):
    x, y, w, h = d["bbox"]
    return x, y, x + w, y + h


def covered_fraction(inner, outer):
    ax0, ay0, ax1, ay1 = _box_xyxy(inner)
    bx0, by0, bx1, by1 = _box_xyxy(outer)
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return iw * ih / area


def is_empty_table(d):
    t = d.get("text") or ""
    if canonical_label(d.get("label")) != "table" and not t.lstrip().lower().startswith(("<table", "<html")):
        return False
    return not re.sub(r"<[^>]+>|&nbsp;|\s", "", t)


LEFTOVER_COVER = 0.6


def find_leftover_boxes(dets):
    reviewed = [d for d in dets if d.get("status") == "reviewed"]
    out = []
    for i, d in enumerate(dets):
        if d.get("status") == "reviewed":
            continue
        if is_empty_table(d) or any(covered_fraction(d, r) >= LEFTOVER_COVER for r in reviewed):
            out.append(i)
    return out


def qa_page_warnings(dets):
    out = []
    flow = [d for d in dets if canonical_label(d.get("label")) not in NON_FLOW_LABELS]
    orders = sorted(d["order"] for d in flow if isinstance(d.get("order"), int))
    if orders != list(range(1, len(orders) + 1)):
        out.append(f"reading order is not 1..{len(orders)} without gaps: {orders[:15]}")
    missing = sum(1 for d in flow if not isinstance(d.get("order"), int))
    if missing:
        out.append(f"{missing} content block(s) without an order (use Auto Order)")
    unrev = sum(1 for d in dets if d.get("status") != "reviewed")
    if unrev:
        out.append(f"{unrev} block(s) not marked reviewed")
    for i in find_leftover_boxes(dets):
        out.append(f"#{i} ({dets[i].get('label')}): leftover OCR box "
                   f"(empty table or under a reviewed box) - delete it")
    seen = {}
    for i, d in enumerate(dets):
        t = re.sub(r"\s+", " ", (d.get("text") or "")).strip()
        if len(t) >= 8:
            if t in seen:
                out.append(f"#{i} has the same text as #{seen[t]} - copy-paste? check the image")
            else:
                seen[t] = i
    for i, d in enumerate(dets):
        for w in qa_block_warnings(d):
            out.append(f"#{i} (order {d.get('order')}, {d.get('label')}): {w}")
    return out


def _order_key(d, idx=0):
    o = d.get("order")
    return (o if isinstance(o, (int, float)) else 10**9, idx)


def scale_detections(dets, factor):
    out = []
    for d in dets:
        nd = dict(d)
        x, y, w, h = d["bbox"]
        nd["bbox"] = (int(round(x * factor)), int(round(y * factor)),
                      max(1, int(round(w * factor))), max(1, int(round(h * factor))))
        if isinstance(d.get("quad"), list) and len(d["quad"]) == 4:
            nd["quad"] = [[float(p[0]) * factor, float(p[1]) * factor] for p in d["quad"]]
        out.append(nd)
    return out


def reading_order(dets):
    items = sorted(dets, key=lambda d: (d["bbox"][1], d["bbox"][0]))
    bands = []
    for d in items:
        x, y, w, h = d["bbox"]
        if bands:
            b = bands[-1]
            overlap = min(b["y1"], y + h) - max(b["y0"], y)
            if overlap >= 0.5 * min(h, b["h"]):
                b["items"].append(d)
                b["y1"] = max(b["y1"], y + h)
                continue
        bands.append({"y0": y, "y1": y + h, "h": h, "items": [d]})
    out = []
    for b in bands:
        out.extend(sorted(b["items"], key=lambda d: d["bbox"][0]))
    return out


# ----------------------------------------------------------------------
# Rule-based Nepali OCR fixes (OPT-IN)
# ----------------------------------------------------------------------
NEPALI_OCR_FIXES = [
    (r"गाउपालिका", "गाउँपालिका"),
    (r"गाउं", "गाउँ"),
    (r"ठाउं", "ठाउँ"),
    (r"बाउं", "बाउँ"),
    (r"प्रमाणीत", "प्रमाणित"),
    (r"\bदजा\b", "दर्जा"),
    (r"\bकर\.", "क्र."),
    (r"कार्यलय", "कार्यालय"),
    (r"कर्मचारिहरु", "कर्मचारीहरु"),
    (r"[ \t]{2,}", " "),
]

_NEPALI_FIX_RE = None


def fix_nepali_ocr(text):
    global _NEPALI_FIX_RE
    if not text:
        return text
    if _NEPALI_FIX_RE is None:
        _NEPALI_FIX_RE = [(re.compile(p), r) for p, r in NEPALI_OCR_FIXES]
    for rx, repl in _NEPALI_FIX_RE:
        text = rx.sub(repl, text)
    return text


# ----------------------------------------------------------------------
# Table HTML: normalize, repair, validate, and cell-level parsing
# ----------------------------------------------------------------------
_TABLE_TAGS = ("table", "thead", "tbody", "tfoot", "tr", "th", "td")


def normalize_table_html(html):
    if not html:
        return html
    return re.sub(r">\s+<", "><", html.strip())


def repair_table_html(html):
    if not html:
        return html
    h = normalize_table_html(html)
    h = re.sub(r"</?(?:html|body)\b[^>]*>", "", h).strip()
    h = re.sub(r"(<(?:table|thead|tbody|tfoot)\b[^>]*>)(<t[hd]\b)", r"\1<tr>\2", h)
    h = re.sub(r"(</tr>)(<t[hd]\b)", r"\1<tr>\2", h)
    return h


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []
        self.rows = []
        self.cur_row = None

    @staticmethod
    def _span(attrs, name):
        for k, v in attrs:
            if k == name:
                try:
                    return max(1, int(v))
                except (TypeError, ValueError):
                    return 1
        return 1

    def handle_starttag(self, tag, attrs):
        if tag not in _TABLE_TAGS:
            return
        if tag == "tr":
            if self.cur_row is not None and "tr" in self.stack:
                self.errors.append(f"row {len(self.rows)} is not closed before a new <tr>")
                while self.stack and self.stack[-1] != "tr":
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()
            self.cur_row = []
            self.rows.append(self.cur_row)
        elif tag in ("th", "td"):
            if self.cur_row is None:
                self.errors.append(f"<{tag}> outside a <tr> (row {len(self.rows) + 1})")
                self.cur_row = []
                self.rows.append(self.cur_row)
            self.cur_row.append((self._span(attrs, "rowspan"), self._span(attrs, "colspan")))
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag not in _TABLE_TAGS:
            return
        if tag == "tr":
            self.cur_row = None
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.errors.append(f"unclosed <{self.stack.pop()}>")
            self.stack.pop()
        else:
            self.errors.append(f"stray </{tag}>")


def _row_widths(rows):
    carry = {}
    widths = []
    for cells in rows:
        occupied = set(carry)
        new = {}
        col = 0
        for rs, cs in cells:
            while col in occupied:
                col += 1
            for k in range(cs):
                occupied.add(col + k)
                if rs > 1:
                    new[col + k] = rs - 1
            col += cs
        widths.append(max(occupied) + 1 if occupied else 0)
        carry = {c: n - 1 for c, n in carry.items() if n - 1 > 0}
        carry.update(new)
    return widths


def validate_table_html(html):
    p = _TableParser()
    try:
        p.feed(html or "")
        p.close()
    except Exception as e:
        return [f"parse error: {e}"]
    errors = list(p.errors)
    for t in reversed(p.stack):
        errors.append(f"unclosed <{t}>")
    widths = _row_widths(p.rows)
    if not widths:
        errors.append("table has no rows")
        return errors
    expected = Counter(widths).most_common(1)[0][0]
    bad = [(i + 1, w) for i, w in enumerate(widths) if w != expected]
    for r, w in bad[:5]:
        errors.append(f"row {r}: {w} columns (expected {expected})")
    if len(bad) > 5:
        errors.append(f"... {len(bad) - 5} more rows with the wrong width")
    return errors


# ----------------------------------------------------------------------
# Rich table parsing: <table> HTML -> structured cells
# ----------------------------------------------------------------------
class _RichTableParser(HTMLParser):
    """Walk a <table> and yield per-cell dicts including rowspan/colspan/text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.row_index = -1
        self.col_cursor = 0
        self.occupied = set()
        self.cells = []
        self._cur = None
        self._buf = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.row_index += 1
            self.col_cursor = 0
        elif tag in ("td", "th"):
            if self.row_index < 0:
                self.row_index = 0
            try:
                rs = max(1, int(a.get("rowspan", 1) or 1))
            except (TypeError, ValueError):
                rs = 1
            try:
                cs = max(1, int(a.get("colspan", 1) or 1))
            except (TypeError, ValueError):
                cs = 1
            while (self.row_index, self.col_cursor) in self.occupied:
                self.col_cursor += 1
            self._cur = {
                "row": self.row_index,
                "col": self.col_cursor,
                "rowspan": rs,
                "colspan": cs,
                "is_header": tag == "th",
            }
            for r in range(rs):
                for c in range(cs):
                    self.occupied.add((self.row_index + r, self.col_cursor + c))
            self.col_cursor += cs
            self._buf = []
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            text = "".join(self._buf).strip()
            self._cur["text"] = text
            self._cur["empty"] = not text
            self._cur["cell_type"] = ("header" if self._cur["is_header"] else "data")
            self.cells.append(self._cur)
            self._cur = None
            self._in_cell = False

    def handle_data(self, data):
        if self._in_cell:
            self._buf.append(data)


def parse_table_cells(html):
    """Return (num_rows, num_cols, header_rows, cells) from <table> HTML."""
    if not html:
        return 0, 0, 0, []
    p = _RichTableParser()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    if not p.cells:
        return 0, 0, 0, []
    max_row = max(c["row"] + c["rowspan"] - 1 for c in p.cells)
    max_col = max(c["col"] + c["colspan"] - 1 for c in p.cells)
    num_rows = max_row + 1
    num_cols = max_col + 1
    header_rows = 0
    for c in p.cells:
        if c["is_header"] and c["row"] == header_rows:
            header_rows = c["row"] + 1
        elif not c["is_header"]:
            break
    return num_rows, num_cols, header_rows, p.cells


def guess_col_types(cells, num_cols, header_rows):
    """Return per-column type hints from cell contents (serial, date, money, ...)."""
    types = ["text"] * num_cols
    for col in range(num_cols):
        samples = [c.get("text", "") for c in cells
                   if c["col"] == col and not c["is_header"] and c.get("text")]
        if not samples:
            continue
        joined = " ".join(samples)
        if all(re.fullmatch(r"[\u0966-\u096f0-9]+", s or "") for s in samples):
            types[col] = "serial" if col == 0 else "number"
        elif any(re.search(r"[\u0966-\u096f0-9]{4}[\-/.][\u0966-\u096f0-9]", s) for s in samples):
            types[col] = "date"
        elif any(re.search(r"(?:रु\.|rs\.?|npr|\$)", s, re.I) for s in samples):
            types[col] = "money"
        elif all(re.fullmatch(r"[\u0966-\u096f0-9+\-() ]{6,}", s or "") for s in samples):
            types[col] = "phone"
        elif sum(1 for s in samples if len(s) > 20) >= len(samples) / 2:
            types[col] = "address"
        elif col == 0 and all(len(s) <= 4 for s in samples):
            types[col] = "serial"
    return types


# ----------------------------------------------------------------------
# Annotation file loading (all supported formats)
# ----------------------------------------------------------------------
def _read_json_any(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        items = []
        for ln, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{os.path.basename(path)} line {ln}: {e}") from e
        return items


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _xyxy_from_record(rec):
    bb = rec.get("block_bbox")
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return [float(v) for v in bb]
    bb = rec.get("block_bbox_xywh")
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        x, y, w, h = [float(v) for v in bb]
        return [x, y, x + w, y + h]
    bb = rec.get("coordinate")
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return [float(v) for v in bb]
    bb = rec.get("bbox")
    if isinstance(bb, dict):
        try:
            x, y = float(bb["x"]), float(bb["y"])
            w, h = float(bb["width"]), float(bb["height"])
            return [x, y, x + w, y + h]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        x, y, w, h = [float(v) for v in bb]
        return [x, y, x + w, y + h]
    return None


def _quad_from_record(rec):
    for key in ("quad", "block_polygon_points", "polygon", "polygon_points"):
        q = rec.get(key)
        if isinstance(q, (list, tuple)) and len(q) == 4:
            try:
                return [[float(p[0]), float(p[1])] for p in q]
            except (TypeError, IndexError):
                continue
    return None


def _make_det(rec, xyxy, conf):
    x0, y0, x1, y1 = xyxy
    content = rec.get("block_content")
    if content is None:
        content = rec.get("text", "")
    conf_n = _num(conf)
    d = {
        "text": normalize_text(_fix_double_escaped(content or "")),
        "conf": int(round(conf_n)) if conf_n is not None else 100,
        "bbox": (int(round(x0)), int(round(y0)),
                 max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))),
        "label": canonical_label(rec.get("block_label") or rec.get("label")),
        "order": rec.get("block_order", rec.get("order")),
        "status": rec.get("status") or "imported",
    }
    q = _quad_from_record(rec)
    if q is not None:
        d["quad"] = q
    for k in ("font_id", "font_size", "is_heading", "table_index", "cell_id",
              "line_in_block", "block_id", "reading_order", "group_id",
              "is_underlined", "is_bold", "is_italic"):
        if rec.get(k) is not None:
            d[k] = rec[k]
    if rec.get("ocr_text") is not None:
        d["ocr_text"] = normalize_text(rec["ocr_text"])
    return d


def parse_annotation_items(items):
    """
    Turn loaded JSON items into blocks:
        {"page": 0-based, "image": str|None, "width", "height", "zoom", "det": {...}}
    Supports: annotations JSON/JSONL (per-box), PaddleOCR-VL JSON/JSONL,
    simple JSON, rich synthetic-style page JSON with lines/tables sections,
    and {"res": {...}} wrappers. Returns (blocks, kinds).
    """
    blocks, kinds = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("res"), dict):
            item = item["res"]

        # Rich synthetic-style page record
        if "parsing_res_list" in item or "layout_det_res" in item or \
                ("layout" in item and isinstance(item.get("layout"), list)
                 and ("blocks" in item or "lines" in item)
                 and "schema_version" not in item):
            kinds.add("rich")
            page = _num(item.get("page_index"))
            if page is None:
                page = _num(item.get("page_number"))
            meta = {
                "page": (int(page) - 1 if page is not None and page >= 1
                         else (int(page) if page is not None else 0)),
                "image": item.get("input_path") or item.get("image"),
                "width": _num(item.get("width")),
                "height": _num(item.get("height")),
                "zoom": _num(item.get("render_zoom")),
                "rich_meta": {
                    "v": item.get("v"),
                    "document_id": item.get("document_id"),
                    "language": item.get("language"),
                    "digit_style": item.get("digit_style"),
                    "text": item.get("text"),
                    "fonts": item.get("fonts"),
                    "font_ids": item.get("font_ids"),
                    "quality": item.get("quality"),
                },
            }
            plist = item.get("parsing_res_list") or item.get("layout") or []
            if not plist and isinstance(item.get("blocks"), list):
                plist = item["blocks"]
            score_by = {}
            layout = item.get("layout_det_res") or {}
            for lb in (layout.get("boxes") or []):
                c = lb.get("coordinate") or []
                if len(c) == 4:
                    s = _num(lb.get("score"))
                    score_by[tuple(int(v) for v in c)] = s if s is not None else 1.0
            for b in plist:
                if not isinstance(b, dict):
                    continue
                xyxy = _xyxy_from_record(b)
                if not xyxy:
                    continue
                score = score_by.get(tuple(int(v) for v in xyxy), 1.0)
                blocks.append(dict(meta, det=_make_det(b, xyxy, score * 100)))

            # Standalone lines (from rich "lines[]" section)
            for ln in (item.get("lines") or []):
                if not isinstance(ln, dict):
                    continue
                xyxy = _xyxy_from_record(ln)
                if not xyxy:
                    continue
                rec = {
                    "block_label": "text_line",
                    "block_content": ln.get("text", ""),
                    "block_order": ln.get("line_in_block"),
                    "font_id": ln.get("font_id"),
                    "font_size": ln.get("font_size"),
                    "is_heading": ln.get("is_heading"),
                    "is_underlined": ln.get("is_underlined"),
                    "table_index": ln.get("table_index"),
                    "cell_id": ln.get("cell_id"),
                    "quad": ln.get("quad"),
                }
                blocks.append(dict(meta, det=_make_det(rec, xyxy, 100)))

            # Standalone tables (from rich "tables[]" section)
            for tbl in (item.get("tables") or []):
                if not isinstance(tbl, dict):
                    continue
                html = tbl.get("html")
                tb = tbl.get("bbox")
                if not html or not isinstance(tb, (list, tuple)) or len(tb) != 4:
                    continue
                x0, y0, x1, y1 = [float(v) for v in tb]
                rec = {
                    "block_label": "table",
                    "block_content": html,
                    "block_order": None,
                    "quad": tbl.get("quad"),
                    "table_index": tbl.get("table_index"),
                }
                blocks.append(dict(meta, det=_make_det(rec, [x0, y0, x1, y1], 100)))

        elif isinstance(item.get("blocks"), list):
            kinds.add("annotations")
            page = _num(item.get("page"))
            meta = {
                "page": int(page) - 1 if page is not None and page >= 1 else 0,
                "image": item.get("image"),
                "width": _num(item.get("width")),
                "height": _num(item.get("height")),
                "zoom": _num(item.get("render_zoom")),
                "source_pdf": item.get("source_pdf"),
                "source_image": item.get("source_image"),
                "page_count": _num(item.get("page_count")),
            }
            for b in item["blocks"]:
                if not isinstance(b, dict):
                    continue
                xyxy = _xyxy_from_record(b)
                if not xyxy:
                    continue
                blocks.append(dict(meta, det=_make_det(b, xyxy, b.get("confidence", 100))))

        elif "block_bbox" in item or "block_bbox_xywh" in item:
            kinds.add("annotations")
            page = _num(item.get("page"))
            xyxy = _xyxy_from_record(item)
            if not xyxy:
                continue
            blocks.append({
                "page": int(page) - 1 if page is not None and page >= 1 else 0,
                "image": item.get("image"),
                "width": _num(item.get("width")),
                "height": _num(item.get("height")),
                "zoom": _num(item.get("render_zoom")),
                "det": _make_det(item, xyxy, item.get("confidence", 100)),
            })

        elif isinstance(item.get("bbox"), dict):
            kinds.add("simple")
            xyxy = _xyxy_from_record(item)
            if not xyxy:
                continue
            page = _num(item.get("page"))
            blocks.append({
                "page": int(page) - 1 if page is not None and page >= 1 else 0,
                "image": item.get("image"),
                "width": None, "height": None, "zoom": None,
                "det": _make_det(item, xyxy, item.get("confidence", 100)),
            })
    return blocks, kinds


def load_annotation_blocks(path):
    return parse_annotation_items(_read_json_any(path))


def file_sig(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def split_pages_dir(pdf_path):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(pdf_path)), stem + "_pages")


def split_page_stem(pdf_path, page):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return f"{stem}_page{page + 1:03d}"


SPLIT_MANIFEST = "manifest.json"

_SIDECAR_SUFFIXES = ("_annotations", "_paddle", "_simple", "_rich",
                     "_detections", "_train", "_gt")


def find_sidecar_annotations(src_path):
    d = os.path.dirname(os.path.abspath(src_path))
    stem = os.path.splitext(os.path.basename(src_path))[0]
    dirs = [d, os.path.join(d, "annotations")]
    preferred = [stem + s for s in (
        "_annotations.jsonl", "_annotations.json",
        "_rich.json", "_rich.jsonl",
        "_paddle.jsonl", "_paddle.json",
        ".jsonl", ".json", "_simple.json")]
    out = []
    for dd in dirs:
        for n in preferred:
            p = os.path.join(dd, n)
            if os.path.isfile(p) and p not in out:
                out.append(p)
    extra = []
    for dd in dirs:
        if not os.path.isdir(dd):
            continue
        for p in glob.glob(os.path.join(dd, glob.escape(stem) + "*")):
            name = os.path.basename(p)
            rest = name[len(stem):]
            if (p in out or not name.lower().endswith((".json", ".jsonl"))
                    or not rest[:1] in ("_", ".") or "_train" in rest):
                continue
            try:
                if os.path.getsize(p) > 200 * 1024 * 1024:
                    continue
            except OSError:
                continue
            extra.append(p)
    extra.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return out + extra


# ======================================================================
# OCR BACKENDS
# ======================================================================
class OCRBackends:
    """Lazily-initialised OCR backends, cached per language where relevant."""

    def __init__(self):
        self._surya = None
        self._easyocr = None
        self._paddle = {}
        self._paddle_struct = {}
        self._table_only = {}
        self.apply_fixes = False

    def _maybe_fix(self, text, lang_str):
        if self.apply_fixes and wants_nepali(lang_str):
            return fix_nepali_ocr(text)
        return text

    # ---------- Tesseract ----------
    def run_tesseract(self, img, lang_str, min_conf):
        data = pytesseract.image_to_data(
            img, lang=lang_str, config="--oem 3 --psm 6", output_type=Output.DICT,
        )
        detections = []
        order_counter = 0
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            raw_conf = data["conf"][i]
            conf = int(float(raw_conf)) if raw_conf not in ("-1", "") else -1
            if conf >= min_conf and text:
                x, y, w, h = (data["left"][i], data["top"][i],
                              data["width"][i], data["height"][i])
                order_counter += 1
                detections.append({
                    "text": text, "conf": conf, "bbox": (x, y, w, h),
                    "label": "text_line", "order": order_counter,
                    "quad": quad_from_bbox((x, y, w, h)),
                })
        full_text = pytesseract.image_to_string(img, lang=lang_str)
        return detections, full_text

    # ---------- EasyOCR ----------
    def _get_easyocr(self, langs):
        if self._easyocr is None:
            import easyocr
            use_gpu = False
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except Exception:
                pass
            self._easyocr = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
        return self._easyocr

    def run_easyocr(self, img, lang_str, min_conf):
        langs = LANG_HINTS.get(lang_str, ["en"])
        reader = self._get_easyocr(langs)
        results = reader.readtext(img)
        detections = []
        for i, (box, text, conf) in enumerate(results):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0 = int(min(xs)), int(min(ys))
            x1, y1 = int(max(xs)), int(max(ys))
            quad = [[float(p[0]), float(p[1])] for p in box] if len(box) == 4 else \
                   quad_from_bbox((x0, y0, x1 - x0, y1 - y0))
            detections.append({
                "text": text, "conf": int(conf * 100),
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "label": "text_line", "order": i + 1,
                "quad": quad,
            })
        full_text = "\n".join(d["text"] for d in detections)
        return detections, full_text

    # ---------- Surya ----------
    def _get_surya(self):
        if self._surya is None:
            try:
                from surya.recognition import RecognitionPredictor
                from surya.detection import DetectionPredictor
                try:
                    from surya.foundation import FoundationPredictor
                    foundation = FoundationPredictor()
                    rec = RecognitionPredictor(foundation)
                except Exception:
                    rec = RecognitionPredictor()
                det = DetectionPredictor()
                self._surya = ("new", rec, det)
            except Exception:
                try:
                    from surya.ocr import run_ocr
                    from surya.model.detection.segformer import (
                        load_model as load_det_model,
                        load_processor as load_det_processor,
                    )
                    from surya.model.recognition.model import load_model as load_rec_model
                    from surya.model.recognition.processor import load_processor as load_rec_processor
                    det_model = load_det_model()
                    det_proc = load_det_processor()
                    rec_model = load_rec_model()
                    rec_proc = load_rec_processor()
                    self._surya = ("old", run_ocr, det_model, det_proc,
                                   rec_model, rec_proc)
                except Exception as e:
                    raise RuntimeError(
                        "Could not import Surya. Install: pip install surya-ocr"
                    ) from e
        return self._surya

    @staticmethod
    def _surya_predict(rec, det, pil, lang_str):
        try:
            params = set(inspect.signature(rec.__call__).parameters)
        except (TypeError, ValueError):
            params = set()
        langs = LANG_HINTS.get(lang_str, ["en"])
        extra = {"langs": [langs]} if "langs" in params else {}
        for kw in ("det_predictor", "detection_predictor"):
            if kw in params:
                return rec([pil], **{kw: det}, **extra)
        det_res = det([pil])
        found = (getattr(det_res[0], "bboxes", None) or []) if det_res else []
        if not found:
            return []
        if "bboxes" in params:
            boxes = [[[int(v) for v in b.bbox] for b in found]]
            return rec([pil], bboxes=boxes, **extra)
        if "polygons" in params:
            polys = [[[[int(v) for v in pt] for pt in b.polygon] for b in found]]
            return rec([pil], polygons=polys, **extra)
        return rec([pil], [langs], det)

    def run_surya(self, img, lang_str, min_conf):
        kind = self._get_surya()
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if kind[0] == "new":
            _, rec, det = kind
            preds = self._surya_predict(rec, det, pil, lang_str)
        else:
            _, run_ocr, det_model, det_proc, rec_model, rec_proc = kind
            langs = LANG_HINTS.get(lang_str, ["en"])
            preds = run_ocr([pil], [langs], det_model, det_proc,
                            rec_model, rec_proc)
        detections = []
        order = 0
        for pred in (preds or []):
            lines = getattr(pred, "text_lines", None) or []
            for line in lines:
                text = getattr(line, "text", "") or ""
                conf = getattr(line, "confidence", 0.0) or 0.0
                bbox = getattr(line, "bbox", None)
                if not text.strip() or not bbox:
                    continue
                c = int(conf * 100) if conf <= 1 else int(conf)
                if c < min_conf:
                    continue
                x0, y0, x1, y1 = [int(v) for v in bbox]
                order += 1
                detections.append({
                    "text": self._maybe_fix(text.strip(), lang_str),
                    "conf": c,
                    "bbox": (x0, y0, x1 - x0, y1 - y0),
                    "label": "text_line", "order": order,
                    "quad": quad_from_bbox((x0, y0, x1 - x0, y1 - y0)),
                })
        full_text = "\n".join(d["text"] for d in detections)
        return detections, full_text

    # ---------- PaddleOCR Devanagari (basic) ----------
    def _get_paddle(self, lang_str):
        key = "nep" if wants_nepali(lang_str) else "en"
        if key not in self._paddle:
            from paddleocr import PaddleOCR
            base = dict(use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)
            attempts = []
            if key == "nep":
                attempts.append(dict(base, text_recognition_model_name=DEVANAGARI_REC_MODEL))
                attempts.append(dict(base, lang="ne", ocr_version="PP-OCRv5"))
                attempts.append(dict(lang="ne"))
            else:
                attempts.append(dict(base, lang="en", ocr_version="PP-OCRv5"))
                attempts.append(dict(lang="en"))
            last_err = None
            for kwargs in attempts:
                try:
                    self._paddle[key] = PaddleOCR(**kwargs)
                    break
                except (TypeError, ValueError) as e:
                    last_err = e
                    continue
            if key not in self._paddle:
                raise RuntimeError(f"Could not initialise PaddleOCR: {last_err}")
        return self._paddle[key]

    @staticmethod
    def _parse_paddle_results(results, min_conf):
        detections = []
        order = 0
        for res in results:
            texts = res.get("rec_texts") or res.get("rec_text") or []
            scores = res.get("rec_scores") or res.get("rec_score") or []
            polys = res.get("rec_polys")
            if polys is None or len(polys) == 0:
                polys = res.get("dt_polys") or []
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                score = scores[i] if i < len(scores) else 1.0
                conf = int(score * 100) if score <= 1 else int(score)
                if conf < min_conf:
                    continue
                poly = polys[i] if i < len(polys) else None
                if poly is None:
                    continue
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                x0, y0 = int(min(xs)), int(min(ys))
                x1, y1 = int(max(xs)), int(max(ys))
                order += 1
                quad = [[float(p[0]), float(p[1])] for p in poly] if len(poly) == 4 else \
                       quad_from_bbox((x0, y0, x1 - x0, y1 - y0))
                detections.append({
                    "text": text.strip(), "conf": conf,
                    "bbox": (x0, y0, x1 - x0, y1 - y0),
                    "label": "text", "order": order,
                    "quad": quad,
                })
        return detections

    def run_paddle(self, img, lang_str, min_conf):
        ocr = self._get_paddle(lang_str)
        if hasattr(ocr, "predict"):
            detections = self._parse_paddle_results(ocr.predict(img), min_conf)
            for d in detections:
                d["text"] = self._maybe_fix(d["text"], lang_str)
                d["label"] = "text_line"
            full_text = "\n".join(d["text"] for d in detections)
            return detections, full_text
        results = ocr.ocr(img, cls=False)
        detections = []
        order = 0
        for line in (results or []):
            if not line:
                continue
            for entry in line:
                box, (text, score) = entry[0], entry[1]
                conf = int(score * 100) if score <= 1 else int(score)
                if conf < min_conf or not text.strip():
                    continue
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                x0, y0 = int(min(xs)), int(min(ys))
                x1, y1 = int(max(xs)), int(max(ys))
                order += 1
                detections.append({
                    "text": self._maybe_fix(text.strip(), lang_str), "conf": conf,
                    "bbox": (x0, y0, x1 - x0, y1 - y0),
                    "label": "text_line", "order": order,
                    "quad": [[float(p[0]), float(p[1])] for p in box] if len(box) == 4 else
                            quad_from_bbox((x0, y0, x1 - x0, y1 - y0)),
                })
        full_text = "\n".join(d["text"] for d in detections)
        return detections, full_text

    # ---------- Fast local re-read of a crop (no LLM) ----------
    def reread_crop(self, crop_bgr, lang_str):
        ocr = self._get_paddle(lang_str)
        crop = _pad_white(crop_bgr, 8)
        if hasattr(ocr, "predict"):
            dets = self._parse_paddle_results(ocr.predict(crop), 0)
        else:
            dets, _ = self.run_paddle(crop, lang_str, 0)
        if not dets:
            return ""
        heights = sorted(d["bbox"][3] for d in dets)
        tol = max(4, int(heights[len(heights) // 2] * 0.5))
        dets.sort(key=lambda d: d["bbox"][1] + d["bbox"][3] / 2)
        lines = []
        for d in dets:
            cy = d["bbox"][1] + d["bbox"][3] / 2
            if lines and abs(cy - lines[-1][0]) <= tol:
                lines[-1][1].append(d)
            else:
                lines.append([cy, [d]])
        out = []
        for _, items in lines:
            items.sort(key=lambda d: d["bbox"][0])
            out.append(" ".join(d["text"] for d in items))
        return normalize_text(self._maybe_fix("\n".join(out), lang_str))

    # ---------- PaddleStructureV3 ----------
    def _get_paddle_structure(self, lang_str):
        key = "nep" if wants_nepali(lang_str) else "en"
        if key not in self._paddle_struct:
            from paddleocr import PPStructureV3
            base = dict(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_table_recognition=True,
                use_formula_recognition=True,
                use_seal_recognition=False,
                use_chart_recognition=False,
                format_block_content=True,
            )
            if key == "nep":
                try:
                    self._paddle_struct[key] = PPStructureV3(
                        **base, text_recognition_model_name=DEVANAGARI_REC_MODEL)
                except (TypeError, ValueError) as e:
                    print(f"[ocr-tool] PPStructureV3 rejected Devanagari rec "
                          f"model ({e}); falling back to default recognizer.")
                    self._paddle_struct[key] = PPStructureV3(**base)
            else:
                self._paddle_struct[key] = PPStructureV3(**base)
        return self._paddle_struct[key]

    def run_paddle_structure(self, img, lang_str, min_conf):
        pipeline = self._get_paddle_structure(lang_str)
        results = pipeline.predict(img)

        detections = []
        order = 0
        for res in results:
            parsing = res.get("parsing_res_list") or []
            layout = res.get("layout_det_res") or {}
            layout_boxes = layout.get("boxes") or []

            score_by_bbox = {}
            for lb in layout_boxes:
                coord = lb.get("coordinate")
                if coord is not None and len(coord) == 4:
                    key = tuple(int(v) for v in coord)
                    score_by_bbox[key] = float(lb.get("score", 1.0))

            for block in parsing:
                if isinstance(block, dict):
                    get = block.get
                else:
                    get = lambda k, default=None, b=block: getattr(
                        b, k.replace("block_", ""), getattr(b, k, default))
                label = get("block_label", "text") or "text"
                content = get("block_content", "") or ""
                bbox = get("block_bbox")
                if bbox is None:
                    continue
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                x0, y0, x1, y1 = [int(v) for v in bbox]
                x, y, w, h = x0, y0, x1 - x0, y1 - y0

                if _looks_like_html(content):
                    if label not in ("table", "table_body", "table_cell",
                                     "table_title"):
                        label = "table"
                if label in ("table_body", "table_cell"):
                    label = "table"

                if label != "table" and not _looks_like_html(content):
                    content = self._maybe_fix(content, lang_str)

                score = score_by_bbox.get((x0, y0, x1, y1), 1.0)
                conf = int(round(score * 100))

                order += 1
                if label in ("image", "figure", "chart") and not content.strip():
                    content = ""
                detections.append({
                    "text": content, "conf": conf,
                    "bbox": (x, y, w, h),
                    "label": label, "order": order,
                    "quad": quad_from_bbox((x, y, w, h)),
                })

        text_parts = []
        for d in detections:
            if d["label"] in ("text", "doc_title", "paragraph_title",
                              "table_title", "formula", "header", "footer"):
                if d["text"].strip():
                    text_parts.append(d["text"])
            elif d["label"] == "table" or _looks_like_html(d.get("text", "")):
                text_parts.append(_strip_html_tags(d["text"]))
            elif d["label"] in ("image", "figure", "chart", "seal"):
                text_parts.append(f"[{d['label']}]")

        full_text = "\n\n".join(text_parts)
        return detections, full_text

    # ---------- Table-only pipeline ----------
    def _get_table_only_pipeline(self, lang_str):
        key = "nep" if wants_nepali(lang_str) else "en"
        if key not in self._table_only:
            from paddleocr import TableRecognitionPipelineV2
            base = dict(use_doc_orientation_classify=False,
                        use_doc_unwarping=False)
            if key == "nep":
                try:
                    self._table_only[key] = TableRecognitionPipelineV2(
                        **base, text_recognition_model_name=DEVANAGARI_REC_MODEL)
                except (TypeError, ValueError) as e:
                    print(f"[ocr-tool] Table pipeline rejected Devanagari rec "
                          f"model ({e}); using default recognizer.")
                    self._table_only[key] = TableRecognitionPipelineV2(**base)
            else:
                self._table_only[key] = TableRecognitionPipelineV2(**base)
        return self._table_only[key]

    def run_table_only(self, crop_bgr, lang_str="eng+nep"):
        pipeline = self._get_table_only_pipeline(lang_str)
        crop = _pad_white(crop_bgr, 16)
        try:
            results = pipeline.predict(crop, use_layout_detection=False)
        except TypeError:
            results = pipeline.predict(crop)
        for res in results:
            tables = res.get("table_res_list") or []
            for t in tables:
                if isinstance(t, dict):
                    html = t.get("pred_html") or t.get("html") or ""
                else:
                    html = getattr(t, "pred_html", "") or ""
                if html and html.strip():
                    return html
            for k in ("pred_html", "html", "table_html"):
                v = res.get(k) if hasattr(res, "get") else None
                if v:
                    return v
        return None

    # ---------- Reconstruct tables from line detections ----------
    def reconstruct_tables(self, detections):
        text_like = {"text", "text_line", "paragraph"}
        items = [d for d in detections if d.get("label") in text_like
                 and d["text"].strip()]
        if len(items) < 4:
            return detections

        heights = sorted(d["bbox"][3] for d in items)
        med_h = heights[len(heights) // 2]
        row_tol = max(6, int(med_h * 0.6))

        items_sorted = sorted(items, key=lambda d: d["bbox"][1] + d["bbox"][3] / 2)
        rows = []
        for d in items_sorted:
            cy = d["bbox"][1] + d["bbox"][3] / 2
            if rows and abs(cy - rows[-1]["cy"]) <= row_tol:
                rows[-1]["items"].append(d)
                n = len(rows[-1]["items"])
                rows[-1]["cy"] = (rows[-1]["cy"] * (n - 1) + cy) / n
            else:
                rows.append({"cy": cy, "items": [d]})

        x_starts = sorted(d["bbox"][0] for d in items)
        col_tol = max(10, int(med_h * 1.2))
        col_anchors = []
        for x in x_starts:
            if not col_anchors or x - col_anchors[-1] > col_tol:
                col_anchors.append(x)
            else:
                n = sum(1 for d in items
                        if abs(d["bbox"][0] - col_anchors[-1]) <= col_tol)
                col_anchors[-1] = (col_anchors[-1] * (n - 1) + x) / n
        if len(col_anchors) < 2:
            return detections

        def col_index(d):
            x = d["bbox"][0]
            return min(range(len(col_anchors)),
                       key=lambda i: abs(col_anchors[i] - x))

        tabular_rows = []
        for r in rows:
            cols_hit = {col_index(d) for d in r["items"]}
            if len(cols_hit) >= 2 or (len(rows) >= 4 and len(r["items"]) >= 1
                                      and len(col_anchors) >= 3):
                tabular_rows.append(r)

        if len(tabular_rows) < 3 or len(col_anchors) < 2:
            return detections

        for r in tabular_rows:
            r["items"].sort(key=lambda d: d["bbox"][0])

        html_rows = []
        for r in tabular_rows:
            cells_by_col = {}
            for d in r["items"]:
                ci = col_index(d)
                cells_by_col.setdefault(ci, []).append(d["text"])
            row_cells = []
            for ci in range(len(col_anchors)):
                row_cells.append(" ".join(cells_by_col.get(ci, [])))
            html_rows.append("<tr>" + "".join(
                f"<td>{c}</td>" for c in row_cells) + "</tr>")
        html = "<table>" + "".join(html_rows) + "</table>"

        all_cells = [c for r in tabular_rows for c in r["items"]]
        xs0 = min(c["bbox"][0] for c in all_cells)
        ys0 = min(c["bbox"][1] for c in all_cells)
        xs1 = max(c["bbox"][0] + c["bbox"][2] for c in all_cells)
        ys1 = max(c["bbox"][1] + c["bbox"][3] for c in all_cells)
        table_bbox = (xs0, ys0, xs1 - xs0, ys1 - ys0)
        consumed = {id(c) for c in all_cells}
        remaining = [d for d in detections if id(d) not in consumed]
        remaining.append({
            "text": html, "conf": 100, "bbox": table_bbox,
            "label": "table", "order": None,
            "quad": quad_from_bbox(table_bbox),
        })
        return remaining


# ======================================================================
# OLLAMA CLIENT
# ======================================================================
class OllamaClient:
    def __init__(self, base_url=OLLAMA_URL):
        self.base_url = base_url

    def is_available(self):
        if not REQUESTS_AVAILABLE:
            return False
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def _post(self, payload, timeout):
        payload.setdefault("stream", False)
        payload.setdefault("keep_alive", OLLAMA_KEEP_ALIVE)
        payload.setdefault("options", dict(OLLAMA_OPTIONS))
        r = requests.post(f"{self.base_url}/api/generate", json=payload,
                          timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "")

    def generate_text(self, model, prompt, timeout=90):
        start = time.time()
        out = self._post({"model": model, "prompt": prompt}, timeout)
        return out.strip(), time.time() - start

    def generate_json(self, model, prompt, timeout=180):
        start = time.time()
        out = self._post({"model": model, "prompt": prompt, "format": "json"},
                         timeout)
        try:
            data = json.loads(out or "{}")
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", out or "", re.S)
            data = json.loads(m.group(0)) if m else {}
        if not isinstance(data, dict):
            data = {}
        return data, time.time() - start

    def generate_vision(self, model, prompt, image_bgr, timeout=180):
        h, w = image_bgr.shape[:2]
        s = 1024 / max(h, w)
        if s < 1:
            image_bgr = cv2.resize(image_bgr, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image crop.")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        start = time.time()
        out = self._post({"model": model, "prompt": prompt, "images": [b64]},
                         timeout)
        return out.strip(), time.time() - start


# ======================================================================
# MAIN APP
# ======================================================================
class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Tool — Images & PDFs + Multi-Backend Devanagari OCR")
        self.root.geometry("1380x900")
        self.root.minsize(1000, 700)

        self.image_path = None
        self.original_image = None
        self.raw_page_image = None
        self.annotated_image = None
        self.tk_preview = None
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)

        self.pdf_doc = None
        self.current_page = 0
        self.page_count = 1

        self.detections = []
        self.pdf_zoom = 2.0
        self._coord_scale = 1.0

        self.selected_bbox = None
        self.drag_start = None
        self.selected_detection_index = None

        self.interaction_mode = None
        self.active_handle = None
        self.resize_orig_bbox = None
        self.move_orig_bbox = None
        self.move_start_img = None
        self._edit_snap = None
        self.HANDLE_HIT_TOL = 7

        self.page_ocr_cache = {}
        self.lines_by_page = {}          # page -> list of per-line dicts
        self._history = {}
        self._dirty = False
        self._annotation_path = None
        self._annotation_kind = None
        self._split_meta = None
        self._last_autofix = 0
        self._disk_sig = {}
        self._baseline = {}
        self._conflict_seen = {}
        self._last_ext_check = 0.0
        self._reloaded_external = 0
        self._pending_underline = False   # for the "new box" case

        # Extended document metadata (synthetic-schema compatible).
        self.doc_meta = {
            "v": 2,
            "document_id": "",
            "language": "ne",
            "digit_style": "dev",
        }
        self.rich_page_meta = {}

        self._label_font_cache = {}
        self._unicode_font_ok = None
        self._devanagari_font_path = None
        self._font_discovery_done = False

        self.backends = OCRBackends()

        self.ollama = OllamaClient()
        self.ollama_available = False
        self.ollama_models = []

        self._build_ui()
        self._bind_shortcuts()
        self._setup_drag_drop()
        self._check_ollama_async()
        self.root.after(200, self._warn_if_no_devanagari_font)
        self.root.after(250, self._setup_tk_fonts)

    # ==================================================================
    # UI
    # ==================================================================
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open File", command=self.open_file).pack(side=tk.LEFT, padx=3)
        self.run_btn = ttk.Button(toolbar, text="Run OCR", command=self.run_ocr, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=3)
        self.run_all_btn = ttk.Button(toolbar, text="OCR All Pages",
                                      command=self.run_all_pages, state=tk.DISABLED)
        self.run_all_btn.pack(side=tk.LEFT, padx=3)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.undo_btn = ttk.Button(toolbar, text="↶ Undo", command=self.undo, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=2)
        self.redo_btn = ttk.Button(toolbar, text="↷ Redo", command=self.redo, state=tk.DISABLED)
        self.redo_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save (Ctrl+S)", command=self.quick_save).pack(side=tk.LEFT, padx=3)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar, text="Save Annotated", command=self.save_annotated).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Copy Text", command=self.copy_text).pack(side=tk.LEFT, padx=3)

        export_btn = ttk.Menubutton(toolbar, text="Export ▾")
        export_btn.pack(side=tk.LEFT, padx=3)
        export_menu = tk.Menu(export_btn, tearoff=0)
        export_btn["menu"] = export_menu
        export_menu.add_command(label="Ground truth v2 JSONL (benchmark)",
                                command=self.export_gt_v2_jsonl)
        export_menu.add_command(label="Rich page JSON (synthetic-style)",
                                command=self.export_rich_json)
        export_menu.add_command(label="Rich page JSONL (per-page)",
                                command=self.export_rich_jsonl)
        export_menu.add_separator()
        export_menu.add_command(label="Annotations JSONL (per-page)",
                                command=self.export_jsonl)
        export_menu.add_command(label="Annotations JSON (per-page)",
                                command=self.export_annotations_json)
        export_menu.add_command(label="Training JSONL (ERNIEKit)",
                                command=self.export_training_jsonl)
        export_menu.add_separator()
        export_menu.add_command(label="PaddleOCR-VL JSON", command=self.export_paddle_json)
        export_menu.add_command(label="PaddleOCR-VL JSONL (per-page)",
                                command=self.export_paddle_jsonl)
        export_menu.add_separator()
        export_menu.add_command(label="Detections as CSV", command=self.export_csv)
        export_menu.add_command(label="Detections as JSON (simple)", command=self.export_json)
        export_menu.add_command(label="Searchable PDF", command=self.export_searchable_pdf)
        export_menu.add_command(label="Extracted Fields as CSV", command=self.export_fields_csv)

        import_btn = ttk.Menubutton(toolbar, text="Import ▾")
        import_btn.pack(side=tk.LEFT, padx=3)
        import_menu = tk.Menu(import_btn, tearoff=0)
        import_btn["menu"] = import_menu
        import_menu.add_command(label="Annotations / PaddleOCR-VL / Rich JSON (JSON or JSONL)...",
                                command=self.import_annotations)
        import_menu.add_command(label="Reload sidecar annotations",
                                command=self.reload_sidecar)
        import_menu.add_command(label="Reload current page from its JSON (Ctrl+R)",
                                command=self.reload_current_page)

        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=3)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(toolbar, text="Min Conf:").pack(side=tk.LEFT)
        self.conf_var = tk.IntVar(value=60)
        ttk.Spinbox(toolbar, from_=0, to=100, textvariable=self.conf_var,
                    width=5).pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Lang:").pack(side=tk.LEFT, padx=(8, 2))
        self.lang_var = tk.StringVar(value="eng+nep")
        ttk.Combobox(toolbar, textvariable=self.lang_var, width=8, state="readonly",
                     values=["eng", "nep", "eng+nep", "fra", "deu", "spa", "ita", "por",
                             "nld", "rus", "chi_sim", "chi_tra", "jpn", "kor"]
                     ).pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Backend:").pack(side=tk.LEFT, padx=(10, 2))
        self.backend_var = tk.StringVar(value=DEFAULT_BACKEND)
        self.backend_combo = ttk.Combobox(
            toolbar, textvariable=self.backend_var, width=30,
            state="readonly", values=OCR_BACKENDS,
        )
        self.backend_combo.pack(side=tk.LEFT)

        self.show_boxes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Show Boxes", variable=self.show_boxes_var,
                        command=self.refresh_preview).pack(side=tk.LEFT, padx=8)

        prep = ttk.Frame(self.root, padding=(6, 2))
        prep.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(prep, text="Preprocess:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))

        self.pp_grayscale = tk.BooleanVar(value=False)
        ttk.Checkbutton(prep, text="Grayscale+Threshold", variable=self.pp_grayscale,
                        command=self._on_preprocess_toggled).pack(side=tk.LEFT, padx=4)
        self.pp_denoise = tk.BooleanVar(value=False)
        ttk.Checkbutton(prep, text="Denoise", variable=self.pp_denoise,
                        command=self._on_preprocess_toggled).pack(side=tk.LEFT, padx=4)
        self.pp_deskew = tk.BooleanVar(value=False)
        ttk.Checkbutton(prep, text="Deskew", variable=self.pp_deskew,
                        command=self._on_preprocess_toggled).pack(side=tk.LEFT, padx=4)
        self.pp_upscale = tk.BooleanVar(value=False)
        ttk.Checkbutton(prep, text="Upscale 2x", variable=self.pp_upscale,
                        command=self._on_preprocess_toggled).pack(side=tk.LEFT, padx=4)

        ttk.Separator(prep, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.apply_fixes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(prep, text="Nepali auto-fix (OCR only)",
                        variable=self.apply_fixes_var).pack(side=tk.LEFT, padx=4)
        self.autoload_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prep, text="Auto-load annotations",
                        variable=self.autoload_var).pack(side=tk.LEFT, padx=4)
        self.split_pdf_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prep, text="Page folders (PDF + images)",
                        variable=self.split_pdf_var).pack(side=tk.LEFT, padx=4)
        self.keep_ocr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prep, text="Keep original OCR text",
                        variable=self.keep_ocr_var).pack(side=tk.LEFT, padx=4)
        self.auto_lines_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(prep, text="Auto line detect after OCR",
                        variable=self.auto_lines_var).pack(side=tk.LEFT, padx=4)
        self.auto_underline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(prep, text="Auto underline detect",
                        variable=self.auto_underline_var).pack(side=tk.LEFT, padx=4)

        ttk.Button(prep, text="Font...", command=self._pick_font).pack(side=tk.RIGHT, padx=6)
        self.font_lbl = ttk.Label(prep, text="", foreground="#666")
        self.font_lbl.pack(side=tk.RIGHT, padx=6)
        self.table_sel_btn = ttk.Button(prep, text="Table OCR on selection",
                                        command=self._run_table_on_selection)
        self.table_sel_btn.pack(side=tk.RIGHT, padx=4)
        ttk.Button(prep, text="Re-read selection",
                   command=self._run_paddle_reread).pack(side=tk.RIGHT, padx=4)
        ttk.Button(prep, text="Check Page",
                   command=self.validate_tables).pack(side=tk.RIGHT, padx=4)
        ttk.Button(prep, text="Detect Underlines",
                   command=self.detect_underlines_page).pack(side=tk.RIGHT, padx=4)
        ttk.Button(prep, text="Detect Lines",
                   command=self.run_line_ocr_page).pack(side=tk.RIGHT, padx=4)
        ttk.Button(prep, text="Toggle Underline (Ctrl+U)",
                   command=self._toggle_underline).pack(side=tk.RIGHT, padx=4)
        ttk.Button(prep, text="Auto Order",
                   command=self.auto_order).pack(side=tk.RIGHT, padx=4)

        hint = ttk.Frame(self.root, padding=(6, 0))
        hint.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(hint, foreground="#666",
                  text=("Drag: new box · Shift+drag: nested box · Ctrl+Z / Ctrl+Y: undo / redo · "
                        "Ctrl+S: save · Ctrl+A: select all text · Ctrl+U: toggle underline · "
                        "Del: delete box · Esc: deselect")
                  ).pack(side=tk.LEFT)

        exp = ttk.Frame(self.root, padding=(6, 2))
        exp.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(exp, text="Export:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))

        self.export_type_var = tk.StringVar(value=DEFAULT_EXPORT_TYPE)
        self.export_type_combo = ttk.Combobox(
            exp, textvariable=self.export_type_var, width=34, state="readonly",
            values=list(EXPORT_TYPES.keys()),
        )
        self.export_type_combo.pack(side=tk.LEFT)
        self.export_type_combo.bind("<<ComboboxSelected>>", self._on_export_type_changed)

        ttk.Label(exp, text="Save to:").pack(side=tk.LEFT, padx=(10, 4))
        self.export_path_var = tk.StringVar()
        self.export_path_entry = ttk.Entry(exp, textvariable=self.export_path_var)
        self.export_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.export_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(exp, text="Write page PNGs",
                        variable=self.export_images_var).pack(side=tk.LEFT, padx=4)
        ttk.Button(exp, text="Browse...", command=self._browse_export_path).pack(side=tk.LEFT, padx=2)
        self.export_now_btn = ttk.Button(exp, text="Export", command=self._do_export_from_ui)
        self.export_now_btn.pack(side=tk.LEFT, padx=2)

        pager = ttk.Frame(self.root, padding=(6, 2))
        pager.pack(side=tk.TOP, fill=tk.X)

        self.prev_btn = ttk.Button(pager, text="< Prev", command=self.prev_page, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=3)
        self.page_lbl = ttk.Label(pager, text="Page 1 / 1", width=14, anchor="center")
        self.page_lbl.pack(side=tk.LEFT, padx=3)
        self.next_btn = ttk.Button(pager, text="Next >", command=self.next_page, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=3)

        ttk.Label(pager, text="Render Zoom:").pack(side=tk.LEFT, padx=(20, 4))
        self.zoom_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(pager, from_=1.0, to=4.0, increment=0.5,
                    textvariable=self.zoom_var, width=5,
                    command=self._apply_zoom).pack(side=tk.LEFT)

        self.ollama_status_lbl = ttk.Label(pager, text="Ollama: checking...", foreground="#888")
        self.ollama_status_lbl.pack(side=tk.RIGHT, padx=6)
        self.anno_lbl = ttk.Label(pager, text="", foreground="#2a7")
        self.anno_lbl.pack(side=tk.RIGHT, padx=12)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = ttk.Frame(main)
        main.add(left, weight=4)

        sel_frame = ttk.Frame(left, padding=(4, 4))
        sel_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(sel_frame, text="Label:").pack(side=tk.LEFT)
        self.sel_label_var = tk.StringVar(value=DEFAULT_LABEL)
        self.sel_label_combo = ttk.Combobox(
            sel_frame, textvariable=self.sel_label_var, width=24, height=20,
            state="readonly", values=PADDLE_LABELS,
        )
        self.sel_label_combo.pack(side=tk.LEFT, padx=(2, 6))
        self.sel_label_combo.bind("<<ComboboxSelected>>", lambda e: self._on_label_changed())

        ttk.Label(sel_frame, text="Order:").pack(side=tk.LEFT)
        self.sel_order_var = tk.StringVar(value="")
        ttk.Entry(sel_frame, textvariable=self.sel_order_var, width=5).pack(side=tk.LEFT, padx=(2, 6))

        ttk.Label(sel_frame, text="Text:").pack(side=tk.LEFT)
        self.sel_text_var = tk.StringVar()
        self.sel_entry = ttk.Entry(sel_frame, textvariable=self.sel_text_var)
        self.sel_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.sel_entry.bind("<Return>", lambda e: self._save_selected_text())

        self.sel_save_btn = ttk.Button(sel_frame, text="Save", command=self._save_selected_text,
                                       state=tk.DISABLED)
        self.sel_save_btn.pack(side=tk.LEFT, padx=2)
        self.sel_review_btn = ttk.Button(sel_frame, text="Mark Reviewed",
                                         command=self._mark_reviewed, state=tk.DISABLED)
        self.sel_review_btn.pack(side=tk.LEFT, padx=2)
        self.sel_delete_btn = ttk.Button(sel_frame, text="Delete Box", command=self._delete_selected,
                                         state=tk.DISABLED)
        self.sel_delete_btn.pack(side=tk.LEFT, padx=2)

        self.sel_review_all_btn = ttk.Button(
            sel_frame, text="Mark All Page Reviewed",
            command=self._mark_all_reviewed)
        self.sel_review_all_btn.pack(side=tk.LEFT, padx=2)

        preview_row = ttk.Frame(left)
        preview_row.pack(side=tk.BOTTOM, fill=tk.X, padx=4)
        ttk.Label(preview_row, text="Preview:", foreground="#888").pack(side=tk.LEFT, anchor="n")
        self.preview_lbl = tk.Label(preview_row, bg="white", anchor="w", justify="left",
                                    relief=tk.SOLID, borderwidth=1)
        self.preview_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self._preview_photo = None
        self.sel_text_var.trace_add("write", lambda *_: self._schedule_text_preview())

        info_row = ttk.Frame(left)
        info_row.pack(side=tk.BOTTOM, fill=tk.X, padx=4)
        self.sel_info_lbl = ttk.Label(info_row, text="No selection.", foreground="#888")
        self.sel_info_lbl.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(left, bg="#1e1e1e", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Delete>", lambda e: self._delete_selected())
        self.canvas.bind("<BackSpace>", lambda e: self._delete_selected())
        self.canvas.bind("<Escape>", lambda e: self._clear_selection())

        right = ttk.Frame(main)
        main.add(right, weight=3)
        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)

        self._build_info_tab(nb)
        self._build_text_tab(nb)
        self._build_detections_tab(nb)
        self._build_fields_tab(nb)
        self._build_ai_tab(nb)

        self.status = tk.StringVar(value="Ready. Open an image, PDF or annotation JSON to begin.")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN,
                  anchor="w", padding=4).pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.bind("<Configure>", lambda e: self.refresh_preview())
        self._set_default_export_path_for_current_type()

    def _build_info_tab(self, nb):
        size_tab = ttk.Frame(nb, padding=10)
        nb.add(size_tab, text="Image Info")
        self.size_labels = {}
        for key, label in [
            ("file", "File"), ("type", "Type"), ("page", "Page"),
            ("resolution", "Resolution"), ("width", "Width (px)"),
            ("height", "Height (px)"), ("channels", "Channels"),
            ("size_kb", "File Size (KB)"), ("annotations", "Annotations"),
        ]:
            row = ttk.Frame(size_tab)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label + ":", width=14, anchor="w",
                      font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            val = ttk.Label(row, text="-", anchor="w")
            val.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.size_labels[key] = val

        ttk.Separator(size_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 6))
        ttk.Label(size_tab, text="Document metadata (rich schema):",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w")
        meta_row = ttk.Frame(size_tab)
        meta_row.pack(fill=tk.X, pady=4)
        ttk.Label(meta_row, text="document_id:", width=13).pack(side=tk.LEFT)
        self.doc_id_var = tk.StringVar(value="")
        ttk.Entry(meta_row, textvariable=self.doc_id_var, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Label(meta_row, text="language:", width=9).pack(side=tk.LEFT)
        self.doc_lang_var = tk.StringVar(value="ne")
        ttk.Combobox(meta_row, textvariable=self.doc_lang_var, width=6,
                     values=["ne", "en", "ne+en", "hi", "sa"],
                     state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Label(meta_row, text="digit_style:", width=11).pack(side=tk.LEFT)
        self.doc_digit_var = tk.StringVar(value="dev")
        ttk.Combobox(meta_row, textvariable=self.doc_digit_var, width=6,
                     values=["dev", "latn", "mixed"],
                     state="readonly").pack(side=tk.LEFT, padx=4)

    def _build_text_tab(self, nb):
        text_tab = ttk.Frame(nb, padding=6)
        nb.add(text_tab, text="Extracted Text")
        self.text_box = tk.Text(text_tab, wrap="word", font="TkTextFont",
                                undo=True, maxundo=-1, autoseparators=True)
        self.text_box.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb = ttk.Scrollbar(text_tab, command=self.text_box.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.config(yscrollcommand=sb.set)

    def _build_detections_tab(self, nb):
        det_tab = ttk.Frame(nb, padding=6)
        nb.add(det_tab, text="Detections")

        ttk.Label(det_tab,
                  text="Only rows with status 'reviewed' are exported to Training JSONL. "
                       "style flags: B=bold, I=italic, U=underlined.",
                  foreground="#666").pack(side=tk.BOTTOM, anchor="w", pady=(4, 0))

        cols = ("label", "order", "status", "style", "text", "conf", "x", "y", "w", "h")
        self.tree = ttk.Treeview(det_tab, columns=cols, show="headings", height=15)
        widths = {"label": 120, "order": 45, "status": 80, "style": 45,
                  "text": 150, "conf": 45, "x": 45, "y": 45, "w": 45, "h": 45}
        for c in cols:
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=widths[c], anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb2 = ttk.Scrollbar(det_tab, command=self.tree.yview)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.config(yscrollcommand=sb2.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Delete>", lambda e: self._delete_selected())

    def _build_fields_tab(self, nb):
        fields_tab = ttk.Frame(nb, padding=6)
        nb.add(fields_tab, text="Field Extraction")

        top = ttk.Frame(fields_tab)
        top.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(top, text="Field name:").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text="Regex pattern:").grid(row=1, column=0, sticky="w")
        self.field_name_var = tk.StringVar()
        self.field_pattern_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.field_name_var, width=20).grid(row=0, column=1, sticky="we", padx=4, pady=2)
        ttk.Entry(top, textvariable=self.field_pattern_var, width=40).grid(row=1, column=1, sticky="we", padx=4, pady=2)
        top.columnconfigure(1, weight=1)

        btn_row = ttk.Frame(fields_tab)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_row, text="Add Field", command=self._add_field_pattern).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Load Common Fields", command=self._load_default_fields).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Remove Selected", command=self._remove_field_pattern).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Run Extraction", command=self.run_field_extraction).pack(side=tk.LEFT, padx=2)

        self.pattern_tree = ttk.Treeview(fields_tab, columns=("name", "pattern"), show="headings", height=5)
        self.pattern_tree.heading("name", text="FIELD")
        self.pattern_tree.heading("pattern", text="PATTERN")
        self.pattern_tree.column("name", width=140)
        self.pattern_tree.column("pattern", width=280)
        self.pattern_tree.pack(fill=tk.X, pady=(0, 8))

        ttk.Separator(fields_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(fields_tab, text="Extracted values:",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.results_tree = ttk.Treeview(fields_tab, columns=("field", "value"), show="headings", height=10)
        self.results_tree.heading("field", text="FIELD")
        self.results_tree.heading("value", text="VALUE")
        self.results_tree.column("field", width=140)
        self.results_tree.column("value", width=300)
        self.results_tree.pack(fill=tk.BOTH, expand=True)

    def _build_ai_tab(self, nb):
        ai_tab = ttk.Frame(nb, padding=8)
        nb.add(ai_tab, text="AI Assist")

        ttk.Label(
            ai_tab,
            text=("Select a detection row or draw a box, then choose an action.\n"
                  "'Re-read (PaddleOCR)' is local and fast — try it first.\n"
                  "'Fix Text' uses the Ollama text model. 'Re-read (vision LLM)' is slow.\n"
                  "NOTE: small local LLMs frequently mistranslate Nepali, and labels\n"
                  "must match what is printed — always check suggestions against the image."),
            foreground="#555", justify="left", wraplength=460,
        ).pack(fill=tk.X, pady=(0, 8))

        model_row = ttk.Frame(ai_tab)
        model_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(model_row, text="Text model:").grid(row=0, column=0, sticky="w")
        self.text_model_var = tk.StringVar(value=DEFAULT_TEXT_MODEL)
        self.text_model_combo = ttk.Combobox(model_row, textvariable=self.text_model_var, width=22)
        self.text_model_combo.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(model_row, text="Vision model:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.vision_model_var = tk.StringVar(value=DEFAULT_VISION_MODEL)
        self.vision_model_combo = ttk.Combobox(model_row, textvariable=self.vision_model_var, width=22)
        self.vision_model_combo.grid(row=1, column=1, sticky="w", padx=4, pady=(4, 0))

        btn_row = ttk.Frame(ai_tab)
        btn_row.pack(fill=tk.X, pady=(8, 2))
        ttk.Button(btn_row, text="Re-read Region (PaddleOCR, fast)",
                   command=self._run_paddle_reread).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Re-read Region (vision LLM, slow)",
                   command=self._run_ollama_vision).pack(side=tk.LEFT, padx=2)

        btn_row2 = ttk.Frame(ai_tab)
        btn_row2.pack(fill=tk.X, pady=(2, 4))
        ttk.Button(btn_row2, text="Fix Selected Text",
                   command=lambda: self._run_ollama_fix(whole_page=False)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="Fix Whole Page Text",
                   command=lambda: self._run_ollama_fix(whole_page=True)).pack(side=tk.LEFT, padx=2)

        batch_row = ttk.Frame(ai_tab)
        batch_row.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(batch_row, text="Batch fix — only blocks with conf <").pack(side=tk.LEFT, padx=(0, 4))
        self.batch_threshold_var = tk.IntVar(value=70)
        ttk.Spinbox(batch_row, from_=0, to=100, textvariable=self.batch_threshold_var,
                    width=5).pack(side=tk.LEFT, padx=(0, 6))
        self.batch_fix_btn = ttk.Button(
            batch_row, text="AI-fix all low-confidence blocks",
            command=self._run_ollama_batch_fix,
        )
        self.batch_fix_btn.pack(side=tk.LEFT, padx=2)

        self.ai_timing_lbl = ttk.Label(ai_tab, text="", foreground="#888")
        self.ai_timing_lbl.pack(anchor="w")

        ttk.Label(ai_tab, text="Suggestion (editable):",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self.ai_result_box = tk.Text(ai_tab, wrap="word", height=10, font="TkTextFont",
                                     undo=True, maxundo=-1, autoseparators=True)
        self.ai_result_box.pack(fill=tk.BOTH, expand=True)

        accept_row = ttk.Frame(ai_tab)
        accept_row.pack(fill=tk.X, pady=6)
        ttk.Button(accept_row, text="Accept -> Apply to Selected Row",
                   command=self._accept_ai_to_row).pack(side=tk.LEFT, padx=2)
        ttk.Button(accept_row, text="Accept -> Replace Whole Page Text",
                   command=self._accept_ai_to_page).pack(side=tk.LEFT, padx=2)
        ttk.Button(accept_row, text="Discard",
                   command=lambda: self.ai_result_box.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=2)

    # ==================================================================
    # KEYBOARD SHORTCUTS
    # ==================================================================
    def _bind_shortcuts(self):
        r = self.root

        def entry_select_all(event):
            w = event.widget
            try:
                w.select_range(0, tk.END)
                w.icursor(tk.END)
            except tk.TclError:
                pass
            return "break"

        def text_select_all(event):
            w = event.widget
            try:
                w.tag_add(tk.SEL, "1.0", "end-1c")
                w.mark_set(tk.INSERT, "end-1c")
                w.see(tk.INSERT)
            except tk.TclError:
                pass
            return "break"

        for cls in ("Entry", "TEntry", "TCombobox", "TSpinbox", "Spinbox"):
            for seq in ("<Control-a>", "<Control-A>"):
                r.bind_class(cls, seq, entry_select_all)
        for seq in ("<Control-a>", "<Control-A>"):
            r.bind_class("Text", seq, text_select_all)

        r.bind_all("<Control-z>", self._on_undo_key)
        r.bind_all("<Control-y>", self._on_redo_key)
        r.bind_all("<Control-Z>", self._on_redo_key)
        r.bind_all("<Control-s>", self._on_save_key)
        r.bind_all("<Control-S>", self._on_save_key)
        r.bind_all("<Control-o>", lambda e: (self.open_file(), "break")[1])
        r.bind_all("<Control-r>", lambda e: (self.reload_current_page(), "break")[1])
        r.bind_all("<Control-u>", lambda e: (self._toggle_underline(), "break")[1])
        r.bind_all("<Control-U>", lambda e: (self._toggle_underline(), "break")[1])
        r.bind("<FocusIn>", self._check_current_page_external, add="+")

    def _focus_is_text_widget(self):
        try:
            w = self.root.focus_get()
        except (KeyError, tk.TclError):
            return False
        return isinstance(w, tk.Text)

    def _on_undo_key(self, event=None):
        if self._focus_is_text_widget():
            return None
        self.undo()
        return "break"

    def _on_redo_key(self, event=None):
        if self._focus_is_text_widget():
            return None
        self.redo()
        return "break"

    def _on_save_key(self, event=None):
        self.quick_save()
        return "break"

    # ==================================================================
    # UNDERLINE TOGGLE (manual)
    # ==================================================================
    def _toggle_underline(self):
        """Flip is_underlined on the selected detection (or the pending new box)."""
        if self.selected_detection_index is None:
            if self.selected_bbox is not None:
                self._pending_underline = not getattr(self, "_pending_underline", False)
                self.status.set(f"Underline for the next box: "
                                f"{'ON' if self._pending_underline else 'OFF'}")
            else:
                self.status.set("Select a detection or draw a box first.")
            return
        idx = self.selected_detection_index
        self._push_undo()
        d = self.detections[idx]
        d["is_underlined"] = not d.get("is_underlined", False)
        self._sync_tree_row(idx)
        self._update_selection_panel()
        self.refresh_preview()
        self.status.set(f"Detection #{idx}: is_underlined = {d['is_underlined']}")

    # ==================================================================
    # UNDO / REDO
    # ==================================================================
    def _hist(self, page=None):
        p = self.current_page if page is None else page
        return self._history.setdefault(p, {"undo": [], "redo": []})

    def _snapshot(self):
        return {"dets": copy.deepcopy(self.detections), "scale": self._coord_scale}

    def _push_undo(self, snap=None, page=None, mark_dirty=True):
        h = self._hist(page)
        h["undo"].append(snap if snap is not None else self._snapshot())
        if len(h["undo"]) > MAX_UNDO:
            del h["undo"][0]
        h["redo"].clear()
        if mark_dirty:
            self._set_dirty(True)
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        if not hasattr(self, "undo_btn"):
            return
        h = self._history.get(self.current_page, {"undo": [], "redo": []})
        self.undo_btn.config(state=tk.NORMAL if h["undo"] else tk.DISABLED)
        self.redo_btn.config(state=tk.NORMAL if h["redo"] else tk.DISABLED)

    def _restore_snapshot(self, snap):
        dets = snap["dets"]
        sc = snap.get("scale", 1.0)
        if abs(sc - self._coord_scale) > 1e-6:
            dets = scale_detections(dets, self._coord_scale / sc)
        self._set_detections(dets)
        self._set_dirty(True)
        self._update_undo_buttons()

    def undo(self):
        if self.original_image is None:
            return
        h = self._hist()
        if not h["undo"]:
            self.status.set("Nothing to undo.")
            return
        h["redo"].append(self._snapshot())
        self._restore_snapshot(h["undo"].pop())
        self.status.set(f"Undo ({len(h['undo'])} more). Ctrl+Y to redo.")

    def redo(self):
        if self.original_image is None:
            return
        h = self._hist()
        if not h["redo"]:
            self.status.set("Nothing to redo.")
            return
        h["undo"].append(self._snapshot())
        self._restore_snapshot(h["redo"].pop())
        self.status.set(f"Redo ({len(h['redo'])} more).")

    def _set_dirty(self, value):
        self._dirty = bool(value)
        self._update_annotation_label()

    def _update_annotation_label(self):
        if not hasattr(self, "anno_lbl"):
            return
        name = os.path.basename(self._annotation_path) if self._annotation_path else ""
        if self._dirty:
            txt = f"● unsaved changes{(' — ' + name) if name else ''}"
            self.anno_lbl.config(text=txt, foreground="#c60")
        elif name:
            self.anno_lbl.config(text=f"✓ {name}", foreground="#2a7")
        else:
            self.anno_lbl.config(text="")
        if hasattr(self, "size_labels"):
            self.size_labels["annotations"].config(text=self._annotation_path or "-")

    def _set_detections(self, dets):
        self.detections = dets
        self._stash_current_page()
        self._drop_selection(refresh=False)
        self._rebuild_tree()
        self.refresh_preview()

    # ==================================================================
    # INLINE EXPORT ROW
    # ==================================================================
    def _default_export_path(self, key):
        _m, ext, suffix = EXPORT_TYPES.get(key, EXPORT_TYPES[DEFAULT_EXPORT_TYPE])
        if self.image_path:
            base = os.path.splitext(self.image_path)[0]
        else:
            base = os.path.join(os.path.expanduser("~"), "annotations")
        return base + suffix + ext

    def _set_default_export_path_for_current_type(self):
        t = getattr(self, "export_type_var", None)
        key = t.get() if t else DEFAULT_EXPORT_TYPE
        if hasattr(self, "export_path_var"):
            self.export_path_var.set(self._default_export_path(key))

    def _on_export_type_changed(self, event=None):
        key = self.export_type_var.get()
        _m, ext, suffix = EXPORT_TYPES.get(key, EXPORT_TYPES[DEFAULT_EXPORT_TYPE])
        current = self.export_path_var.get().strip()
        if not current:
            self._set_default_export_path_for_current_type()
            return
        base, _old = os.path.splitext(current)
        for _mm, _ee, suf in EXPORT_TYPES.values():
            if suf and base.endswith(suf):
                base = base[:-len(suf)]
                break
        self.export_path_var.set(base + suffix + ext)

    def _browse_export_path(self):
        key = self.export_type_var.get()
        _m, ext, _s = EXPORT_TYPES.get(key, ("", ".json", ""))
        ft_map = {
            ".jsonl": [("JSON Lines", "*.jsonl"), ("All files", "*.*")],
            ".json": [("JSON", "*.json"), ("All files", "*.*")],
            ".csv": [("CSV", "*.csv"), ("All files", "*.*")],
            ".pdf": [("PDF", "*.pdf"), ("All files", "*.*")],
        }
        ft = ft_map.get(ext, [("All files", "*.*")])
        current = self.export_path_var.get().strip()
        init_dir = os.path.dirname(current) if current else os.path.expanduser("~")
        init_file = os.path.basename(current) if current else ""
        path = filedialog.asksaveasfilename(
            defaultextension=ext, filetypes=ft,
            initialdir=init_dir, initialfile=init_file,
        )
        if path:
            self.export_path_var.set(path)

    def _do_export_from_ui(self):
        key = self.export_type_var.get()
        entry = EXPORT_TYPES.get(key)
        if not entry:
            messagebox.showerror("Export", f"Unknown export type: {key!r}")
            return
        method_name, ext, _suffix = entry
        path = self.export_path_var.get().strip()
        if not path:
            self._set_default_export_path_for_current_type()
            path = self.export_path_var.get()
        base, old_ext = os.path.splitext(path)
        if old_ext.lower() != ext.lower():
            path = base + ext
            self.export_path_var.set(path)
        folder = os.path.dirname(path)
        if folder and not os.path.isdir(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Export Error",
                                     f"Could not create folder:\n{folder}\n\n{e}")
                return
        method = getattr(self, method_name, None)
        if method is None:
            messagebox.showerror("Export", f"Missing method: {method_name}")
            return
        self._autosave_pending_new()
        method(path=path)

    # ==================================================================
    # DEVANAGARI FONT HANDLING
    # ==================================================================
    def _set_font_path(self, path, source):
        self._devanagari_font_path = path or ""
        self._label_font_cache.clear()
        if path:
            print(f"[ocr-tool] Label font ({source}): {path}")
            try:
                self.font_lbl.config(text=os.path.basename(path))
            except Exception:
                pass
        else:
            print("[ocr-tool] Label font: NONE — Devanagari will render as boxes.")
            try:
                self.font_lbl.config(text="(no Devanagari font!)")
            except Exception:
                pass

    def _get_label_font(self, size=14):
        if size in self._label_font_cache:
            return self._label_font_cache[size]

        if not self._font_discovery_done:
            path = None
            source = "auto"
            if FORCE_DEVANAGARI_FONT and os.path.isfile(FORCE_DEVANAGARI_FONT) \
                    and _font_renders_devanagari(FORCE_DEVANAGARI_FONT):
                path = FORCE_DEVANAGARI_FONT
                source = "override"
            else:
                path = find_devanagari_font()
            self._set_font_path(path, source)
            self._font_discovery_done = True

        font = None
        if self._devanagari_font_path:
            font = _open_font(self._devanagari_font_path, size)
        if font is None:
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:
                font = ImageFont.load_default()
        self._label_font_cache[size] = font
        return font

    def _pick_font(self):
        path = filedialog.askopenfilename(
            title="Pick a Devanagari-capable font (.ttf/.ttc/.otf)",
            initialdir="/usr/share/fonts/truetype/noto"
                if os.path.isdir("/usr/share/fonts/truetype/noto")
                else "/usr/share/fonts",
            filetypes=[("Fonts", "*.ttf *.ttc *.otf"), ("All files", "*.*")],
        )
        if not path:
            return
        if not _font_renders_devanagari(path):
            messagebox.showwarning(
                "No Devanagari glyphs",
                "That font doesn't seem to contain Devanagari characters.\n\n"
                "On Ubuntu use:\n"
                "  /usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            )
            return
        self._set_font_path(path, "manual picker")
        self._font_discovery_done = True
        self.status.set(f"Using font: {os.path.basename(path)}")
        self.refresh_preview()

    def _warn_if_no_devanagari_font(self):
        if self._unicode_font_ok is not None:
            return
        self._get_label_font(14)
        path = self._devanagari_font_path
        self._unicode_font_ok = bool(path)
        if not path:
            if "Ready" in self.status.get():
                self.status.set(
                    "Warning: no Devanagari-capable font found. "
                    "On Ubuntu run: sudo apt install fonts-noto-devanagari"
                )
        else:
            try:
                from PIL import features
                if not features.check("raqm"):
                    print("[ocr-tool] Note: Pillow built without RaQM — Devanagari "
                          "conjuncts/matras may render slightly wrong. "
                          "Fix: sudo apt install libraqm0 && pip install "
                          "--upgrade --force-reinstall pillow")
            except Exception:
                pass

    # ==================================================================
    # TK FONTS + RENDERED PREVIEW
    # ==================================================================
    def _tk_fontsystem(self):
        try:
            fs = str(self.root.tk.call("::tk::pkgconfig", "get", "fontsystem"))
            if fs in ("xft", "x11"):
                return fs
        except tk.TclError:
            pass
        if platform.system() == "Linux":
            try:
                with open("/proc/self/maps", encoding="utf-8", errors="ignore") as f:
                    maps = f.read()
                if "libtk" in maps.lower():
                    return "xft" if "libXft" in maps else "x11"
            except OSError:
                pass
        return "unknown"

    @staticmethod
    def _font_file_families(path):
        if not path or not os.path.isfile(path):
            return []
        try:
            import subprocess
            out = subprocess.run(["fc-scan", "--format", "%{family}\n", path],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            return []
        fams = []
        for line in out.splitlines():
            for f in line.split(","):
                f = f.strip()
                if f and f not in fams:
                    fams.append(f)
        return fams

    def _setup_tk_fonts(self):
        families = list(tkfont.families(self.root))
        lower = {f.lower(): f for f in families}
        wanted = self._font_file_families(self._devanagari_font_path) + TK_DEVANAGARI_FAMILIES
        family = next((lower[w.lower()] for w in wanted if w.lower() in lower), None)
        if not family:
            family = next((f for f in families if any(
                k in f.lower() for k in ("devanagari", "nirmala", "mangal", "kalimati"))), None)
        if family:
            for name, size in (("TkDefaultFont", 10), ("TkTextFont", 11),
                               ("TkFixedFont", 11), ("TkMenuFont", 10),
                               ("TkHeadingFont", 10)):
                try:
                    tkfont.nametofont(name).configure(family=family, size=size)
                except tk.TclError:
                    pass
            try:
                ttk.Style().configure("Treeview", rowheight=26)
            except tk.TclError:
                pass
            print(f"[ocr-tool] Tk widget font: {family}")
        else:
            print("[ocr-tool] Tk sees no Devanagari font family "
                  f"({len(families)} families visible to Tk).")

        fs = self._tk_fontsystem()
        print(f"[ocr-tool] Tk {tk.TkVersion}, font system: {fs}")
        if fs == "x11":
            print("[ocr-tool] Your Tk has no Xft, so text fields can't draw Devanagari "
                  "(they show \\uXXXX). The saved data and the Preview line are fine.\n"
                  "  Fix A (system Python): sudo apt install python3-tk, run with /usr/bin/python3\n"
                  "  Fix B (conda): conda install -c conda-forge \"tk=*=xft_*\"")
            self.status.set("Tk has no Xft: Devanagari in text fields shows as \\uXXXX — "
                            "data is fine, use the Preview line (fix: see console).")
        elif not family:
            self.status.set("Tk can't see a Devanagari font — run 'fc-cache -f' and restart; "
                            "use the Preview line meanwhile.")

    def _schedule_text_preview(self):
        if getattr(self, "_preview_job", None):
            try:
                self.root.after_cancel(self._preview_job)
            except tk.TclError:
                pass
        self._preview_job = self.root.after(120, self._render_text_preview)

    def _render_text_preview(self):
        self._preview_job = None
        text = self.sel_text_var.get()
        if _looks_like_html(text):
            text = re.sub(r"\s+", " ", _strip_html_tags(text))
        text = _fix_double_escaped(text).replace("\n", " ")[:300]
        width = max(200, self.preview_lbl.winfo_width() - 4)
        if not text.strip():
            self.preview_lbl.config(image="", text="", height=1)
            self._preview_photo = None
            return
        font = self._get_label_font(20)
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        lines, cur = [], ""
        for word in text.split(" "):
            cand = (cur + " " + word).strip()
            try:
                wpx = probe.textlength(cand, font=font)
            except Exception:
                wpx = len(cand) * 11
            if wpx > width and cur:
                lines.append(cur)
                cur = word
                if len(lines) == 3:
                    break
            else:
                cur = cand
        if len(lines) < 3 and cur:
            lines.append(cur)
        line_h = 30
        img = Image.new("RGB", (width, line_h * len(lines) + 6), "white")
        draw = ImageDraw.Draw(img)
        for i, ln in enumerate(lines):
            draw.text((4, 3 + i * line_h), ln, fill=(0, 0, 0), font=font)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_lbl.config(image=self._preview_photo, text="", height=img.height)

    # ==================================================================
    # DISPLAY TEXT CLEANING
    # ==================================================================
    def _clean_display_text(self, d, max_len=60):
        label = canonical_label(d.get("label")).lower()
        text = d.get("text", "") or ""
        looks_like_html = _looks_like_html(text)

        if label == "table" or looks_like_html:
            if looks_like_html:
                preview = _strip_html_tags(text).replace("\n", " ").strip()
                preview = re.sub(r"\s+", " ", preview)
                if preview:
                    return preview[:max_len]
            return "[table]"

        return text[:max_len].replace("\n", " ")

    @staticmethod
    def _text_from_dets(dets):
        ordered = sorted(enumerate(dets), key=lambda t: _order_key(t[1], t[0]))
        parts = []
        for _i, d in ordered:
            t = d.get("text") or ""
            if d.get("label") == "table" or _looks_like_html(t):
                t = _strip_html_tags(t)
            if t.strip():
                parts.append(t)
        return "\n\n".join(parts)

    # ==================================================================
    # OLLAMA STATUS
    # ==================================================================
    def _check_ollama_async(self):
        threading.Thread(target=self._check_ollama_worker, daemon=True).start()

    def _check_ollama_worker(self):
        if not REQUESTS_AVAILABLE:
            self.root.after(0, lambda: self.ollama_status_lbl.config(
                text="Ollama: 'requests' not installed", foreground="#a33"))
            return
        available = self.ollama.is_available()
        models = self.ollama.list_models() if available else []
        self.root.after(0, lambda: self._update_ollama_status(available, models))

    @staticmethod
    def _pick_preferred_model(models, default):
        if default in models:
            return default
        for prefix in PREFERRED_MODEL_PREFIXES:
            for m in models:
                if m.startswith(prefix):
                    return m
        return models[0] if models else default

    def _update_ollama_status(self, available, models):
        self.ollama_available = available
        self.ollama_models = models
        if available:
            self.ollama_status_lbl.config(
                text=f"Ollama: connected ({len(models)} models)", foreground="#2a7")
            values = models if models else [DEFAULT_TEXT_MODEL]
            self.text_model_combo["values"] = values
            self.vision_model_combo["values"] = values
            self.text_model_var.set(self._pick_preferred_model(models, DEFAULT_TEXT_MODEL))
            self.vision_model_var.set(self._pick_preferred_model(models, DEFAULT_VISION_MODEL))
        else:
            self.ollama_status_lbl.config(
                text="Ollama: not reachable (LLM features disabled)", foreground="#a33")

    # ==================================================================
    # CANVAS SELECTION
    # ==================================================================
    def _canvas_to_image_coords(self, cx, cy):
        ox, oy = self.preview_offset
        return (cx - ox) / self.preview_scale, (cy - oy) / self.preview_scale

    def _get_handles(self, bbox):
        x, y, w, h = bbox
        ox, oy = self.preview_offset
        s = self.preview_scale
        x0c, y0c = ox + x * s, oy + y * s
        x1c, y1c = ox + (x + w) * s, oy + (y + h) * s
        xm, ym = (x0c + x1c) / 2, (y0c + y1c) / 2
        return {
            "nw": (x0c, y0c), "n": (xm, y0c), "ne": (x1c, y0c),
            "w": (x0c, ym), "e": (x1c, ym),
            "sw": (x0c, y1c), "s": (xm, y1c), "se": (x1c, y1c),
        }

    def _hit_test_handle(self, cx, cy, bbox):
        for name, (hx, hy) in self._get_handles(bbox).items():
            if abs(cx - hx) <= self.HANDLE_HIT_TOL and abs(cy - hy) <= self.HANDLE_HIT_TOL:
                return name
        return None

    def _hit_test_detection(self, ix, iy):
        best, best_area = None, None
        for idx, d in enumerate(self.detections):
            x, y, w, h = d["bbox"]
            if x <= ix <= x + w and y <= iy <= y + h:
                area = w * h
                if best_area is None or area < best_area:
                    best, best_area = idx, area
        return best

    def _on_canvas_press(self, event):
        if self.original_image is None:
            return
        self.canvas.focus_set()
        self._autosave_pending_new()
        self.drag_start = (event.x, event.y)
        self.interaction_mode = None
        self._edit_snap = None
        force_new = bool(event.state & 0x0001)
        if not force_new and self.selected_bbox is not None:
            handle = self._hit_test_handle(event.x, event.y, self.selected_bbox)
            if handle:
                self.interaction_mode = "resize"
                self.active_handle = handle
                self.resize_orig_bbox = self.selected_bbox
                self._edit_snap = self._snapshot()
                return
            ix, iy = self._canvas_to_image_coords(event.x, event.y)
            x, y, w, h = self.selected_bbox
            if x <= ix <= x + w and y <= iy <= y + h:
                self.interaction_mode = "move"
                self.move_orig_bbox = self.selected_bbox
                self.move_start_img = (ix, iy)
                self._edit_snap = self._snapshot()
                return

    def _on_canvas_drag(self, event):
        if self.drag_start is None:
            return
        if self.interaction_mode == "resize":
            self._update_resize(event)
            self.refresh_preview()
            return
        if self.interaction_mode == "move":
            self._update_move(event)
            self.refresh_preview()
            return
        self.interaction_mode = "new"
        self.canvas.delete("dragbox")
        x0, y0 = self.drag_start
        self.canvas.create_rectangle(x0, y0, event.x, event.y,
                                     outline="#ffd000", width=2, tag="dragbox")

    def _update_resize(self, event):
        ix, iy = self._canvas_to_image_coords(event.x, event.y)
        x, y, w, h = self.resize_orig_bbox
        x0, y0, x1, y1 = x, y, x + w, y + h
        handle = self.active_handle
        if "n" in handle: y0 = iy
        if "s" in handle: y1 = iy
        if "w" in handle: x0 = ix
        if "e" in handle: x1 = ix
        nx0, nx1 = min(x0, x1), max(x0, x1)
        ny0, ny1 = min(y0, y1), max(y0, y1)
        h_img, w_img = self.original_image.shape[:2]
        nx0, ny0 = max(0, nx0), max(0, ny0)
        nx1, ny1 = min(w_img, nx1), min(h_img, ny1)
        self.selected_bbox = (int(nx0), int(ny0),
                              max(1, int(nx1 - nx0)), max(1, int(ny1 - ny0)))

    def _update_move(self, event):
        ix, iy = self._canvas_to_image_coords(event.x, event.y)
        dx, dy = ix - self.move_start_img[0], iy - self.move_start_img[1]
        x, y, w, h = self.move_orig_bbox
        h_img, w_img = self.original_image.shape[:2]
        nx = min(max(0, x + dx), max(0, w_img - w))
        ny = min(max(0, y + dy), max(0, h_img - h))
        self.selected_bbox = (int(nx), int(ny), w, h)

    def _on_canvas_release(self, event):
        if self.drag_start is None or self.original_image is None:
            return
        mode = self.interaction_mode
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None
        self.interaction_mode = None
        self.canvas.delete("dragbox")

        if mode in ("resize", "move"):
            idx = self.selected_detection_index
            if idx is not None and 0 <= idx < len(self.detections):
                old = tuple(self.detections[idx]["bbox"])
                if tuple(self.selected_bbox) != old:
                    self._push_undo(self._edit_snap)
                    self.detections[idx]["bbox"] = tuple(self.selected_bbox)
                    self.detections[idx]["quad"] = quad_from_bbox(self.selected_bbox)
                    self._sync_tree_row(idx)
            self._edit_snap = None
            self._update_selection_panel()
            self.refresh_preview()
            return

        if abs(x1 - x0) < 4 and abs(y1 - y0) < 4:
            ix, iy = self._canvas_to_image_coords(x0, y0)
            idx = self._hit_test_detection(ix, iy)
            if idx is not None:
                self._select_detection(idx)
            else:
                self._clear_selection()
            return

        self._autosave_pending_new()
        ix0, iy0 = self._canvas_to_image_coords(min(x0, x1), min(y0, y1))
        ix1, iy1 = self._canvas_to_image_coords(max(x0, x1), max(y0, y1))
        h, w = self.original_image.shape[:2]
        ix0, iy0 = max(0, int(ix0)), max(0, int(iy0))
        ix1, iy1 = min(w, int(ix1)), min(h, int(iy1))
        if ix1 <= ix0 or iy1 <= iy0:
            return
        self.selected_bbox = (ix0, iy0, ix1 - ix0, iy1 - iy0)
        self.selected_detection_index = None
        self.status.set(f"New box {self.selected_bbox} — type text and press Enter.")
        self._update_selection_panel()
        self.refresh_preview()
        self.sel_entry.focus_set()

    def _on_canvas_motion(self, event):
        if self.selected_bbox is None or self.drag_start is not None:
            return
        handle = self._hit_test_handle(event.x, event.y, self.selected_bbox)
        cursor_map = {
            "nw": "top_left_corner", "se": "bottom_right_corner",
            "ne": "top_right_corner", "sw": "bottom_left_corner",
            "n": "top_side", "s": "bottom_side", "e": "right_side", "w": "left_side",
        }
        if handle:
            self.canvas.config(cursor=cursor_map[handle])
            return
        ix, iy = self._canvas_to_image_coords(event.x, event.y)
        x, y, w, h = self.selected_bbox
        self.canvas.config(cursor="fleur" if (x <= ix <= x + w and y <= iy <= y + h)
                           else "crosshair")

    def _select_detection(self, idx):
        self._autosave_pending_new()
        if idx >= len(self.detections):
            return
        self.selected_detection_index = idx
        self.selected_bbox = tuple(self.detections[idx]["bbox"])
        children = self.tree.get_children()
        if idx < len(children):
            self.tree.selection_set(children[idx])
            self.tree.see(children[idx])
        self._update_selection_panel()
        self.refresh_preview()

    def _drop_selection(self, refresh=True):
        self.selected_bbox = None
        self.selected_detection_index = None
        try:
            self.tree.selection_remove(self.tree.selection())
        except tk.TclError:
            pass
        self._update_selection_panel()
        if refresh:
            self.refresh_preview()

    def _clear_selection(self):
        self._autosave_pending_new()
        self._drop_selection()

    def _tree_values(self, d):
        x, y, w, h = d["bbox"]
        flags = ""
        if d.get("is_bold"):       flags += "B"
        if d.get("is_italic"):     flags += "I"
        if d.get("is_underlined"): flags += "U"
        return (
            d.get("label", DEFAULT_LABEL),
            "" if d.get("order") is None else d["order"],
            d.get("status", ""),
            flags,
            self._clean_display_text(d, max_len=80),
            d.get("conf", 100), x, y, w, h,
        )

    def _rebuild_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for d in self.detections:
            self.tree.insert("", tk.END, values=self._tree_values(d))

    def _sync_tree_row(self, idx):
        children = self.tree.get_children()
        if idx >= len(children):
            return
        self.tree.item(children[idx], values=self._tree_values(self.detections[idx]))

    def _on_label_changed(self):
        idx = self.selected_detection_index
        if idx is None:
            return
        new = self.sel_label_var.get()
        if self.detections[idx].get("label") == new:
            return
        self._push_undo()
        self.detections[idx]["label"] = new
        self._sync_tree_row(idx)
        self.refresh_preview()

    def _update_selection_panel(self):
        if self.selected_bbox is None:
            self.sel_text_var.set("")
            self.sel_label_var.set(DEFAULT_LABEL)
            self.sel_order_var.set("")
            self.sel_save_btn.config(state=tk.DISABLED, text="Save")
            self.sel_review_btn.config(state=tk.DISABLED)
            self.sel_delete_btn.config(state=tk.DISABLED)
            self.sel_info_lbl.config(text="No selection — drag on the image to add a box.")
            return
        x, y, w, h = self.selected_bbox
        if self.selected_detection_index is not None:
            d = self.detections[self.selected_detection_index]
            self.sel_text_var.set(d["text"])
            self.sel_label_var.set(canonical_label(d.get("label")))
            self.sel_order_var.set("" if d.get("order") is None else str(d["order"]))
            self.sel_save_btn.config(state=tk.NORMAL, text="Save")
            self.sel_review_btn.config(state=tk.NORMAL)
            self.sel_delete_btn.config(state=tk.NORMAL)
            info = f"Editing detection #{self.selected_detection_index} — {w}x{h} @ ({x},{y})."
            if d.get("status"):
                info += f" [status: {d['status']}]"
            if d.get("label") == "table" or _looks_like_html(d.get("text", "")):
                info += " [TABLE: HTML content preserved in Text field]"
            if d.get("is_underlined"):
                info += " [U]"
            if d.get("is_bold"):
                info += " [B]"
            if d.get("is_italic"):
                info += " [I]"
            self.sel_info_lbl.config(text=info)
        else:
            self.sel_text_var.set("")
            self.sel_label_var.set(DEFAULT_LABEL)
            self.sel_order_var.set(str(len(self.detections) + 1))
            self.sel_save_btn.config(state=tk.NORMAL, text="Save as New")
            self.sel_review_btn.config(state=tk.DISABLED)
            self.sel_delete_btn.config(state=tk.DISABLED)
            extra = ""
            if getattr(self, "_pending_underline", False):
                extra = " [U for next box]"
            self.sel_info_lbl.config(
                text=f"New box {w}x{h} @ ({x},{y}) — type text and press Enter/Save.{extra}"
            )

    def _autosave_pending_new(self):
        if (self.selected_bbox is not None
                and self.selected_detection_index is None
                and self.sel_text_var.get().strip()):
            self._save_selected_text()
            self.selected_bbox = None
            self.selected_detection_index = None
            self._update_selection_panel()

    def _parse_order(self):
        raw = self.sel_order_var.get().strip()
        if raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _clean_label_text(self, text, label):
        text, label, fixes = clean_block_content(text, label)
        if label == "table" and text.strip():
            problems = validate_table_html(text)
            if problems:
                ok = messagebox.askyesno(
                    "Table HTML problems",
                    "This table HTML has structural problems:\n\n  • "
                    + "\n  • ".join(problems[:8])
                    + "\n\nSave it anyway?")
                if not ok:
                    return None, label, []
        return text, label, fixes

    def _save_selected_text(self):
        if self.selected_bbox is None:
            return
        label = self.sel_label_var.get() or DEFAULT_LABEL
        text, label, fixes = self._clean_label_text(self.sel_text_var.get(), label)
        if text is None:
            return
        if label != self.sel_label_var.get():
            self.sel_label_var.set(label)
        order = self._parse_order()
        if label in NON_FLOW_LABELS:
            order = None

        if self.selected_detection_index is not None:
            idx = self.selected_detection_index
            d = self.detections[idx]
            new_bbox = tuple(self.selected_bbox)
            changed = (text != d["text"] or label != d.get("label")
                       or order != d.get("order") or new_bbox != tuple(d["bbox"]))
            if not changed:
                self.status.set("No changes.")
                self._update_selection_panel()
                return
            self._push_undo()
            if text != d["text"]:
                d.setdefault("ocr_text", d["text"])
                d["status"] = "reviewed"
            d["text"] = text
            d["bbox"] = new_bbox
            d["quad"] = quad_from_bbox(new_bbox)
            d["label"] = label
            d["order"] = order
            self._sync_tree_row(idx)
            self.status.set("Updated detection." + (f" Auto-fixed: {'; '.join(fixes)}." if fixes else ""))
        else:
            needs_text = label in TEXT_REQUIRED_LABELS
            if needs_text and not text.strip():
                messagebox.showinfo("No text",
                                    f"'{label}' boxes need a transcription. Type the text, "
                                    "or pick a label that allows empty text "
                                    "(e.g. image, photo, logo, signature, seal, table, checkbox).")
                return
            self._push_undo()
            new_det = {
                "text": text, "conf": 100, "bbox": tuple(self.selected_bbox),
                "quad": quad_from_bbox(self.selected_bbox),
                "label": label, "order": order, "status": "reviewed",
            }
            if getattr(self, "_pending_underline", False):
                new_det["is_underlined"] = True
                self._pending_underline = False
            replaced = [i for i, d in enumerate(self.detections)
                        if d.get("status") != "reviewed"
                        and covered_fraction(d, new_det) >= LEFTOVER_COVER]
            if replaced:
                if new_det["order"] is None and label not in NON_FLOW_LABELS:
                    olds = [self.detections[i].get("order") for i in replaced]
                    olds = [o for o in olds if isinstance(o, int)]
                    new_det["order"] = min(olds) if olds else None
                for i in sorted(replaced, reverse=True):
                    del self.detections[i]
            self.detections.append(new_det)
            idx = len(self.detections) - 1
            if replaced:
                self._rebuild_tree()
            else:
                self.tree.insert("", tk.END, values=self._tree_values(self.detections[idx]))
            self.selected_detection_index = idx
            self._stash_current_page()
            msg = f"Added new detection ({label})."
            if replaced:
                msg += f" Replaced {len(replaced)} unreviewed OCR box(es) under it (Ctrl+Z to undo)."
            self.status.set(msg)
        self._update_selection_panel()
        self.refresh_preview()

    def _mark_reviewed(self):
        if self.selected_detection_index is None:
            return
        idx = self.selected_detection_index
        if self.detections[idx].get("status") == "reviewed":
            return
        self._push_undo()
        self.detections[idx]["status"] = "reviewed"
        self._sync_tree_row(idx)
        self._update_selection_panel()
        self.status.set(f"Detection #{idx} marked reviewed.")

    def _mark_all_reviewed(self):
        if not self.detections:
            return
        leftovers = find_leftover_boxes(self.detections)
        todo = [d for d in self.detections if d.get("status") != "reviewed"]
        if not todo:
            self.status.set("All detections on this page are already reviewed.")
            return
        self._push_undo()
        for i in sorted(leftovers, reverse=True):
            del self.detections[i]
        if leftovers:
            self.selected_detection_index = None
            self.selected_bbox = None
            self._stash_current_page()
        todo = [d for d in self.detections if d.get("status") != "reviewed"]
        for d in todo:
            d["status"] = "reviewed"
        keep = self.selected_detection_index
        self._rebuild_tree()
        if keep is not None:
            self._select_detection(keep)
        self._update_selection_panel()
        self.status.set(f"Marked {len(todo)} detection(s) reviewed on this page."
                        + (f" Deleted {len(leftovers)} leftover OCR box(es) first (Ctrl+Z to undo)."
                           if leftovers else ""))

    def _delete_selected(self):
        if self.selected_detection_index is None:
            return
        idx = self.selected_detection_index
        self._push_undo()
        children = self.tree.get_children()
        if idx < len(children):
            self.tree.delete(children[idx])
        del self.detections[idx]
        self.selected_detection_index = None
        self.selected_bbox = None
        self._stash_current_page()
        self._update_selection_panel()
        self.refresh_preview()
        self.status.set("Deleted detection box (Ctrl+Z to undo).")

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx >= len(self.detections) or idx == self.selected_detection_index:
            return
        self._autosave_pending_new()
        self.selected_detection_index = idx
        self.selected_bbox = tuple(self.detections[idx]["bbox"])
        self._update_selection_panel()
        self.refresh_preview()

    # ==================================================================
    # PAGE-LEVEL TOOLS: reading order, table check, line detection
    # ==================================================================
    def auto_order(self):
        if not self.detections:
            return
        self._push_undo()
        ordered = reading_order(self.detections)
        n = 0
        for d in ordered:
            if canonical_label(d.get("label")) in NON_FLOW_LABELS:
                d["order"] = None
            else:
                n += 1
                d["order"] = n
        self._set_detections(ordered)
        self.status.set("Reading order reassigned top-to-bottom, left-to-right "
                        "(headers/footers/graphics excluded; Ctrl+Z to undo).")

    def validate_tables(self):
        if not self.detections:
            messagebox.showinfo("Check Page", "No boxes on this page.")
            return
        snap = self._snapshot()
        fixed_lines = []
        for i, d in enumerate(self.detections):
            text, label, fixes = clean_block_content(d.get("text", ""), d.get("label", DEFAULT_LABEL))
            if label in NON_FLOW_LABELS and d.get("order") is not None:
                d["order"] = None
                fixes.append(f"order cleared ({label} is outside the reading flow)")
            if fixes or text != d.get("text") or label != d.get("label"):
                d["text"], d["label"] = text, label
                self._sync_tree_row(i)
                fixed_lines.append(f"#{i} (order {d.get('order')}): " + "; ".join(fixes or ["normalized"]))
        if fixed_lines:
            self._push_undo(snap)
            self.refresh_preview()
            self._update_selection_panel()

        tables = [(i, d) for i, d in enumerate(self.detections) if d.get("label") == "table"]
        table_lines = []
        for i, d in tables:
            problems = validate_table_html(d.get("text") or "")
            tag = f"#{i} (order {d.get('order')})"
            table_lines.append(f"{tag}: OK" if not problems
                               else f"{tag}:\n    - " + "\n    - ".join(problems))
        warns = [w for w in qa_page_warnings(self.detections) if not w.split(": ", 1)[-1].startswith("table:")]

        parts = []
        if fixed_lines:
            parts.append(f"AUTO-FIXED ({len(fixed_lines)}, Ctrl+Z to undo):\n  " + "\n  ".join(fixed_lines))
        if table_lines:
            parts.append("TABLES:\n  " + "\n  ".join(table_lines))
        parts.append(("CHECK AGAINST THE IMAGE:\n  " + "\n  ".join(warns[:40])
                      + (f"\n  ... {len(warns) - 40} more (see console)" if len(warns) > 40 else ""))
                     if warns else "No other warnings. Page looks clean.")
        for w in warns:
            print(f"[qa] page {self.current_page + 1}: {w}")
        messagebox.showinfo("Check Page", "\n\n".join(parts))

    # ==================================================================
    # LINE DETECTION (per block, inside the block crop)
    # ==================================================================
    def _detect_lines_in_crop(self, crop_bgr, lang_str="eng+nep"):
        """
        Run a line-level OCR on a single block crop with Tesseract psm 6.
        Words are grouped by (block_num, par_num, line_num) into lines.
        Returns: list of {"text", "bbox_in_crop": (x,y,w,h), "conf"}.
        """
        try:
            data = pytesseract.image_to_data(
                crop_bgr, lang=lang_str, config="--oem 3 --psm 6",
                output_type=Output.DICT,
            )
        except Exception as e:
            print(f"[line-ocr] {e}")
            return []
        groups = {}
        n = len(data["text"])
        for i in range(n):
            t = (data["text"][i] or "").strip()
            if not t:
                continue
            raw_conf = data["conf"][i]
            try:
                conf = int(float(raw_conf))
            except (TypeError, ValueError):
                conf = -1
            if conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            g = groups.setdefault(key, {
                "words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h, "confs": [],
            })
            g["words"].append(t)
            g["x0"] = min(g["x0"], x)
            g["y0"] = min(g["y0"], y)
            g["x1"] = max(g["x1"], x + w)
            g["y1"] = max(g["y1"], y + h)
            g["confs"].append(conf)
        out = []
        for key in sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2])):
            g = groups[key]
            avg_conf = int(sum(g["confs"]) / len(g["confs"])) if g["confs"] else 0
            out.append({
                "text": " ".join(g["words"]),
                "bbox_in_crop": (g["x0"], g["y0"], g["x1"] - g["x0"], g["y1"] - g["y0"]),
                "conf": avg_conf,
            })
        return out

    @staticmethod
    def _emit_lines_for_block(bid, d, out_lines):
        """
        Fallback when the crop yields no lines: split the block's content on
        newlines and divide the block's bbox height proportionally.
        """
        text = d.get("text") or ""
        if not text.strip():
            return
        label = canonical_label(d.get("label"))
        if label in EMPTY_CONTENT_LABELS or label == "table" or _looks_like_html(text):
            return
        x, y, w, h = [int(v) for v in d["bbox"]]
        parts = text.split("\n") or [text]
        n = max(1, len(parts))
        for k, part in enumerate(parts):
            y0 = int(y + h * k / n)
            y1 = int(y + h * (k + 1) / n)
            out_lines.append({
                "text": part,
                "bbox": [x, y0, x + w, max(y0 + 1, y1)],
                "block_id": d.get("block_id", bid),
                "line_in_block": k,
                "is_heading": bool(d.get("is_heading", False)),
                "is_underlined": bool(d.get("is_underlined", False)),
                "font_size": d.get("font_size") or max(1, (y1 - y0)),
                "font_id": d.get("font_id", 0),
                "quad": quad_from_bbox((x, y0, w, max(1, y1 - y0))),
                "table_index": d.get("table_index"),
                "cell_id": d.get("cell_id"),
                "conf": d.get("conf", 100),
            })

    def run_line_ocr_page(self):
        """
        Crop each non-table block on the current page and detect the individual
        lines inside it. Stores results in self.lines_by_page[current_page].
        """
        if self.original_image is None:
            return
        if not self.detections:
            messagebox.showinfo("No blocks",
                                "Run block-level OCR first, or draw blocks by hand.")
            return
        lang = self.lang_var.get()
        self.status.set("Detecting lines inside each block...")
        self.root.update_idletasks()
        lines = []
        for bid, d in enumerate(self.detections):
            label = canonical_label(d.get("label"))
            if label in EMPTY_CONTENT_LABELS:
                continue
            if label == "table" or _looks_like_html(d.get("text", "")):
                continue
            x, y, w, h = [int(v) for v in d["bbox"]]
            h_img, w_img = self.original_image.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            if w < 4 or h < 4:
                continue
            crop = self.original_image[y:y + h, x:x + w]
            hits = self._detect_lines_in_crop(crop, lang)
            if not hits:
                self._emit_lines_for_block(bid, d, lines)
                continue
            for k, lh in enumerate(hits):
                lx, ly, lw, lh_h = lh["bbox_in_crop"]
                gx, gy = x + lx, y + ly
                lines.append({
                    "text": lh["text"],
                    "bbox": [gx, gy, gx + lw, gy + lh_h],
                    "block_id": d.get("block_id", bid),
                    "line_in_block": k,
                    "is_heading": bool(d.get("is_heading", False)),
                    "is_underlined": bool(d.get("is_underlined", False)),
                    "font_size": d.get("font_size") or lh_h,
                    "font_id": d.get("font_id", 0),
                    "quad": quad_from_bbox((gx, gy, lw, lh_h)),
                    "table_index": d.get("table_index"),
                    "cell_id": d.get("cell_id"),
                    "conf": lh.get("conf", 100),
                })
        self.lines_by_page[self.current_page] = lines
        self._set_dirty(True)
        self.status.set(f"Line detection complete: {len(lines)} line(s) on page "
                        f"{self.current_page + 1}.")
        if getattr(self, "auto_underline_var", None) and self.auto_underline_var.get():
            try:
                self.detect_underlines_page()
            except Exception as e:
                print(f"[underline auto] {e}")

    # ==================================================================
    # UNDERLINE DETECTION (automatic)
    # ==================================================================
    def _crop_has_underline(self, crop_bgr, min_cover=0.55):
        """
        True if a horizontal dark run of pixels sits just below the bottom of
        the ink in this crop. Operates on a single-line crop (height roughly
        equals the line height + a few pixels below).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # Binarize: ink = white, background = black.
        bw = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        h, w = bw.shape
        if h < 4 or w < 8:
            return False
        # Row-wise ink ratio.
        row_ratio = (bw > 0).sum(axis=1) / float(w)
        # Bottom third of the crop is where an underline would sit.
        start = max(0, int(h * 0.66))
        best = row_ratio[start:].max() if start < h else 0.0
        if best < min_cover:
            return False
        # Require the row above to be much lighter: an underline is a single
        # dark band, not the bottom of dense glyphs.
        for r in range(start, h):
            if row_ratio[r] >= min_cover:
                above = row_ratio[r - 1] if r - 1 >= 0 else 0.0
                if above < best * 0.6:
                    return True
                break
        return False

    def detect_underlines_page(self):
        """
        Run underline detection on every text line on the current page.
        Lines must exist (run Detect Lines first). Marks is_underlined on the
        corresponding line records and on their parent blocks.
        """
        if self.original_image is None:
            return
        lines = self.lines_by_page.get(self.current_page)
        if not lines:
            if not messagebox.askyesno(
                    "No lines yet",
                    "Underline detection runs on line crops.\n\n"
                    "Run 'Detect Lines' first?\n\n"
                    "(Yes: run it now, then continue.)"):
                return
            self.run_line_ocr_page()
            lines = self.lines_by_page.get(self.current_page) or []
            if not lines:
                return
        h_img, w_img = self.original_image.shape[:2]
        hits_by_block = {}
        for ln in lines:
            x0, y0, x1, y1 = ln["bbox"]
            # Extend the crop a little below the line so an underline is included.
            pad_below = max(4, int((y1 - y0) * 0.4))
            cx0 = max(0, x0)
            cy0 = max(0, y0)
            cx1 = min(w_img, x1)
            cy1 = min(h_img, y1 + pad_below)
            if cx1 - cx0 < 8 or cy1 - cy0 < 4:
                ln["is_underlined"] = False
                continue
            crop = self.original_image[cy0:cy1, cx0:cx1]
            ln["is_underlined"] = bool(self._crop_has_underline(crop))
            if ln["is_underlined"]:
                hits_by_block[ln.get("block_id", 0)] = True
        # Propagate to parent blocks.
        self._push_undo()
        for bid, d in enumerate(self.detections):
            key = d.get("block_id", bid)
            if key in hits_by_block:
                d["is_underlined"] = True
        self._set_dirty(True)
        n = sum(1 for ln in lines if ln.get("is_underlined"))
        self.status.set(f"Underline detection: {n} of {len(lines)} line(s) marked.")
        self.refresh_preview()

    # ==================================================================
    # REGION HELPERS
    # ==================================================================
    def _current_region(self):
        if self.original_image is None:
            return None
        if self.selected_bbox is not None:
            bbox = self.selected_bbox
        elif self.selected_detection_index is not None:
            bbox = self.detections[self.selected_detection_index]["bbox"]
        else:
            return None
        x, y, w, h = [int(v) for v in bbox]
        if w < 2 or h < 2:
            return None
        return x, y, w, h

    def _crop(self, x, y, w, h):
        return self.original_image[y:y + h, x:x + w].copy()

    # ==================================================================
    # TABLE OCR ON SELECTION
    # ==================================================================
    def _run_table_on_selection(self):
        region = self._current_region()
        if region is None:
            messagebox.showinfo("No selection",
                                "Draw a box around the table you want to OCR.")
            return
        x, y, w, h = region
        crop = self._crop(x, y, w, h)
        lang = self.lang_var.get()
        replace_idx = self.selected_detection_index
        self.backends.apply_fixes = self.apply_fixes_var.get()

        def worker():
            try:
                html = self.backends.run_table_only(crop, lang)
                self.root.after(0, lambda: self._insert_table_detection(
                    x, y, w, h, html, replace_idx))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._table_error(err))

        self.table_sel_btn.config(state=tk.DISABLED)
        self.status.set("Running table OCR on selected region "
                        "(first run downloads models)...")
        threading.Thread(target=worker, daemon=True).start()

    def _table_error(self, err):
        self.table_sel_btn.config(state=tk.NORMAL)
        self.status.set("Table OCR failed.")
        messagebox.showerror("Table OCR Error", f"Table OCR failed:\n{err}")

    def _insert_table_detection(self, x, y, w, h, html, replace_idx=None):
        self.table_sel_btn.config(state=tk.NORMAL)
        if not html:
            self.status.set("No table detected in that region.")
            messagebox.showinfo("No table",
                                "No table structure was found in the selected region.")
            return
        self._push_undo()
        new_det = {
            "text": repair_table_html(normalize_text(html)), "conf": 100,
            "bbox": (x, y, w, h), "quad": quad_from_bbox((x, y, w, h)),
            "label": "table", "order": None, "status": "auto",
        }
        if replace_idx is not None and 0 <= replace_idx < len(self.detections):
            old = self.detections[replace_idx]
            new_det["order"] = old.get("order")
            self.detections[replace_idx] = new_det
            self._sync_tree_row(replace_idx)
            idx = replace_idx
        else:
            self.detections.append(new_det)
            idx = len(self.detections) - 1
            self.tree.insert("", tk.END, values=self._tree_values(new_det))
        self._stash_current_page()
        self.selected_detection_index = idx
        self.selected_bbox = (x, y, w, h)
        self._update_selection_panel()
        self.refresh_preview()
        problems = validate_table_html(new_det["text"])
        if problems:
            self.status.set("Table inserted — structure issues: " + "; ".join(problems[:3]))
        else:
            self.status.set("Table inserted. Export via any JSON/CSV format.")

    # ==================================================================
    # FAST LOCAL RE-READ (PaddleOCR, no LLM)
    # ==================================================================
    def _run_paddle_reread(self):
        region = self._current_region()
        if region is None:
            messagebox.showinfo("No selection", "Draw a box or select a detection.")
            return
        crop = self._crop(*region)
        lang = self.lang_var.get()
        self.backends.apply_fixes = self.apply_fixes_var.get()
        self.status.set("Re-reading region with PaddleOCR...")
        self.ai_timing_lbl.config(text="Working (local PaddleOCR)...")

        def worker():
            t0 = time.time()
            try:
                txt = self.backends.reread_crop(crop, lang)
                el = time.time() - t0
                self.root.after(0, lambda: self._reread_done(txt, el))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._ollama_error(err, title="Re-read Error"))

        threading.Thread(target=worker, daemon=True).start()

    def _reread_done(self, text, elapsed):
        self._ollama_fix_done(text, elapsed)
        if self.selected_detection_index is None and self.selected_bbox is not None:
            self.sel_text_var.set(text)
        self.status.set("Re-read complete — review in AI Assist tab, then Accept.")

    # ==================================================================
    # DRAG AND DROP
    # ==================================================================
    def _setup_drag_drop(self):
        if not DND_AVAILABLE:
            return
        try:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event):
        path = event.data.strip("{}")
        if os.path.isfile(path):
            self._load_path(path)

    # ==================================================================
    # FILE LOADING
    # ==================================================================
    def _confirm_discard(self):
        if not self._dirty:
            return True
        ans = messagebox.askyesnocancel(
            "Unsaved annotations",
            "You have unsaved annotation changes.\n\n"
            "Yes — save them first (same as Ctrl+S)\n"
            "No — discard them\n"
            "Cancel — go back")
        if ans is None:
            return False
        if ans:
            return self.quick_save()
        return True

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select an image, PDF or annotation file",
            filetypes=[
                ("Images, PDFs & annotations",
                 "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf *.json *.jsonl"),
                ("PDF", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Annotations", "*.json *.jsonl"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_path(path)

    def _load_path(self, path):
        if not self._confirm_discard():
            return
        if path.lower().endswith((".json", ".jsonl")):
            self._open_annotation_file(path)
        else:
            self._load_source(path, autoload=True)

    def _load_source(self, path, autoload=True):
        self.image_path = path
        self.annotated_image = None
        self.detections = []
        self.page_ocr_cache = {}
        self.lines_by_page = {}
        self.rich_page_meta = {}
        self._pending_underline = False
        self.current_page = 0
        self.selected_bbox = None
        self.selected_detection_index = None
        self._history = {}
        self._annotation_path = None
        self._annotation_kind = None
        self._split_meta = None
        self._disk_sig, self._baseline, self._conflict_seen = {}, {}, {}
        self._set_dirty(False)
        self._update_undo_buttons()
        self._update_selection_panel()
        self.text_box.delete("1.0", tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except Exception:
                pass
            self.pdf_doc = None

        if path.lower().endswith(".pdf"):
            try:
                self.pdf_doc = fitz.open(path)
            except Exception as e:
                messagebox.showerror("PDF Error", f"Could not open PDF:\n{e}")
                return False
            self.page_count = self.pdf_doc.page_count
            if self.split_pdf_var.get():
                self._adopt_manifest_zoom()
            self._render_pdf_page(0)
            if self.split_pdf_var.get():
                res = self._split_pdf_pages()
                if res and res[1]:
                    print(f"[ocr-tool] wrote {res[1]} page image(s) to {res[0]}")
        else:
            self.page_count = 1
            try:
                buf = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is None:
                messagebox.showerror("Error", f"Could not read image:\n{path}")
                return False
            self.raw_page_image = img
            self._after_load()
            if self._split_mode():
                res = self._split_pdf_pages()
                if res and res[1]:
                    print(f"[ocr-tool] wrote page image to {res[0]}")

        if autoload:
            if not (self._split_mode() and self._autoload_split_pages()):
                if not self._autoload_annotations():
                    self._adopt_page_file_meta()
        return True

    # ------------------------------------------------------------------
    # Page-folder helpers (PDFs and single images share one layout)
    # ------------------------------------------------------------------
    _PAGE_FILE_RE = re.compile(r"_page\d{3}$")

    def _is_page_file(self, path=None):
        path = path or self.image_path or ""
        if path.lower().endswith(".pdf"):
            return False
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        stem = os.path.splitext(os.path.basename(path))[0]
        return parent.endswith("_pages") and bool(self._PAGE_FILE_RE.search(stem))

    def _split_mode(self):
        if not self.split_pdf_var.get() or not self.image_path:
            return False
        return self.pdf_doc is not None or not self._is_page_file()

    def _adopt_page_file_meta(self):
        if self.pdf_doc is not None or not self._is_page_file() or self._split_meta:
            return
        d = os.path.dirname(os.path.abspath(self.image_path))
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        man = {}
        try:
            with open(os.path.join(d, SPLIT_MANIFEST), encoding="utf-8") as f:
                man = json.load(f)
        except Exception:
            pass
        key = "source_pdf" if man.get("source_pdf") else "source_image"
        m = re.search(r"_page(\d{3})$", stem)
        self._split_meta = {
            "source_key": key,
            "source": man.get(key) or "",
            "page": int(m.group(1)) if m else 1,
            "page_count": man.get("page_count"),
            "render_zoom": man.get("render_zoom") or 1.0,
        }
        self._annotation_path = os.path.join(d, stem + ".json")
        self._annotation_kind = "split_page"

    def _render_page_image(self, page_num):
        page = self.pdf_doc.load_page(page_num)
        mat = fitz.Matrix(self.pdf_zoom, self.pdf_zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if pix.n == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    def _render_pdf_page(self, page_num):
        try:
            img = self._render_page_image(page_num)
        except Exception as e:
            messagebox.showerror("Render Error", f"Failed to render page:\n{e}")
            return
        self.raw_page_image = img
        self.annotated_image = None
        self.detections = []
        self.selected_bbox = None
        self.selected_detection_index = None
        self._update_selection_panel()
        self._after_load()
        if page_num in self.page_ocr_cache:
            self._show_cached_page(page_num)
        else:
            self._rebuild_tree()
            self.text_box.delete("1.0", tk.END)
            self.refresh_preview()
        self._update_undo_buttons()
        self.root.after_idle(self._check_current_page_external)

    def _after_load(self):
        self._compute_preprocessed()
        self._update_size_info()
        self.refresh_preview()
        self.run_btn.config(state=tk.NORMAL)
        self.run_all_btn.config(state=tk.NORMAL if self.pdf_doc else tk.DISABLED)
        self._update_page_buttons()
        self._set_default_export_path_for_current_type()

        name = os.path.basename(self.image_path)
        if self.pdf_doc:
            self.status.set(f"Loaded PDF: {name} - page {self.current_page + 1}/{self.page_count}.")
        else:
            self.status.set(f"Loaded: {name} - Click 'Run OCR'.")

    def _raw_page_size(self, page):
        if page == self.current_page and self.raw_page_image is not None:
            h, w = self.raw_page_image.shape[:2]
            return int(w), int(h)
        if self.pdf_doc is not None:
            r = self.pdf_doc.load_page(page).rect
            return int(round(r.width * self.pdf_zoom)), int(round(r.height * self.pdf_zoom))
        if self.raw_page_image is not None:
            h, w = self.raw_page_image.shape[:2]
            return int(w), int(h)
        return 0, 0

    # ==================================================================
    # ANNOTATION LOADING (sidecar / explicit)
    # ==================================================================
    def _apply_loaded_blocks(self, blocks, strict=True):
        cur_base = os.path.basename(self.image_path or "")
        cur_stem = os.path.splitext(cur_base)[0]

        def matches(b):
            img = b.get("image")
            if not img:
                return True
            ib = os.path.basename(str(img).replace("\\", "/"))
            return ib == cur_base or os.path.splitext(ib)[0] == cur_stem

        sel = [b for b in blocks if matches(b)]
        if not sel:
            if strict:
                return 0, 0
            sel = blocks

        for b in sel:
            rm = b.get("rich_meta")
            if rm:
                page = b.get("page", 0)
                self.rich_page_meta.setdefault(page, {})
                for k, v in rm.items():
                    if v is not None and k not in self.rich_page_meta[page]:
                        self.rich_page_meta[page][k] = v
                if rm.get("document_id") and not self.doc_meta.get("document_id"):
                    self.doc_meta["document_id"] = rm["document_id"]
                if rm.get("language"):
                    self.doc_meta["language"] = rm["language"]
                    if hasattr(self, "doc_lang_var"):
                        self.doc_lang_var.set(rm["language"])
                if rm.get("digit_style"):
                    self.doc_meta["digit_style"] = rm["digit_style"]
                    if hasattr(self, "doc_digit_var"):
                        self.doc_digit_var.set(rm["digit_style"])

        if self.pdf_doc is None:
            src = next((b for b in sel if b.get("source_pdf") or b.get("source_image")), None)
            if src:
                key = "source_pdf" if src.get("source_pdf") else "source_image"
                self._split_meta = {
                    "source_key": key,
                    "source": src[key],
                    "page": src["page"] + 1,
                    "page_count": src.get("page_count"),
                    "render_zoom": src.get("zoom"),
                }
            else:
                self._split_meta = None
            sel = [dict(b, page=0) for b in sel]

        by_page = {}
        for b in sel:
            by_page.setdefault(b["page"], []).append(b)

        self._stash_current_page()
        loaded, skipped = 0, 0
        for page, bl in sorted(by_page.items()):
            if page < 0 or page >= self.page_count:
                skipped += 1
                continue
            raw_w, raw_h = self._raw_page_size(page)
            fw = next((b["width"] for b in bl if b.get("width")), None)
            fz = next((b["zoom"] for b in bl if b.get("zoom")), None)
            factor = 1.0
            if fw and raw_w:
                factor = raw_w / float(fw)
            elif fz and self.pdf_doc is not None:
                factor = self.pdf_zoom / float(fz)
            dets = [dict(b["det"]) for b in bl]
            if abs(factor - 1.0) > 1e-3:
                dets = scale_detections(dets, factor)
            old = self.page_ocr_cache.get(page)
            if old and old.get("detections"):
                self._push_undo(snap={"dets": copy.deepcopy(old["detections"]),
                                      "scale": old.get("coord_scale", 1.0)},
                                page=page, mark_dirty=False)
            self.page_ocr_cache[page] = {
                "detections": dets, "text": self._text_from_dets(dets),
                "width": raw_w, "height": raw_h, "coord_scale": 1.0,
            }
            loaded += len(dets)

        if self.current_page in self.page_ocr_cache:
            self._show_cached_page(self.current_page)
        self._update_undo_buttons()
        return loaded, skipped

    def _kind_for(self, kinds):
        if self._split_meta:
            return "split_page"
        if kinds == {"annotations"}:
            return "annotations"
        if "rich" in kinds:
            return "rich"
        return next(iter(kinds))

    def _autoload_annotations(self):
        if not self.autoload_var.get() or not self.image_path:
            return False
        for cand in find_sidecar_annotations(self.image_path):
            try:
                blocks, kinds = load_annotation_blocks(cand)
            except Exception as e:
                print(f"[ocr-tool] skipping {cand}: {e}")
                continue
            if not blocks:
                continue
            n, skipped = self._apply_loaded_blocks(blocks, strict=True)
            if not n:
                continue
            self._annotation_path = cand
            self._annotation_kind = self._kind_for(kinds)
            self._set_dirty(False)
            self._disk_sig[os.path.abspath(cand)] = file_sig(cand)
            self._record_disk_state()
            pages = len({p for p in self.page_ocr_cache})
            msg = (f"Loaded {n} existing box(es) on {pages} page(s) from "
                   f"{os.path.basename(cand)}.")
            if skipped:
                msg += f" {skipped} page(s) out of range were skipped."
            self.status.set(msg)
            return True
        return False

    def reload_sidecar(self):
        if not self.image_path:
            messagebox.showinfo("No file", "Open an image or PDF first.")
            return
        if not self._confirm_discard():
            return
        self.page_ocr_cache = {}
        self.lines_by_page = {}
        self.detections = []
        self._history = {}
        self._set_detections([])
        if not self._autoload_annotations_force():
            self.status.set("No matching annotation file found next to this image.")

    def _autoload_annotations_force(self):
        prev = self.autoload_var.get()
        self.autoload_var.set(True)
        try:
            return self._autoload_annotations()
        finally:
            self.autoload_var.set(prev)

    @staticmethod
    def _find_image_for_annotations(json_path, blocks):
        jdir = os.path.dirname(os.path.abspath(json_path))
        seen = []
        for b in blocks:
            img = b.get("image")
            if img and img not in seen:
                seen.append(img)
        for img in seen:
            if os.path.isfile(img):
                return img
            cand = os.path.join(jdir, os.path.basename(str(img).replace("\\", "/")))
            if os.path.isfile(cand):
                return cand
            cand = os.path.join(os.path.dirname(jdir),
                                os.path.basename(str(img).replace("\\", "/")))
            if os.path.isfile(cand):
                return cand
        stem = os.path.splitext(os.path.basename(json_path))[0]
        for suf in _SIDECAR_SUFFIXES:
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                break
        for d in (jdir, os.path.dirname(jdir)):
            for ext in IMAGE_EXTS:
                for e in (ext, ext.upper()):
                    cand = os.path.join(d, stem + e)
                    if os.path.isfile(cand):
                        return cand
        return None

    def _open_annotation_file(self, path):
        try:
            blocks, kinds = load_annotation_blocks(path)
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to read annotations:\n{e}")
            return
        if not blocks:
            messagebox.showinfo("Empty", "No bounding boxes found in that file.")
            return
        img = self._find_image_for_annotations(path, blocks)
        if not img:
            messagebox.showinfo(
                "Image not found",
                "Couldn't find the image/PDF these annotations belong to.\n"
                "Please pick it in the next dialog.")
            img = filedialog.askopenfilename(
                title="Select the image or PDF for these annotations",
                initialdir=os.path.dirname(os.path.abspath(path)),
                filetypes=[("Images & PDFs",
                            "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf"),
                           ("All files", "*.*")])
            if not img:
                return
        if not self._load_source(img, autoload=False):
            return
        n, skipped = self._apply_loaded_blocks(blocks, strict=False)
        self._annotation_path = path
        self._annotation_kind = self._kind_for(kinds)
        self._set_dirty(False)
        self._disk_sig[os.path.abspath(path)] = file_sig(path)
        self._record_disk_state()
        msg = f"Opened {os.path.basename(img)} with {n} box(es) from {os.path.basename(path)}."
        if skipped:
            msg += f" {skipped} page(s) out of range were skipped."
        self.status.set(msg)

    def import_annotations(self):
        path = filedialog.askopenfilename(
            title="Import annotations (annotations JSON/JSONL, PaddleOCR-VL, rich schema)",
            filetypes=[("JSON / JSONL", "*.json *.jsonl"), ("All files", "*.*")],
        )
        if not path:
            return
        if self.original_image is None:
            if self._confirm_discard():
                self._open_annotation_file(path)
            return
        try:
            blocks, kinds = load_annotation_blocks(path)
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to read annotations:\n{e}")
            return
        if not blocks:
            messagebox.showinfo("Empty", "No bounding boxes found in that file.")
            return
        n, skipped = self._apply_loaded_blocks(blocks, strict=True)
        if n == 0:
            names = sorted({os.path.basename(str(b.get("image"))) for b in blocks if b.get("image")})
            if not messagebox.askyesno(
                    "Different image",
                    "These annotations reference a different file:\n  "
                    + ", ".join(names[:5])
                    + f"\n\nThe open file is {os.path.basename(self.image_path)}.\n"
                      "Import them onto this file anyway?"):
                return
            n, skipped = self._apply_loaded_blocks(blocks, strict=False)
        self._annotation_path = path
        self._annotation_kind = self._kind_for(kinds)
        self._set_dirty(True)
        msg = f"Imported {n} box(es) from {os.path.basename(path)} (Ctrl+Z restores the previous boxes)."
        if skipped:
            msg += f" {skipped} page(s) out of range were skipped."
        self.status.set(msg)

    import_paddle_json = import_annotations

    # ==================================================================
    # PREPROCESSING
    # ==================================================================
    def _compute_preprocessed(self):
        if self.raw_page_image is None:
            return
        img = self.raw_page_image.copy()
        if self.pp_deskew.get():
            img = self._deskew(img)
        if self.pp_denoise.get():
            img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
        if self.pp_upscale.get():
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        if self.pp_grayscale.get():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
            )
            img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.original_image = img
        self.annotated_image = None
        self._coord_scale = 2.0 if self.pp_upscale.get() else 1.0

    def _on_preprocess_toggled(self):
        if self.raw_page_image is None:
            return
        old_scale = self._coord_scale
        had = bool(self.detections)
        self._compute_preprocessed()
        if had and abs(self._coord_scale - old_scale) > 1e-6:
            self._push_undo({"dets": copy.deepcopy(self.detections), "scale": old_scale})
            self._set_detections(scale_detections(self.detections, self._coord_scale / old_scale))
        else:
            if self.current_page in self.page_ocr_cache:
                self.page_ocr_cache[self.current_page]["coord_scale"] = self._coord_scale
            self._drop_selection(refresh=False)
            self.refresh_preview()
        self._update_size_info()
        if had and self.pp_deskew.get():
            self.status.set("Deskew is on — existing boxes were kept but may be "
                            "slightly misaligned with the rotated image.")
        else:
            self.status.set("Preprocessing updated; annotations kept.")

    @staticmethod
    def _deskew(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 20:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.1 or abs(angle) > 20:
            return img
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # ==================================================================
    # SIZE INFO & PAGING
    # ==================================================================
    def _update_size_info(self):
        if self.original_image is None:
            return
        img = self.original_image
        h, w = img.shape[:2]
        ch = img.shape[2] if len(img.shape) == 3 else 1
        try:
            size_kb = round(os.path.getsize(self.image_path) / 1024, 2)
        except (OSError, TypeError):
            size_kb = 0
        self.size_labels["file"].config(text=os.path.basename(self.image_path or "-"))
        self.size_labels["type"].config(text="PDF" if self.pdf_doc else "Image")
        self.size_labels["page"].config(
            text=f"{self.current_page + 1} / {self.page_count}" if self.pdf_doc else "-"
        )
        self.size_labels["resolution"].config(text=f"{w} x {h}")
        self.size_labels["width"].config(text=str(w))
        self.size_labels["height"].config(text=str(h))
        self.size_labels["channels"].config(text=str(ch))
        self.size_labels["size_kb"].config(text=str(size_kb))
        self._update_annotation_label()
        if hasattr(self, "doc_id_var") and not self.doc_id_var.get() and self.image_path:
            stem = os.path.splitext(os.path.basename(self.image_path))[0]
            self.doc_id_var.set(stem)
            self.doc_meta["document_id"] = stem

    def _update_page_buttons(self):
        if not self.pdf_doc:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.page_lbl.config(text="Page 1 / 1")
            return
        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_page < self.page_count - 1 else tk.DISABLED)
        self.page_lbl.config(text=f"Page {self.current_page + 1} / {self.page_count}")

    def _stash_current_page(self):
        if self.original_image is None or self.raw_page_image is None:
            return
        if not self.detections and self.current_page not in self.page_ocr_cache:
            return
        rh, rw = self.raw_page_image.shape[:2]
        self.page_ocr_cache[self.current_page] = {
            "detections": self.detections,
            "text": self.text_box.get("1.0", tk.END).strip(),
            "width": int(rw), "height": int(rh),
            "coord_scale": self._coord_scale,
        }

    def prev_page(self):
        if self.pdf_doc and self.current_page > 0:
            self._autosave_pending_new()
            self._stash_current_page()
            self.current_page -= 1
            self._render_pdf_page(self.current_page)

    def next_page(self):
        if self.pdf_doc and self.current_page < self.page_count - 1:
            self._autosave_pending_new()
            self._stash_current_page()
            self.current_page += 1
            self._render_pdf_page(self.current_page)

    def _apply_zoom(self):
        try:
            z = float(self.zoom_var.get())
        except (tk.TclError, ValueError):
            return
        z = max(1.0, min(4.0, z))
        if z == self.pdf_zoom:
            return
        if not self.pdf_doc:
            self.pdf_zoom = z
            return
        self._autosave_pending_new()
        self._stash_current_page()
        factor = z / self.pdf_zoom
        for entry in self.page_ocr_cache.values():
            entry["detections"] = scale_detections(entry["detections"], factor)
            entry["width"] = int(round(entry["width"] * factor))
            entry["height"] = int(round(entry["height"] * factor))
        self.pdf_zoom = z
        self._history = {}
        self._render_pdf_page(self.current_page)
        if self.split_pdf_var.get() and self.page_ocr_cache:
            self._set_dirty(True)
        self.status.set(f"Render zoom {z}x — existing boxes were rescaled "
                        f"(undo history cleared)."
                        + (" Ctrl+S re-renders the page images." if self.split_pdf_var.get() else ""))

    # ==================================================================
    # REBUILD ANNOTATED IMAGE
    # ==================================================================
    def _rebuild_annotated_image(self):
        if self.original_image is None:
            self.annotated_image = None
            return
        img = self.original_image.copy()
        if not self.show_boxes_var.get():
            self.annotated_image = img
            return
        for d in self.detections:
            x, y, w, h = d["bbox"]
            box_color = label_color(canonical_label(d.get("label")))
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), box_color, 2)

        # Underlined line boxes: draw a red bar just below the line.
        for ln in self.lines_by_page.get(self.current_page, []):
            if not ln.get("is_underlined"):
                continue
            x0, y0, x1, y1 = [int(v) for v in ln["bbox"]]
            cv2.line(img, (x0, y1 + 1), (x1, y1 + 1), (0, 0, 255), 2)

        # Thin green rectangles for every detected line (this page).
        for ln in self.lines_by_page.get(self.current_page, []):
            x0, y0, x1, y1 = [int(v) for v in ln["bbox"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 200, 60), 1)

        # If a block is underlined but no line data exists yet, draw the bar
        # along the block bottom.
        if not self.lines_by_page.get(self.current_page):
            for d in self.detections:
                if d.get("is_underlined"):
                    x, y, w, h = [int(v) for v in d["bbox"]]
                    cv2.line(img, (x, y + h + 1), (x + w, y + h + 1), (0, 0, 255), 2)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_label_font(14)
        try:
            sample_bbox = draw.textbbox((0, 0), "Ag", font=font)
            line_h = max(sample_bbox[3] - sample_bbox[1], 14)
        except Exception:
            line_h = 14
        for d in self.detections:
            x, y, w, h = [int(v) for v in d["bbox"]]
            label = canonical_label(d.get("label"))
            display = self._clean_display_text(d, max_len=60)
            flags = ""
            if d.get("is_bold"):       flags += "B"
            if d.get("is_italic"):     flags += "I"
            if d.get("is_underlined"): flags += "U"
            cap = f"[{label}]" + (f"{{{flags}}}" if flags else "") + f" {display}"
            bgr = label_color(label)
            rgb_box = (bgr[2], bgr[1], bgr[0])
            txt_fill = (255, 255, 255) if sum(bgr) < 250 else (0, 0, 0)
            try:
                tb = draw.textbbox((0, 0), cap, font=font)
                tw = tb[2] - tb[0]
                th = max(tb[3] - tb[1], line_h)
            except Exception:
                tw, th = max(len(cap) * 7, 40), line_h
            if y - th - 6 >= 0:
                bg_top, bg_bot, text_y = y - th - 6, y, y - th - 4
            else:
                bg_top, bg_bot, text_y = y, y + th + 6, y + 2
            bg_right = min(x + tw + 6, pil_img.width - 1)
            bg_left = max(0, x)
            if bg_right > bg_left:
                draw.rectangle([bg_left, bg_top, bg_right, bg_bot], fill=rgb_box)
                draw.text((bg_left + 3, text_y), cap, fill=txt_fill, font=font)
        self.annotated_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ==================================================================
    # REFRESH PREVIEW
    # ==================================================================
    def refresh_preview(self):
        self._rebuild_annotated_image()
        self.canvas.delete("all")
        if self.original_image is None:
            return
        display = self.annotated_image if self.annotated_image is not None else self.original_image
        if display is None:
            return
        display = display.copy()
        if self.selected_bbox is not None:
            x, y, w, h = self.selected_bbox
            cv2.rectangle(display, (int(x), int(y)), (int(x + w), int(y + h)),
                          (0, 215, 255), 3)
            for hx, hy in self._get_handles_img(self.selected_bbox):
                cv2.rectangle(display, (int(hx - 4), int(hy - 4)),
                              (int(hx + 4), int(hy + 4)), (0, 215, 255), -1)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih)
        self.preview_scale = scale
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self.tk_preview = ImageTk.PhotoImage(pil_img)
        ox = (cw - new_w) // 2
        oy = (ch - new_h) // 2
        self.preview_offset = (ox, oy)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.tk_preview)

    def _get_handles_img(self, bbox):
        x, y, w, h = bbox
        xm, ym = x + w / 2, y + h / 2
        return [
            (int(x), int(y)), (int(xm), int(y)), (int(x + w), int(y)),
            (int(x), int(ym)), (int(x + w), int(ym)),
            (int(x), int(y + h)), (int(xm), int(y + h)), (int(x + w), int(y + h)),
        ]

    # ==================================================================
    # OCR — backend dispatcher
    # ==================================================================
    def _resolve_backend(self, requested):
        if requested.startswith("Auto"):
            return "auto"
        if requested.startswith("Surya"):
            return "surya"
        if requested.startswith("PaddleStructureV3"):
            return "paddlestructurev3"
        if requested.startswith("PaddleOCR"):
            return "paddleocr"
        if requested.startswith("EasyOCR"):
            return "easyocr"
        if requested.startswith("Tesseract"):
            return "tesseract"
        return "tesseract"

    def _run_backend(self, key, img, lang_str, min_conf):
        if key == "tesseract":
            d, t = self.backends.run_tesseract(img, lang_str, min_conf)
            return d, t, "tesseract"
        if key == "easyocr":
            d, t = self.backends.run_easyocr(img, lang_str, min_conf)
            return d, t, "easyocr"
        if key == "surya":
            d, t = self.backends.run_surya(img, lang_str, min_conf)
            return d, t, "surya"
        if key == "paddleocr":
            d, t = self.backends.run_paddle(img, lang_str, min_conf)
            return d, t, "paddleocr"
        if key == "paddlestructurev3":
            d, t = self.backends.run_paddle_structure(img, lang_str, min_conf)
            if not any(x.get("label") == "table" for x in d):
                d2 = self.backends.reconstruct_tables(d)
                if any(x.get("label") == "table" for x in d2):
                    d = d2
            return d, t, "paddlestructurev3"
        if key == "auto":
            last_err = None
            for candidate in AUTO_ORDER:
                try:
                    return self._run_backend(candidate, img, lang_str, min_conf)
                except Exception as e:
                    last_err = e
                    self.root.after(0, lambda c=candidate, err=e:
                        self.status.set(f"Backend '{c}' failed: {err}. Trying next..."))
                    continue
            raise RuntimeError(f"All auto backends failed. Last error: {last_err}")
        raise ValueError(f"Unknown backend: {key}")

    @staticmethod
    def _normalize_ocr_dets(detections):
        for d in detections:
            d["label"] = canonical_label(d.get("label"))
            d.setdefault("status", "auto")
            t = normalize_text(d.get("text") or "")
            if d["label"] == "table" or _looks_like_html(t):
                t = normalize_table_html(t)
            d["text"] = t
            if "quad" not in d:
                d["quad"] = quad_from_bbox(d["bbox"])
        return detections

    def run_ocr(self):
        if self.original_image is None:
            return
        self._autosave_pending_new()
        self.backends.apply_fixes = self.apply_fixes_var.get()
        self.run_btn.config(state=tk.DISABLED)
        self.status.set("Running OCR on current view...")
        params = dict(
            key=self._resolve_backend(self.backend_var.get()),
            lang=self.lang_var.get(),
            min_conf=self.conf_var.get(),
            preprocessing_active=any([
                self.pp_grayscale.get(), self.pp_denoise.get(),
                self.pp_deskew.get(), self.pp_upscale.get(),
            ]),
            img=self.original_image.copy(),
            page=self.current_page,
            zoom=self.pdf_zoom,
        )
        threading.Thread(target=self._ocr_worker, args=(params,), daemon=True).start()

    def _ocr_worker(self, p):
        try:
            if self.pdf_doc is not None and not p["preprocessing_active"]:
                page = self.pdf_doc.load_page(p["page"])
                pdf_text = page.get_text("text").strip()
                if pdf_text:
                    words = page.get_text("words")
                    detections = []
                    z = p["zoom"]
                    for w in words:
                        x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
                        if not word.strip():
                            continue
                        bb = (int(x0 * z), int(y0 * z),
                              int((x1 - x0) * z), int((y1 - y0) * z))
                        detections.append({
                            "text": word, "conf": 100, "bbox": bb,
                            "quad": quad_from_bbox(bb),
                            "label": "text_line", "order": len(detections) + 1,
                            "status": "auto",
                        })
                    self._normalize_ocr_dets(detections)
                    self.root.after(0, lambda: self._ocr_done(detections, pdf_text,
                                                              "embedded-pdf", p["page"]))
                    return

            detections, full_text, used = self._run_backend(
                p["key"], p["img"], p["lang"], p["min_conf"])
            self._normalize_ocr_dets(detections)
            self.root.after(0, lambda: self._ocr_done(detections, full_text, used, p["page"]))

        except pytesseract.TesseractNotFoundError:
            self.root.after(0, lambda: messagebox.showerror(
                "Tesseract not found",
                "Tesseract OCR is not installed or not on PATH.\n"
                "Install: sudo apt install tesseract-ocr tesseract-ocr-nep"
            ))
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror(
                "OCR Error", f"OCR failed:\n{err}"
            ))
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def _ocr_done(self, detections, full_text, backend_used, page=None):
        self.run_btn.config(state=tk.NORMAL)
        if page is not None and page != self.current_page:
            self.status.set("OCR finished for a page you navigated away from — results discarded.")
            return
        if self.detections:
            self._push_undo()
        self.detections = detections
        # Block boxes changed: any stale line data for this page is invalid.
        self.lines_by_page.pop(self.current_page, None)
        self._rebuild_tree()
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", normalize_text(full_text))
        self._drop_selection()
        self._stash_current_page()
        self._set_dirty(True)
        msg = (f"OCR complete ({backend_used}): "
               f"{len(self.detections)} detections. Ctrl+Z restores the previous boxes.")
        if getattr(self, "auto_lines_var", None) and self.auto_lines_var.get():
            try:
                self.run_line_ocr_page()
                msg += f" Line detection: {len(self.lines_by_page.get(self.current_page, []))} line(s)."
            except Exception as e:
                print(f"[line-ocr auto] {e}")
        self.status.set(msg)

    # ==================================================================
    # OCR ALL PAGES
    # ==================================================================
    def run_all_pages(self):
        if self.pdf_doc is None:
            messagebox.showinfo("No PDF", "Open a PDF first to use 'OCR All Pages'.")
            return
        self._autosave_pending_new()
        self._stash_current_page()
        reviewed = sum(1 for e in self.page_ocr_cache.values()
                       for d in e.get("detections", []) if d.get("status") == "reviewed")
        if reviewed and not messagebox.askyesno(
                "Replace annotations?",
                f"This replaces the boxes on every page, including {reviewed} reviewed "
                f"box(es).\nEach page can be restored afterwards with Ctrl+Z.\n\nContinue?"):
            return
        self.backends.apply_fixes = self.apply_fixes_var.get()
        self.run_all_btn.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.DISABLED)
        self.status.set("Running OCR on all pages... this may take a while.")
        params = dict(
            key=self._resolve_backend(self.backend_var.get()),
            lang=self.lang_var.get(),
            min_conf=self.conf_var.get(),
        )
        threading.Thread(target=self._ocr_all_worker, args=(params,), daemon=True).start()

    def _ocr_all_worker(self, p):
        try:
            all_results = {}
            total = self.pdf_doc.page_count
            for page_num in range(total):
                img = self._render_page_image(page_num)
                detections, full_text, _used = self._run_backend(
                    p["key"], img, p["lang"], p["min_conf"]
                )
                self._normalize_ocr_dets(detections)
                all_results[page_num] = {
                    "detections": detections,
                    "text": normalize_text(full_text),
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "coord_scale": 1.0,
                }
                pct = int((page_num + 1) / total * 100)
                self.root.after(0, lambda p_=pct, pn=page_num + 1, t=total:
                    self.status.set(f"OCR All Pages: page {pn}/{t} ({p_}%)"))

            self.root.after(0, lambda: self._apply_ocr_cache(all_results))

        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror(
                "OCR Error", f"OCR all pages failed:\n{err}"
            ))
            self.root.after(0, lambda: self.run_all_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def _show_cached_page(self, page_key):
        cached = self.page_ocr_cache[page_key]
        cs = cached.get("coord_scale", 1.0)
        if abs(cs - self._coord_scale) > 1e-6:
            cached["detections"] = scale_detections(cached["detections"], self._coord_scale / cs)
            cached["coord_scale"] = self._coord_scale
        self.detections = cached["detections"]
        self._rebuild_tree()
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", cached.get("text", "") or self._text_from_dets(self.detections))
        self.selected_bbox = None
        self.selected_detection_index = None
        self._update_selection_panel()
        self.refresh_preview()

    def _apply_ocr_cache(self, all_results):
        self._stash_current_page()
        for page, _entry in all_results.items():
            old = self.page_ocr_cache.get(page)
            if old and old.get("detections"):
                self._push_undo(snap={"dets": copy.deepcopy(old["detections"]),
                                      "scale": old.get("coord_scale", 1.0)}, page=page)
        self.page_ocr_cache.update(all_results)
        # Blocks changed on every page: clear stale line data for all of them.
        self.lines_by_page = {}
        self._set_dirty(True)
        self.run_all_btn.config(state=tk.NORMAL)
        self.run_btn.config(state=tk.NORMAL)
        if self.current_page in self.page_ocr_cache:
            self._show_cached_page(self.current_page)
        self._update_undo_buttons()
        self.status.set(f"OCR All Pages complete. Showing page {self.current_page + 1}.")

    # ==================================================================
    # EXPORT HELPERS — everything on disk is in raw-render coordinates
    # ==================================================================
    def _page_entries(self, current_only=False):
        self._stash_current_page()
        keys = [self.current_page] if current_only else sorted(self.page_ocr_cache)
        entries = []
        for p in keys:
            e = self.page_ocr_cache.get(p)
            if e is None:
                continue
            cs = e.get("coord_scale", 1.0) or 1.0
            dets = e.get("detections", [])
            disk = scale_detections(dets, 1.0 / cs) if abs(cs - 1.0) > 1e-6 \
                else [dict(d) for d in dets]
            entries.append((p, disk, int(e.get("width", 0)), int(e.get("height", 0))))
        return entries

    def _page_png_name(self, page):
        return split_page_stem(self.image_path, page) + ".png"

    def _write_page_png(self, page, img_dir):
        try:
            os.makedirs(img_dir, exist_ok=True)
            name = self._page_png_name(page)
            out = os.path.join(img_dir, name)
            if self.pdf_doc is not None:
                pix = self.pdf_doc.load_page(page).get_pixmap(
                    matrix=fitz.Matrix(self.pdf_zoom, self.pdf_zoom), alpha=False)
                pix.save(out)
            else:
                ok, buf = cv2.imencode(".png", self.raw_page_image)
                if not ok:
                    raise RuntimeError("cv2.imencode failed")
                buf.tofile(out)
            return name
        except Exception as e:
            print(f"[export] could not render page {page}: {e}")
            return None

    def _mark_saved(self, path, kind):
        self._annotation_path = path
        self._annotation_kind = kind
        self._set_dirty(False)
        if path and os.path.isfile(path):
            self._disk_sig[os.path.abspath(path)] = file_sig(path)

    def _ask_path(self, path, ext, types):
        if path:
            return path
        return filedialog.asksaveasfilename(defaultextension=ext, filetypes=types)

    # ==================================================================
    # QUICK SAVE (Ctrl+S)
    # ==================================================================
    def _sidecar_path(self):
        return os.path.splitext(self.image_path)[0] + "_annotations.jsonl"

    def _clean_all_pages(self):
        self._stash_current_page()
        changed = 0
        for page, entry in self.page_ocr_cache.items():
            dets = entry.get("detections") or []
            snap = copy.deepcopy(dets)
            n = 0
            leftovers = find_leftover_boxes(dets)
            if leftovers:
                drop = set(leftovers)
                dets[:] = [d for i, d in enumerate(dets) if i not in drop]
                n += len(leftovers)
                self._removed_leftovers = getattr(self, "_removed_leftovers", 0) + len(leftovers)
            for d in dets:
                text, label, fixes = clean_block_content(d.get("text", ""), d.get("label", DEFAULT_LABEL))
                hit = False
                if fixes or text != d.get("text") or label != d.get("label"):
                    d["text"], d["label"] = text, label
                    hit = True
                if label in NON_FLOW_LABELS and d.get("order") is not None:
                    d["order"] = None
                    hit = True
                n += hit
            if n:
                self._push_undo(snap={"dets": snap, "scale": entry.get("coord_scale", 1.0)},
                                page=page, mark_dirty=False)
                changed += n
        if changed and self.current_page in self.page_ocr_cache:
            self.detections = self.page_ocr_cache[self.current_page]["detections"]
            self.selected_detection_index = None
            self.selected_bbox = None
            self._rebuild_tree()
            self._update_selection_panel()
            self.refresh_preview()
        return changed

    def _qa_summary(self, entries):
        lines = []
        for p, dets, _w, _h in entries:
            for w in qa_page_warnings(dets):
                lines.append(f"page {p + 1}: {w}")
        for ln in lines:
            print(f"[qa] {ln}")
        return len(lines), lines

    def quick_save(self):
        if not self.image_path or self.original_image is None:
            return False
        self._autosave_pending_new()
        self._reloaded_external = 0
        if self._page_file_pages():
            if not self._sync_external_changes():
                self.status.set("Save cancelled — nothing was written.")
                return False
        elif not self._check_whole_file_external():
            return False
        self._removed_leftovers = 0
        self._last_autofix = self._clean_all_pages()
        p, kind = self._annotation_path, self._annotation_kind
        if self._split_mode():
            return self._save_split_pages()
        if p and kind == "split_page" and self._split_meta:
            return self._save_single_split_page()
        if p and kind == "annotations" and p.lower().endswith(".json"):
            return bool(self.export_annotations_json(path=p, write_images=False, quiet=True))
        if p and kind == "annotations" and p.lower().endswith(".jsonl"):
            return bool(self.export_jsonl(path=p, write_images=False, quiet=True))
        if p and kind == "rich" and p.lower().endswith(".json"):
            return bool(self.export_rich_json(path=p, write_images=False, quiet=True))
        if p and kind == "rich" and p.lower().endswith(".jsonl"):
            return bool(self.export_rich_jsonl(path=p, write_images=False, quiet=True))
        return bool(self.export_jsonl(path=self._sidecar_path(), write_images=False, quiet=True))

    # ==================================================================
    # PADDLEOCR-VL JSON / JSONL
    # ==================================================================
    def _make_paddle_payload(self, detections, w_img, h_img,
                             page_index, page_count):
        parsing_res_list = []
        layout_boxes = []
        for i, d in enumerate(detections):
            x, y, w, h = d["bbox"]
            x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
            label = canonical_label(d.get("label"))
            order = d.get("order")
            content = d.get("text", "") or ""
            polygon = d.get("quad") or [
                [float(x0), float(y0)], [float(x1), float(y0)],
                [float(x1), float(y1)], [float(x0), float(y1)],
            ]
            entry = {
                "block_label": label,
                "block_content": content,
                "block_bbox": [x0, y0, x1, y1],
                "block_id": i,
                "block_order": order,
                "group_id": i,
                "block_polygon_points": polygon,
                "status": d.get("status", "auto"),
            }
            for k in ("is_underlined", "is_bold", "is_italic"):
                if d.get(k):
                    entry[k] = True
            parsing_res_list.append(entry)
            conf = d.get("conf", 100)
            score = 1.0 if conf >= 100 else max(0.0, min(1.0, conf / 100.0))
            layout_boxes.append({
                "cls_id": None,
                "label": label,
                "score": score,
                "coordinate": [x0, y0, x1, y1],
                "order": order,
                "polygon_points": polygon,
            })
        return {
            "input_path": self.image_path,
            "page_index": page_index,
            "page_count": page_count,
            "width": int(w_img),
            "height": int(h_img),
            "render_zoom": self.pdf_zoom if self.pdf_doc else 1.0,
            "model_settings": DEFAULT_MODEL_SETTINGS,
            "parsing_res_list": parsing_res_list,
            "layout_det_res": {
                "input_path": None,
                "page_index": None,
                "boxes": layout_boxes,
            },
        }

    def export_paddle_json(self, path=None):
        if self.original_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return False
        entries = self._page_entries(current_only=True)
        if not entries or not entries[0][1]:
            messagebox.showinfo("No data", "No detections to export.")
            return False
        path = self._ask_path(path, ".json", [("JSON", "*.json")])
        if not path:
            return False
        p, dets, w, h = entries[0]
        payload = self._make_paddle_payload(
            dets, w, h,
            p if self.pdf_doc else None,
            self.page_count if self.pdf_doc else None,
        )
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)
            self.status.set(f"Exported PaddleOCR-VL JSON: {path}")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write JSON:\n{e}")
            return False

    def export_paddle_jsonl(self, path=None):
        if self.original_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return False
        entries = self._page_entries()
        if not entries:
            messagebox.showinfo("No data", "No detections to export.")
            return False
        path = self._ask_path(path, ".jsonl",
                              [("JSON Lines", "*.jsonl"), ("All files", "*.*")])
        if not path:
            return False
        page_count = self.page_count if self.pdf_doc else None
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for p, dets, w, h in entries:
                    payload = self._make_paddle_payload(
                        dets, w, h, p if self.pdf_doc else None, page_count)
                    f.write(json.dumps(payload, ensure_ascii=False,
                                       separators=(",", ":")))
                    f.write("\n")
            self.status.set(
                f"Exported PaddleOCR-VL JSONL: {path} ({len(entries)} page line(s))"
            )
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write JSONL:\n{e}")
            return False

    # ==================================================================
    # OTHER EXPORTS
    # ==================================================================
    def save_annotated(self):
        if self.annotated_image is None:
            messagebox.showinfo("Nothing to save", "Run OCR first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if path:
            ext = os.path.splitext(path)[1] or ".png"
            ok, buf = cv2.imencode(ext, self.annotated_image)
            if ok:
                try:
                    buf.tofile(path)
                    self.status.set(f"Saved annotated image: {path}")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to write image:\n{e}")
            else:
                messagebox.showerror("Save Error", "cv2.imencode failed.")

    def copy_text(self):
        text = self.text_box.get("1.0", tk.END)
        if not text.strip():
            messagebox.showinfo("No text", "No OCR text to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.set("Text copied to clipboard.")

    def export_csv(self, path=None):
        entries = self._page_entries(current_only=True)
        if not entries or not entries[0][1]:
            messagebox.showinfo("No data", "No detections to export.")
            return False
        path = self._ask_path(path, ".csv", [("CSV", "*.csv")])
        if not path:
            return False
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["label", "order", "status",
                                 "is_underlined", "is_bold", "is_italic",
                                 "text", "ocr_text",
                                 "confidence", "x", "y", "width", "height"])
                for d in entries[0][1]:
                    x, y, w, h = d["bbox"]
                    writer.writerow([
                        canonical_label(d.get("label")),
                        "" if d.get("order") is None else d["order"],
                        d.get("status", ""),
                        "1" if d.get("is_underlined") else "0",
                        "1" if d.get("is_bold") else "0",
                        "1" if d.get("is_italic") else "0",
                        d["text"], d.get("ocr_text", ""),
                        d.get("conf", 100), x, y, w, h,
                    ])
            self.status.set(f"Exported CSV: {path}")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write CSV:\n{e}")
            return False

    def export_json(self, path=None):
        entries = self._page_entries(current_only=True)
        if not entries or not entries[0][1]:
            messagebox.showinfo("No data", "No detections to export.")
            return False
        path = self._ask_path(path, ".json", [("JSON", "*.json")])
        if not path:
            return False
        export_data = []
        for d in entries[0][1]:
            x, y, w, h = d["bbox"]
            entry = {
                "text": d["text"],
                "confidence": d.get("conf", 100),
                "label": canonical_label(d.get("label")),
                "order": d.get("order"),
                "status": d.get("status"),
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "quad": d.get("quad"),
            }
            if d.get("is_underlined"):
                entry["is_underlined"] = True
            if d.get("is_bold"):
                entry["is_bold"] = True
            if d.get("is_italic"):
                entry["is_italic"] = True
            export_data.append(entry)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            self.status.set(f"Exported simple JSON: {path}")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write JSON:\n{e}")
            return False

    def _annotation_records(self, entries, export_dir, write_images):
        src = os.path.abspath(self.image_path) if self.image_path else ""
        try:
            image_ref = os.path.relpath(src, export_dir).replace(os.sep, "/")
        except ValueError:
            image_ref = src
        is_pdf = self.pdf_doc is not None
        zoom = self.pdf_zoom if is_pdf else 1.0
        records = []
        for p, dets, w, h in entries:
            rec = {
                "image": image_ref,
                "page": (p + 1) if is_pdf else None,
                "width": w,
                "height": h,
                "render_zoom": zoom,
            }
            if is_pdf:
                img_dir = os.path.join(export_dir, "images")
                name = (self._write_page_png(p, img_dir) if write_images
                        else self._page_png_name(p))
                if name and os.path.isfile(os.path.join(img_dir, name)):
                    rec["page_image"] = "images/" + name
            rec["blocks"] = self._blocks_payload(dets)
            rec["lines"] = self._lines_payload(dets, rec["blocks"], page=p)
            records.append(rec)
        return records

    def _blocks_payload(self, dets):
        keep_ocr = getattr(self, "keep_ocr_var", None) is None or self.keep_ocr_var.get()
        blocks = []
        for i, d in enumerate(dets):
            x, y, bw, bh = d["bbox"]
            x0, y0, x1, y1 = int(x), int(y), int(x + bw), int(y + bh)
            text, label, _fixes = clean_block_content(d.get("text", ""), d.get("label", DEFAULT_LABEL))
            order = None if label in NON_FLOW_LABELS else d.get("order")
            b = {
                "block_id": i,
                "block_label": label,
                "block_content": text,
                "block_bbox": [x0, y0, x1, y1],
                "quad": d.get("quad") or quad_from_bbox((x0, y0, x1 - x0, y1 - y0)),
                "block_order": order,
                "reading_order": order,
                "type": label,
                "confidence": d.get("conf", 100),
                "status": d.get("status", "auto"),
            }
            if label in NON_FLOW_LABELS:
                b["ignore_in_eval"] = True
            if "font_id" in d:
                b["font_id"] = d["font_id"]
            if "font_size" in d:
                b["font_size"] = d["font_size"]
            if "is_heading" in d:
                b["is_heading"] = d["is_heading"]
            if "table_index" in d:
                b["table_index"] = d["table_index"]
            if "cell_id" in d:
                b["cell_id"] = d["cell_id"]
            if d.get("is_underlined"):
                b["is_underlined"] = True
            if d.get("is_bold"):
                b["is_bold"] = True
            if d.get("is_italic"):
                b["is_italic"] = True
            if keep_ocr and "ocr_text" in d:
                b["ocr_text"] = d["ocr_text"]
            blocks.append(b)
        return blocks

    # ==================================================================
    # RICH PAGE EXPORT (synthetic-style schema)
    # ==================================================================
    def _lines_payload(self, dets, blocks, page=None):
        """
        Prefer real line detections from self.lines_by_page (produced by
        run_line_ocr_page). Fall back to newline-splitting with proportional
        bboxes if none were computed for this page.
        """
        if page is None:
            page = self.current_page
        stored = self.lines_by_page.get(page)
        if stored:
            return stored
        out = []
        for bid, d in enumerate(dets):
            self._emit_lines_for_block(bid, d, out)
        return out

    def _layout_payload(self, dets):
        ordered = sorted(enumerate(dets), key=lambda t: _order_key(t[1], t[0]))
        out = []
        for i, (idx, d) in enumerate(ordered):
            x, y, w, h = d["bbox"]
            entry = {
                "id": idx,
                "type": canonical_label(d.get("label")),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "reading_order": i,
                "text": d.get("text", ""),
                "quad": d.get("quad") or quad_from_bbox(d["bbox"]),
                "table_index": d.get("table_index"),
                "list_style": d.get("list_style"),
            }
            if d.get("is_underlined"):
                entry["is_underlined"] = True
            out.append(entry)
        return out

    def _tables_payload(self, dets):
        out = []
        t_index = 0
        for d in dets:
            label = canonical_label(d.get("label"))
            text = d.get("text") or ""
            if label != "table" and not _looks_like_html(text):
                continue
            x, y, w, h = d["bbox"]
            num_rows, num_cols, header_rows, cells = parse_table_cells(text)
            col_types = guess_col_types(cells, num_cols, header_rows)
            rendered_cells = []
            if num_rows and num_cols:
                cw = w / num_cols
                ch = h / max(1, num_rows)
                for c in cells:
                    cx0 = int(x + c["col"] * cw)
                    cy0 = int(y + c["row"] * ch)
                    cx1 = int(x + (c["col"] + c["colspan"]) * cw)
                    cy1 = int(y + (c["row"] + c["rowspan"]) * ch)
                    rendered_cells.append({
                        "id": len(rendered_cells),
                        "row": c["row"],
                        "col": c["col"],
                        "rowspan": c["rowspan"],
                        "colspan": c["colspan"],
                        "bbox": [cx0, cy0, cx1, cy1],
                        "text": c.get("text", ""),
                        "is_header": bool(c.get("is_header")),
                        "cell_type": c.get("cell_type", "data"),
                        "empty": bool(c.get("empty", True)),
                        "quad": quad_from_bbox((cx0, cy0, cx1 - cx0, cy1 - cy0)),
                    })
            out.append({
                "table_index": t_index,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "header_rows": header_rows,
                "ruling_style": "full",
                "theme": "generic",
                "col_types": col_types,
                "reading_order": list(range(len(rendered_cells))),
                "html": text,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "quad": d.get("quad") or quad_from_bbox(d["bbox"]),
                "cells": rendered_cells,
            })
            t_index += 1
        return out

    def _fonts_payload(self, dets):
        fonts, font_ids = [], []
        seen = {}
        for i, d in enumerate(dets):
            fname = d.get("font_name") or d.get("font") or None
            fid = d.get("font_id")
            if fname and fname not in seen:
                seen[fname] = len(fonts)
                fonts.append(fname)
                font_ids.append(seen[fname])
            elif fid is not None:
                if fid not in font_ids:
                    font_ids.append(fid)
        return fonts, font_ids

    def _quality_payload(self, dets, w_img, h_img):
        text = " ".join((d.get("text") or "") for d in dets)
        pages_meta = self.rich_page_meta.get(self.current_page, {}) or {}
        q = pages_meta.get("quality") or {}
        lines = self.lines_by_page.get(self.current_page, []) or []
        quality = {
            "skew_deg": q.get("skew_deg", 0.0),
            "perspective": q.get("perspective", False),
            "is_clean": q.get("is_clean", True),
            "is_binarized": q.get("is_binarized", False),
            "is_grayscale": q.get("is_grayscale", False),
            "has_fold": q.get("has_fold", False),
            "has_bleedthrough": q.get("has_bleedthrough", False),
            "ops": q.get("ops", []),
            "downscale": q.get("downscale", 1.0),
            "noise_sigma": q.get("noise_sigma", 0.0),
            "text_contrast": q.get("text_contrast", 0.0),
            "has_stamp": q.get("has_stamp", any(
                canonical_label(d.get("label")) in ("stamp", "seal") for d in dets)),
            "has_signature": q.get("has_signature", any(
                canonical_label(d.get("label")) == "signature" for d in dets)),
            "jpeg_quality": q.get("jpeg_quality", 100),
            "line_count": q.get("line_count", len(lines) if lines else len(dets)),
            "table_count": q.get("table_count", sum(
                1 for d in dets if canonical_label(d.get("label")) == "table")),
            "figure_count": q.get("figure_count", sum(
                1 for d in dets if canonical_label(d.get("label")) in
                ("image", "photo", "diagram", "chart", "map", "illustration"))),
            "underlined_lines": q.get("underlined_lines", sum(
                1 for ln in lines if ln.get("is_underlined"))),
            "stamp_bboxes": q.get("stamp_bboxes", [
                list(d["bbox"]) for d in dets
                if canonical_label(d.get("label")) in ("stamp", "seal")]),
            "signature_bboxes": q.get("signature_bboxes", [
                list(d["bbox"]) for d in dets
                if canonical_label(d.get("label")) == "signature"]),
            "raw_lengths": q.get("raw_lengths", {
                "paragraph_chars": len(text),
                "list_items": sum(1 for d in dets
                                  if canonical_label(d.get("label")) == "list_item"),
                "form_fields": sum(1 for d in dets
                                   if canonical_label(d.get("label")) == "form_field"),
                "table_cells": sum(
                    len(parse_table_cells(d.get("text") or "")[3])
                    for d in dets if canonical_label(d.get("label")) == "table"),
            }),
        }
        return quality

    def _document_payload(self):
        if hasattr(self, "doc_id_var"):
            self.doc_meta["document_id"] = (
                self.doc_id_var.get().strip()
                or self.doc_meta.get("document_id") or "")
        if hasattr(self, "doc_lang_var"):
            self.doc_meta["language"] = (
                self.doc_lang_var.get()
                or self.doc_meta.get("language") or "ne")
        if hasattr(self, "doc_digit_var"):
            self.doc_meta["digit_style"] = (
                self.doc_digit_var.get()
                or self.doc_meta.get("digit_style") or "dev")
        return dict(self.doc_meta)

    def _make_rich_record(self, page, dets, w_img, h_img, page_count,
                          write_images=False, export_dir=None):
        """Assemble a full synthetic-style page record."""
        doc = self._document_payload()
        page_image_name = None
        if export_dir and write_images:
            img_dir = os.path.join(export_dir, "images")
            page_image_name = self._write_page_png(page, img_dir)
            if page_image_name:
                page_image_name = "images/" + page_image_name

        blocks = self._blocks_payload(dets)
        lines = self._lines_payload(dets, blocks, page=page)
        tables = self._tables_payload(dets)
        fonts, font_ids = self._fonts_payload(dets)
        quality = self._quality_payload(dets, w_img, h_img)
        layout = self._layout_payload(dets)

        prev = self.rich_page_meta.get(page, {}) or {}
        if prev.get("fonts"):
            fonts = prev["fonts"]
        if prev.get("font_ids"):
            font_ids = prev["font_ids"]
        if prev.get("quality"):
            for k, v in prev["quality"].items():
                quality.setdefault(k, v)

        text = prev.get("text") or self._text_from_dets(dets)

        rec = {
            "v": doc.get("v", 2),
            "image": os.path.basename(self.image_path) if self.image_path else "",
            "width": int(w_img),
            "height": int(h_img),
            "document_id": doc.get("document_id") or None,
            "page_number": (page + 1) if self.pdf_doc is not None else 1,
            "page_count": int(page_count) if self.pdf_doc is not None else 1,
            "language": doc.get("language") or "ne",
            "digit_style": doc.get("digit_style") or "dev",
            "text": text,
            "fonts": fonts,
            "font_ids": font_ids,
            "layout": layout,
            "blocks": blocks,
            "lines": lines,
            "tables": tables,
            "ruling_lines": [],
            "quality": quality,
        }
        if page_image_name:
            rec["page_image"] = page_image_name
        return rec

    def export_rich_json(self, path=None, write_images=None, quiet=False):
        if self.original_image is None:
            if not quiet:
                messagebox.showinfo("No image", "Open an image/PDF first.")
            return False
        entries = self._page_entries(current_only=True)
        if not entries or not entries[0][1]:
            if not quiet:
                messagebox.showinfo("No data", "No detections to export.")
            return False
        path = self._ask_path(path, ".json", [("JSON", "*.json")])
        if not path:
            return False
        if write_images is None:
            write_images = self.export_images_var.get()
        p, dets, w, h = entries[0]
        export_dir = os.path.dirname(os.path.abspath(path))
        rec = self._make_rich_record(p, dets, w, h,
                                     self.page_count if self.pdf_doc else 1,
                                     write_images=write_images,
                                     export_dir=export_dir)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            self._mark_saved(path, "rich")
            self.status.set(f"Exported rich page JSON: {path} "
                            f"({len(rec['blocks'])} blocks, {len(rec['lines'])} lines, "
                            f"{len(rec['tables'])} tables)")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write rich JSON:\n{e}")
            return False

    def export_rich_jsonl(self, path=None, write_images=None, quiet=False):
        if self.original_image is None:
            if not quiet:
                messagebox.showinfo("No image", "Open an image/PDF first.")
            return False
        entries = self._page_entries()
        if not entries:
            if not quiet:
                messagebox.showinfo("No data", "No detections to export.")
            return False
        path = self._ask_path(path, ".jsonl",
                              [("JSON Lines", "*.jsonl"), ("All files", "*.*")])
        if not path:
            return False
        if write_images is None:
            write_images = self.export_images_var.get()
        export_dir = os.path.dirname(os.path.abspath(path))
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for p, dets, w, h in entries:
                    rec = self._make_rich_record(p, dets, w, h, self.page_count,
                                                 write_images=write_images,
                                                 export_dir=export_dir)
                    f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
                    f.write("\n")
            self._mark_saved(path, "rich")
            self.status.set(f"Exported rich page JSONL: {path} ({len(entries)} page line(s))")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write rich JSONL:\n{e}")
            return False

    # ==================================================================
    # SPLIT PDF: one PNG + one annotation JSON per page, linked to the PDF
    # ==================================================================
    def _adopt_manifest_zoom(self):
        m = os.path.join(split_pages_dir(self.image_path), SPLIT_MANIFEST)
        try:
            with open(m, encoding="utf-8") as f:
                z = float(json.load(f).get("render_zoom"))
        except Exception:
            return
        if 1.0 <= z <= 4.0 and abs(z - self.pdf_zoom) > 1e-6:
            self.pdf_zoom = z
            self.zoom_var.set(z)

    def _split_pdf_pages(self):
        if not self.image_path or (self.pdf_doc is None and self.raw_page_image is None):
            return None
        d = split_pages_dir(self.image_path)
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            self.status.set(f"Could not create page folder {d}: {e}")
            return None
        if self.pdf_doc is None:
            return self._split_image_page(d)
        mat = fitz.Matrix(self.pdf_zoom, self.pdf_zoom)
        pages, written = [], 0
        for p in range(self.page_count):
            name = self._page_png_name(p)
            out = os.path.join(d, name)
            try:
                page = self.pdf_doc.load_page(p)
                ir = (page.rect * mat).irect
                need = True
                if os.path.isfile(out):
                    try:
                        with Image.open(out) as im:
                            w, h = im.size
                        need = abs(w - ir.width) > 1 or abs(h - ir.height) > 1
                    except Exception:
                        need = True
                if need:
                    page.get_pixmap(matrix=mat, alpha=False).save(out)
                    written += 1
            except Exception as e:
                print(f"[ocr-tool] could not render page {p + 1}: {e}")
                continue
            pages.append({"page": p + 1, "image": name,
                          "annotation": split_page_stem(self.image_path, p) + ".json"})
        manifest = {
            "source_pdf": self._rel(self.image_path, d),
            "page_count": self.page_count,
            "render_zoom": self.pdf_zoom,
            "pages": pages,
        }
        try:
            with open(os.path.join(d, SPLIT_MANIFEST), "w", encoding="utf-8", newline="\n") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[ocr-tool] could not write manifest: {e}")
        return d, written

    def _split_image_page(self, d):
        name = self._page_png_name(0)
        out = os.path.join(d, name)
        written = 0
        h, w = self.raw_page_image.shape[:2]
        need = True
        if os.path.isfile(out):
            try:
                with Image.open(out) as im:
                    need = im.size != (w, h)
            except Exception:
                need = True
        if need:
            try:
                if self.image_path.lower().endswith(".png"):
                    import shutil
                    shutil.copy2(self.image_path, out)
                else:
                    ok, buf = cv2.imencode(".png", self.raw_page_image)
                    if not ok:
                        raise RuntimeError("cv2.imencode failed")
                    buf.tofile(out)
                written = 1
            except Exception as e:
                print(f"[ocr-tool] could not write page image: {e}")
                return None
        manifest = {
            "source_image": self._rel(self.image_path, d),
            "page_count": 1,
            "render_zoom": 1.0,
            "pages": [{"page": 1, "image": name,
                       "annotation": split_page_stem(self.image_path, 0) + ".json"}],
        }
        try:
            with open(os.path.join(d, SPLIT_MANIFEST), "w", encoding="utf-8", newline="\n") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[ocr-tool] could not write manifest: {e}")
        return d, written

    @staticmethod
    def _rel(path, start):
        try:
            return os.path.relpath(os.path.abspath(path), start).replace(os.sep, "/")
        except ValueError:
            return os.path.abspath(path)

    def _autoload_split_pages(self):
        if not self.autoload_var.get() or not self._split_mode():
            return False
        d = split_pages_dir(self.image_path)
        found, all_blocks = 0, []
        for p in range(self.page_count):
            path = os.path.join(d, split_page_stem(self.image_path, p) + ".json")
            if not os.path.isfile(path):
                continue
            try:
                blocks, _k = load_annotation_blocks(path)
            except Exception as e:
                print(f"[ocr-tool] skipping {path}: {e}")
                continue
            found += 1
            for b in blocks:
                b["page"] = p
                b["image"] = None
            all_blocks.extend(blocks)
        if not found:
            return False
        n, _skipped = self._apply_loaded_blocks(all_blocks, strict=False)
        self._annotation_path = d
        self._annotation_kind = "split"
        self._set_dirty(False)
        self._record_disk_state()
        self.status.set(f"Loaded {n} box(es) from {found} page file(s) in "
                        f"{os.path.basename(d)}/.")
        return True

    def _split_page_record(self, page, dets, w, h, pages_dir):
        is_pdf = self.pdf_doc is not None
        return {
            "image": self._page_png_name(page),
            ("source_pdf" if is_pdf else "source_image"): self._rel(self.image_path, pages_dir),
            "page": page + 1,
            "page_count": self.page_count,
            "width": w,
            "height": h,
            "render_zoom": self.pdf_zoom if is_pdf else 1.0,
            "blocks": self._blocks_payload(dets),
            "lines": self._lines_payload(dets, [], page=page),
        }

    def _save_split_pages(self):
        res = self._split_pdf_pages()
        if not res:
            messagebox.showerror("Save Error", "Could not write the page images.")
            return False
        d, _written = res
        entries = self._page_entries()
        if not entries:
            self.status.set("Nothing to save yet — draw boxes or run OCR first.")
            return False
        nbox, written, unchanged = 0, 0, 0
        try:
            for p, dets, w, h in entries:
                rec = self._split_page_record(p, dets, w, h, d)
                path = os.path.join(d, split_page_stem(self.image_path, p) + ".json")
                rs = self._record_str(rec)
                exists = os.path.isfile(path)
                if exists and rs == self._baseline.get(p):
                    unchanged += 1
                    continue
                if not dets and not exists:
                    continue
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                self._disk_sig[os.path.abspath(path)] = file_sig(path)
                self._baseline[p] = rs
                written += 1
                nbox += len(dets)
        except OSError as e:
            messagebox.showerror("Save Error", f"Failed to write page annotations:\n{e}")
            return False
        self._mark_saved(d, "split")
        msg = f"Saved {written} changed page(s) ({nbox} boxes) to {os.path.basename(d)}/"
        if unchanged:
            msg += f", {unchanged} unchanged page(s) not rewritten"
        if self._reloaded_external:
            msg += f", {self._reloaded_external} page(s) reloaded from edited files"
        self.status.set(msg + self._save_suffix(entries))
        return True

    def _save_suffix(self, entries):
        parts = []
        fixed = getattr(self, "_last_autofix", 0)
        self._last_autofix = 0
        removed = getattr(self, "_removed_leftovers", 0)
        self._removed_leftovers = 0
        if removed:
            parts.append(f"removed {removed} leftover OCR box(es)")
            fixed = max(0, fixed - removed)
        if fixed:
            parts.append(f"auto-fixed {fixed} box(es)")
        n, _lines = self._qa_summary(entries)
        if n:
            parts.append(f"{n} QA warning(s) - click Check Page")
        return (" — " + " · ".join(parts)) if parts else " — QA clean"

    def _save_single_split_page(self):
        entries = self._page_entries(current_only=True)
        if not entries:
            return False
        _p, dets, w, h = entries[0]
        m = self._split_meta
        rec = self._single_page_record(dets, w, h)
        stem = os.path.splitext(os.path.abspath(self.image_path))[0]
        path = stem + ".json"
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        except OSError as e:
            messagebox.showerror("Save Error", f"Failed to write:\n{e}")
            return False
        self._baseline[0] = self._record_str(rec)
        self._mark_saved(path, "split_page")
        src = os.path.basename(m.get("source") or "") or "source"
        self.status.set(f"Saved {len(dets)} box(es) to {os.path.basename(path)} "
                        f"(page {m['page']} of {src})" + self._save_suffix(entries))
        return True

    def _single_page_record(self, dets, w, h):
        m = self._split_meta or {}
        return {
            "image": os.path.basename(self.image_path),
            m.get("source_key", "source_pdf"): m.get("source", ""),
            "page": m.get("page", 1),
            "page_count": m.get("page_count"),
            "width": w,
            "height": h,
            "render_zoom": m.get("render_zoom") or 1.0,
            "blocks": self._blocks_payload(dets),
            "lines": self._lines_payload(dets, [], page=0),
        }

    # ==================================================================
    # SYNC WITH JSON FILES EDITED OUTSIDE THE APP
    # ==================================================================
    @staticmethod
    def _record_str(rec):
        return json.dumps(rec, ensure_ascii=False, sort_keys=True) if rec is not None else None

    def _page_file_pages(self):
        if not self.image_path:
            return []
        if self._split_mode():
            return list(range(self.page_count))
        if self._annotation_kind == "split_page" and self._split_meta:
            return [0]
        return []

    def _page_file_path(self, page):
        if not self.image_path:
            return None
        if self._split_mode():
            return os.path.abspath(os.path.join(split_pages_dir(self.image_path),
                                                split_page_stem(self.image_path, page) + ".json"))
        if self._annotation_kind == "split_page" and self._split_meta and page == 0:
            return os.path.splitext(os.path.abspath(self.image_path))[0] + ".json"
        return None

    def _entry_for(self, page):
        e = self.page_ocr_cache.get(page)
        if e is None:
            return None
        cs = e.get("coord_scale", 1.0) or 1.0
        dets = e.get("detections", [])
        disk = scale_detections(dets, 1.0 / cs) if abs(cs - 1.0) > 1e-6 else [dict(d) for d in dets]
        return disk, int(e.get("width", 0)), int(e.get("height", 0))

    def _record_for(self, page):
        ent = self._entry_for(page)
        if ent is None:
            return None
        dets, w, h = ent
        if self._split_mode():
            return self._split_page_record(page, dets, w, h, split_pages_dir(self.image_path))
        if self._annotation_kind == "split_page" and self._split_meta:
            return self._single_page_record(dets, w, h)
        return None

    def _app_changed(self, page):
        rec = self._record_for(page)
        if rec is None:
            return False
        if not rec.get("blocks") and self._baseline.get(page) is None:
            return False
        return self._record_str(rec) != self._baseline.get(page)

    def _record_disk_state(self):
        for p in self._page_file_pages():
            path = self._page_file_path(p)
            if path and os.path.isfile(path):
                self._disk_sig[path] = file_sig(path)
            rec = self._record_for(p)
            self._baseline[p] = self._record_str(rec) if rec is not None else None

    def _reload_page_file(self, page, path):
        try:
            blocks, _k = load_annotation_blocks(path)
        except Exception as e:
            messagebox.showerror("Reload Error", f"Could not read {os.path.basename(path)}:\n{e}")
            return False
        for b in blocks:
            b["page"] = page
            b["image"] = None
        if blocks:
            self._apply_loaded_blocks(blocks, strict=False)
        else:
            self._stash_current_page()
            old = self.page_ocr_cache.get(page)
            if old and old.get("detections"):
                self._push_undo(snap={"dets": copy.deepcopy(old["detections"]),
                                      "scale": old.get("coord_scale", 1.0)},
                                page=page, mark_dirty=False)
            w, h = self._raw_page_size(page)
            self.page_ocr_cache[page] = {"detections": [], "text": "", "width": w,
                                         "height": h, "coord_scale": 1.0}
            if page == self.current_page:
                self._show_cached_page(page)
        self._disk_sig[path] = file_sig(path)
        self._conflict_seen.pop(path, None)
        rec = self._record_for(page)
        self._baseline[page] = self._record_str(rec) if rec is not None else None
        self._update_undo_buttons()
        return True

    def _sync_external_changes(self):
        self._stash_current_page()
        reload, conflicts = [], []
        for p in self._page_file_pages():
            path = self._page_file_path(p)
            if not path:
                continue
            cur = file_sig(path)
            if cur == self._disk_sig.get(path):
                continue
            if cur is None:
                self._disk_sig.pop(path, None)
                self._baseline.pop(p, None)
                continue
            (conflicts if self._app_changed(p) else reload).append((p, path, cur))
        for p, path, _cur in reload:
            self._reload_page_file(p, path)
        self._reloaded_external = len(reload)
        if conflicts:
            pages = ", ".join(str(p + 1) for p, _a, _b in conflicts[:15])
            ans = messagebox.askyesnocancel(
                "JSON changed outside the app",
                f"Page(s) {pages}: the JSON file was edited outside the app, "
                "and you also changed these pages here.\n\n"
                "Yes — keep the app version (overwrite those files)\n"
                "No — load the edited files (discard app changes on those pages)\n"
                "Cancel — don't save anything")
            if ans is None:
                return False
            for p, path, cur in conflicts:
                if ans:
                    self._disk_sig[path] = cur
                else:
                    self._reload_page_file(p, path)
                    self._reloaded_external += 1
        return True

    def _check_whole_file_external(self):
        path = self._annotation_path
        if not path or not os.path.isfile(path):
            return True
        key = os.path.abspath(path)
        known = self._disk_sig.get(key)
        if known is None or file_sig(path) == known:
            return True
        ans = messagebox.askyesnocancel(
            "File changed outside the app",
            f"{os.path.basename(path)} was edited outside the app since it was loaded.\n\n"
            "Yes — overwrite it with the app version\n"
            "No — reload the edited file (discard app changes)\n"
            "Cancel — don't save")
        if ans is None:
            return False
        if ans is False:
            try:
                blocks, kinds = load_annotation_blocks(path)
            except Exception as e:
                messagebox.showerror("Reload Error", str(e))
                return False
            self.page_ocr_cache = {}
            self.lines_by_page = {}
            self.detections = []
            self._history = {}
            self._set_detections([])
            self._apply_loaded_blocks(blocks, strict=False)
            self._disk_sig[key] = file_sig(path)
            self._set_dirty(False)
            self.status.set(f"Reloaded {os.path.basename(path)} — nothing was saved.")
            return False
        return True

    def _check_current_page_external(self, event=None):
        now = time.time()
        if event is not None and now - self._last_ext_check < 1.5:
            return
        self._last_ext_check = now
        if self.original_image is None or not self.autoload_var.get():
            return
        p = self.current_page
        path = self._page_file_path(p)
        if not path:
            return
        cur = file_sig(path)
        if cur is None or cur == self._disk_sig.get(path) or self._conflict_seen.get(path) == cur:
            return
        self._stash_current_page()
        if not self._app_changed(p):
            if self._reload_page_file(p, path):
                self.status.set(f"Page {p + 1} reloaded — its JSON was edited outside the app.")
            return
        self._conflict_seen[path] = cur
        if messagebox.askyesno(
                "JSON changed outside the app",
                f"The JSON for page {p + 1} was edited outside the app, but you also have "
                "changes on this page here.\n\nLoad the edited file? "
                "(Yes discards your app changes on this page; No keeps them and "
                "Ctrl+S will ask again.)"):
            self._reload_page_file(p, path)
            self.status.set(f"Page {p + 1} reloaded from the edited file (Ctrl+Z to undo).")

    def reload_current_page(self):
        p = self.current_page
        path = self._page_file_path(p)
        if not path or not os.path.isfile(path):
            self.status.set("This page has no JSON file to reload.")
            return
        self._stash_current_page()
        if self._app_changed(p) and not messagebox.askyesno(
                "Reload page", "Discard your unsaved changes on this page and reload its JSON?"):
            return
        if self._reload_page_file(p, path):
            self.status.set(f"Page {p + 1} reloaded from {os.path.basename(path)} (Ctrl+Z to undo).")

    def _check_tables_before_export(self, entries):
        bad = 0
        for _p, dets, _w, _h in entries:
            for d in dets:
                if d.get("label") == "table" or _looks_like_html(d.get("text", "")):
                    if validate_table_html(d.get("text") or ""):
                        bad += 1
        return bad

    def export_annotations_json(self, path=None, write_images=None, quiet=False):
        if self.image_path is None or self.original_image is None:
            messagebox.showinfo("No image", "Open an image/PDF first.")
            return False
        entries = self._page_entries()
        if not entries:
            if not quiet:
                messagebox.showinfo("No data", "No detections to export.")
            return False
        path = self._ask_path(path, ".json", [("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return False
        if write_images is None:
            write_images = self.export_images_var.get()
        try:
            recs = self._annotation_records(
                entries, os.path.dirname(os.path.abspath(path)), write_images)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(recs, f, ensure_ascii=False, indent=2)
            self._mark_saved(path, "annotations")
            bad = self._check_tables_before_export(entries)
            nbox = sum(len(r["blocks"]) for r in recs)
            msg = f"Saved annotations JSON: {path} ({nbox} boxes, {len(recs)} page(s))"
            if bad:
                msg += f" — warning: {bad} table(s) have HTML problems (Check Page)"
            self.status.set(msg)
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write JSON:\n{e}")
            return False

    def export_jsonl(self, path=None, write_images=None, quiet=False):
        if self.image_path is None or self.original_image is None:
            messagebox.showinfo("No image", "Open an image/PDF first.")
            return False
        entries = self._page_entries()
        if not entries:
            if not quiet:
                messagebox.showinfo("No data", "No detections to export.")
            else:
                self.status.set("Nothing to save yet — draw boxes or run OCR first.")
            return False
        path = self._ask_path(path, ".jsonl",
                              [("JSON Lines", "*.jsonl"), ("All files", "*.*")])
        if not path:
            return False
        if write_images is None:
            write_images = self.export_images_var.get()
        try:
            recs = self._annotation_records(
                entries, os.path.dirname(os.path.abspath(path)), write_images)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
                    f.write("\n")
            self._mark_saved(path, "annotations")
            bad = self._check_tables_before_export(entries)
            nbox = sum(len(r["blocks"]) for r in recs)
            msg = f"Saved annotations JSONL: {path} ({nbox} boxes, {len(recs)} page line(s))"
            if bad:
                msg += f" — warning: {bad} table(s) have HTML problems (Check Page)"
            self.status.set(msg)
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write JSONL:\n{e}")
            return False

    # ==================================================================
    # GROUND TRUTH v2 (one page per line)
    # ==================================================================
    _LEGACY_FONT_HINTS = ("preeti", "kantipur", "sagarmatha", "himalb", "fontasy",
                          "pcs nepali", "ganess", "aakriti", "navjeevan")

    def _pdf_source_kind(self, page):
        try:
            pg = self.pdf_doc.load_page(page)
            if not pg.get_text("text").strip():
                return "scanned"
            fonts = " ".join(str(f[3]).lower() for f in pg.get_fonts())
            if any(h in fonts for h in self._LEGACY_FONT_HINTS):
                return "digital_legacy_font"
            return "digital_unicode"
        except Exception:
            return None

    def _gt_v2_record(self, page, dets, w, h, out_dir, split):
        blocks, statuses = [], set()
        for i, b in enumerate(self._blocks_payload(dets)):
            statuses.add(b.get("status") or "auto")
            nb = {k: b[k] for k in ("block_id", "block_label", "block_bbox",
                                    "block_content", "block_order")}
            nb["block_id"] = i
            if b.get("ignore_in_eval"):
                nb["ignore_in_eval"] = True
            if b.get("is_underlined"):
                nb["is_underlined"] = True
            blocks.append(nb)

        labels = {b["block_label"] for b in blocks}
        tables = [b["block_content"] for b in blocks if b["block_label"] == "table"]
        tags = set()
        if tables:
            tags.add("has_table")
            if any(t.count("<table") > 1 for t in tables):
                tags.add("nested_table")
        allt = " ".join(b["block_content"] for b in blocks)
        if re.search("[\u0966-\u096f]", allt):
            tags.add("nepali_digits")
        if re.search(r"[A-Za-z]", re.sub(r"<[^>]+>", "", allt)):
            tags.add("mixed_script")
        if labels & {"seal", "stamp"}:
            tags.add("seal")
        if any(l.startswith("handwrit") for l in labels):
            tags.add("handwriting")
        if "signature" in labels:
            tags.add("signature")
        if labels & {"form", "form_field", "checkbox", "radio_button"}:
            tags.add("form")
        if labels & {"crossed_out", "redacted", "blurred", "unclear"}:
            tags.add("degraded")
        if any(b.get("is_underlined") for b in blocks):
            tags.add("underlined")

        png = os.path.join(split_pages_dir(self.image_path), self._page_png_name(page))
        if self._is_page_file():
            png = self.image_path
        img = png if os.path.isfile(png) else self.image_path
        pid = (os.path.splitext(os.path.basename(self.image_path))[0] if self._is_page_file()
               else split_page_stem(self.image_path, page))
        status = "reviewed" if statuses == {"reviewed"} else "draft"
        meta = {"status": status, "split": split,
                "source_file": os.path.basename(
                    (self._split_meta or {}).get("source") or self.image_path),
                "tags": sorted(tags)}
        if self.pdf_doc is not None:
            meta["dpi"] = int(round(72 * self.pdf_zoom))
            kind = self._pdf_source_kind(page)
            if kind:
                meta["source_kind"] = kind
        return {
            "schema_version": "2.0",
            "page_id": pid,
            "input_path": self._rel(img, out_dir),
            "page_index": page if self.pdf_doc is not None else None,
            "page_count": self.page_count,
            "width": w, "height": h,
            "gt_meta": meta,
            "parsing_res_list": blocks,
        }

    def export_gt_v2_jsonl(self, path=None):
        if self.image_path is None or self.original_image is None:
            messagebox.showinfo("No image", "Open an image/PDF first.")
            return False
        self._autosave_pending_new()
        self._removed_leftovers = 0
        self._last_autofix = self._clean_all_pages()
        entries = self._page_entries()
        if not entries:
            messagebox.showinfo("No data", "No boxes to export.")
            return False
        path = self._ask_path(path, ".jsonl", [("JSON Lines", "*.jsonl"), ("All files", "*.*")])
        if not path:
            return False
        out_dir = os.path.dirname(os.path.abspath(path))
        try:
            recs = [self._gt_v2_record(p, dets, w, h, out_dir, "test") for p, dets, w, h in entries]
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write ground truth:\n{e}")
            return False
        drafts = sum(1 for r in recs if r["gt_meta"]["status"] == "draft")
        msg = f"Saved ground truth v2: {os.path.basename(path)} ({len(recs)} page(s)"
        msg += f", {drafts} still draft - mark all boxes reviewed)" if drafts else ", all reviewed)"
        self.status.set(msg + self._save_suffix(entries))
        return True

    def export_training_jsonl(self, path=None):
        if self.image_path is None or self.original_image is None:
            messagebox.showinfo("No image", "Open an image/PDF first.")
            return False
        entries = self._page_entries()
        if not entries:
            messagebox.showinfo("No data", "Run OCR or annotate first.")
            return False
        path = self._ask_path(path, ".jsonl",
                              [("JSON Lines", "*.jsonl"), ("All files", "*.*")])
        if not path:
            return False

        img_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "images")
        lines = []
        skipped_unreviewed = 0
        for pnum, dets, _w, _h in entries:
            reviewed = [d for d in dets if d.get("status") == "reviewed"]
            if not reviewed:
                skipped_unreviewed += 1
                continue
            reviewed = [d for _i, d in sorted(enumerate(reviewed),
                                              key=lambda t: _order_key(t[1], t[0]))]
            parts = []
            for d in reviewed:
                label = canonical_label(d.get("label")).lower()
                if label in EMPTY_CONTENT_LABELS:
                    continue
                t = (d.get("text") or "").strip()
                if label == "table" or _looks_like_html(t):
                    t = normalize_table_html(t)
                if not t:
                    continue
                parts.append(t)
            if not parts:
                continue

            name = self._write_page_png(pnum, img_dir)
            if not name:
                continue
            lines.append({
                "image_info": [{"matched_text_index": 0,
                                "image_url": f"./images/{name}"}],
                "text_info": [
                    {"text": "OCR:", "tag": "mask"},
                    {"text": "\n".join(parts), "tag": "no_mask"},
                ],
            })

        if not lines:
            messagebox.showinfo(
                "Nothing to export",
                f"No pages have reviewed blocks.\n"
                f"Skipped {skipped_unreviewed} page(s).\n\n"
                f"Click 'Mark All Page Reviewed' before exporting.")
            return False

        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for line in lines:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            self.status.set(
                f"Exported training JSONL: {path} ({len(lines)} page line(s), "
                f"{skipped_unreviewed} skipped)")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write JSONL:\n{e}")
            return False

    def export_searchable_pdf(self, path=None):
        if self.pdf_doc is None:
            messagebox.showinfo("No PDF", "Open a PDF first.")
            return False
        entries = {p: dets for p, dets, _w, _h in self._page_entries()}
        path = self._ask_path(path, ".pdf", [("PDF", "*.pdf")])
        if not path:
            return False
        try:
            out_doc = fitz.open()
            font_path = self._devanagari_font_path or None
            z = self.pdf_zoom
            for page_num in range(self.pdf_doc.page_count):
                page = self.pdf_doc.load_page(page_num)
                new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
                pix = page.get_pixmap(dpi=150)
                new_page.insert_image(new_page.rect, stream=pix.tobytes("png"))
                for d in entries.get(page_num, []):
                    text = d.get("text") or ""
                    if d.get("label") == "table" or _looks_like_html(text):
                        text = _strip_html_tags(text)
                    if not text.strip():
                        continue
                    x, y, w, h = d["bbox"]
                    px, py, pw, ph = x / z, y / z, w / z, h / z
                    try:
                        kwargs = dict(fontsize=max(6, ph * 0.7), render_mode=3)
                        if font_path:
                            kwargs.update(fontname="devfont", fontfile=font_path)
                        else:
                            kwargs.update(fontname="helv")
                        new_page.insert_textbox(
                            fitz.Rect(px, py, px + pw, py + ph), text, **kwargs)
                    except Exception:
                        pass
            out_doc.save(path)
            out_doc.close()
            self.status.set(f"Exported searchable PDF: {path}")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to create searchable PDF:\n{e}")
            return False

    def export_fields_csv(self, path=None):
        items = self.results_tree.get_children()
        if not items:
            messagebox.showinfo("No data", "Run field extraction first.")
            return False
        path = self._ask_path(path, ".csv", [("CSV", "*.csv")])
        if not path:
            return False
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["field", "value"])
                for item in items:
                    writer.writerow(self.results_tree.item(item, "values"))
            self.status.set(f"Exported fields CSV: {path}")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write CSV:\n{e}")
            return False

    # ==================================================================
    # RESET / CLOSE
    # ==================================================================
    def reset(self):
        if not self._confirm_discard():
            return
        self.image_path = None
        self.original_image = None
        self.raw_page_image = None
        self.annotated_image = None
        self.tk_preview = None
        if self.pdf_doc is not None:
            self.pdf_doc.close()
            self.pdf_doc = None
        self.current_page = 0
        self.page_count = 1
        self.detections = []
        self.page_ocr_cache = {}
        self.lines_by_page = {}
        self.rich_page_meta = {}
        self._pending_underline = False
        self._history = {}
        self._annotation_path = None
        self._annotation_kind = None
        self._split_meta = None
        self._coord_scale = 1.0
        self._set_dirty(False)
        self.selected_bbox = None
        self.selected_detection_index = None
        self.drag_start = None
        self.interaction_mode = None
        self.text_box.delete("1.0", tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self._update_selection_panel()
        self._update_page_buttons()
        self._update_undo_buttons()
        self.canvas.delete("all")
        self.run_btn.config(state=tk.DISABLED)
        self.run_all_btn.config(state=tk.DISABLED)
        self._set_default_export_path_for_current_type()
        self.status.set("Ready. Open an image, PDF or annotation JSON to begin.")

    def on_close(self):
        if self._confirm_discard():
            self.root.destroy()

    # ==================================================================
    # FIELD EXTRACTION
    # ==================================================================
    def _add_field_pattern(self):
        name = self.field_name_var.get().strip()
        pattern = self.field_pattern_var.get().strip()
        if not name or not pattern:
            messagebox.showinfo("Missing", "Enter both a field name and a regex pattern.")
            return
        try:
            re.compile(pattern)
        except re.error as e:
            messagebox.showerror("Bad Regex", f"Invalid regex pattern:\n{e}")
            return
        self.pattern_tree.insert("", tk.END, values=(name, pattern))
        self.field_name_var.set("")
        self.field_pattern_var.set("")

    def _load_default_fields(self):
        for item in self.pattern_tree.get_children():
            self.pattern_tree.delete(item)
        for name, pattern in DEFAULT_FIELD_PATTERNS:
            self.pattern_tree.insert("", tk.END, values=(name, pattern))
        self.status.set(f"Loaded {len(DEFAULT_FIELD_PATTERNS)} common field patterns.")

    def _remove_field_pattern(self):
        sel = self.pattern_tree.selection()
        if sel:
            self.pattern_tree.delete(sel[0])

    def run_field_extraction(self):
        patterns = []
        for item in self.pattern_tree.get_children():
            vals = self.pattern_tree.item(item, "values")
            patterns.append((vals[0], vals[1]))
        if not patterns:
            messagebox.showinfo("No patterns", "Add field patterns first.")
            return
        text = self.text_box.get("1.0", tk.END)
        if not text.strip():
            messagebox.showinfo("No text", "Run OCR first to get text.")
            return
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        count = 0
        for name, pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = (match.group(1) if match.lastindex and match.lastindex >= 1
                         else match.group(0))
                self.results_tree.insert("", tk.END, values=(name, value))
                count += 1
        self.status.set(f"Field extraction complete: {count} values found.")

    # ==================================================================
    # AI ASSIST (Ollama)
    # ==================================================================
    def _require_ollama(self):
        if not self.ollama_available:
            messagebox.showinfo(
                "Ollama not available",
                "Start Ollama locally (ollama serve) and pull a model, e.g.\n"
                "  ollama pull gemma3:4b\n\n"
                "Tip: 'Re-read Region (PaddleOCR, fast)' works without Ollama.")
            return False
        return True

    def _run_ollama_fix(self, whole_page=False):
        if not self._require_ollama():
            return
        crop = None
        text = ""
        if whole_page:
            text = self.text_box.get("1.0", tk.END).strip()
            if not text:
                messagebox.showinfo("No text", "Run OCR first.")
                return
        elif self.selected_detection_index is not None:
            text = self.detections[self.selected_detection_index]["text"]
            d = self.detections[self.selected_detection_index]
            if d.get("label") == "table" or _looks_like_html(text):
                messagebox.showinfo("Table", "Table HTML can't be text-fixed; "
                                             "edit cells manually or re-run Table OCR.")
                return
        else:
            region = self._current_region()
            if region is None:
                messagebox.showinfo("No selection", "Select a row or draw a box.")
                return
            crop = self._crop(*region)

        model = self.text_model_var.get()
        lang = self.lang_var.get()
        self.backends.apply_fixes = self.apply_fixes_var.get()
        self.status.set(f"Asking Ollama ({model}) to fix text...")
        self.ai_timing_lbl.config(text="Working...")

        def worker():
            try:
                src = text
                if crop is not None:
                    src = self.backends.reread_crop(crop, lang)
                if not src.strip():
                    self.root.after(0, lambda: self._ollama_error(
                        "Selected region has no readable text."))
                    return
                result, elapsed = self.ollama.generate_text(
                    model, TEXT_FIX_PROMPT.format(text=src))
                self.root.after(0, lambda: self._ollama_fix_done(result, elapsed))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._ollama_error(err))
        threading.Thread(target=worker, daemon=True).start()

    def _ollama_fix_done(self, result, elapsed):
        self.ai_result_box.delete("1.0", tk.END)
        self.ai_result_box.insert("1.0", result)
        self.ai_timing_lbl.config(text=f"Completed in {elapsed:.1f}s")
        self.status.set("Suggestion ready — review, then Accept or Discard.")

    def _run_ollama_batch_fix(self):
        if not self._require_ollama():
            return
        if not self.detections:
            messagebox.showinfo("No data", "Run OCR first.")
            return

        threshold = self.batch_threshold_var.get()
        candidates = []
        for i, d in enumerate(self.detections):
            if canonical_label(d.get("label")) in EMPTY_CONTENT_LABELS | {"table"}:
                continue
            if _looks_like_html(d.get("text", "")):
                continue
            if d.get("status") == "reviewed":
                continue
            if not (d.get("text") or "").strip():
                continue
            if d.get("conf", 100) < threshold:
                candidates.append(i)

        if not candidates:
            messagebox.showinfo(
                "Nothing to fix",
                f"No unreviewed text detections below confidence {threshold}.\n"
                f"Try raising the threshold, or re-run OCR with a backend that "
                f"records confidence (PaddleOCR / EasyOCR / Surya).",
            )
            return

        self._push_undo()
        model = self.text_model_var.get()
        total = len(candidates)
        snapshot = {i: self.detections[i]["text"] for i in candidates}
        det_ref = self.detections
        self.batch_fix_btn.config(state=tk.DISABLED)
        self.status.set(f"Batch AI-fix: 0/{total} blocks...")
        self.ai_timing_lbl.config(text=f"Batch fix: 0/{total}...")

        def worker():
            start, fixed, errors, done = time.time(), 0, 0, 0
            for c in range(0, total, BATCH_CHUNK_SIZE):
                chunk = candidates[c:c + BATCH_CHUNK_SIZE]
                payload = {str(i): snapshot[i] for i in chunk}
                try:
                    out, _ = self.ollama.generate_json(
                        model, BATCH_FIX_PROMPT.format(
                            payload=json.dumps(payload, ensure_ascii=False)))
                except Exception as e:
                    errors += len(chunk)
                    print(f"[ocr-tool] batch chunk failed: {e}")
                    out = {}
                for i in chunk:
                    new = out.get(str(i))
                    if not isinstance(new, str):
                        continue
                    new = normalize_text(new.strip())
                    if new and new != snapshot[i]:
                        self.root.after(0, lambda i=i, t=new: self._apply_batch_fix(
                            det_ref, i, t))
                        fixed += 1
                done += len(chunk)
                self.root.after(0, lambda n=done: self._update_batch_progress(
                    n, total, int(n / total * 100)))
            elapsed = time.time() - start
            self.root.after(0, lambda: self._finish_batch_fix(fixed, errors, total, elapsed))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_batch_fix(self, det_ref, idx, new_text):
        if 0 <= idx < len(det_ref):
            d = det_ref[idx]
            d.setdefault("ocr_text", d["text"])
            d["text"] = new_text
            d["status"] = "ai_suggested"
            if det_ref is self.detections:
                self._sync_tree_row(idx)

    def _update_batch_progress(self, n, total, pct):
        self.status.set(f"Batch AI-fix: {n}/{total} ({pct}%)...")
        self.ai_timing_lbl.config(text=f"Batch fix: {n}/{total}...")

    def _finish_batch_fix(self, fixed, errors, total, elapsed):
        self.batch_fix_btn.config(state=tk.NORMAL)
        self.refresh_preview()
        self._update_selection_panel()
        msg = (f"Batch fix complete: {fixed}/{total} blocks updated "
               f"(marked ai_suggested) in {elapsed:.1f}s — Ctrl+Z undoes the batch")
        if errors:
            msg += f" ({errors} failed)"
        self.status.set(msg)
        self.ai_timing_lbl.config(text=f"Batch done in {elapsed:.1f}s")

    def _run_ollama_vision(self):
        if not self._require_ollama():
            return
        region = self._current_region()
        if region is None:
            messagebox.showinfo("No selection", "Draw a box or select a detection.")
            return
        crop = self._crop(*region)
        model = self.vision_model_var.get()
        self.status.set(f"Asking Ollama vision ({model})... (slow on CPU)")
        self.ai_timing_lbl.config(text="Working (vision models are slow on CPU)...")

        def worker():
            try:
                result, elapsed = self.ollama.generate_vision(
                    model, VISION_READ_PROMPT, crop)
                self.root.after(0, lambda: self._ollama_fix_done(result, elapsed))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._ollama_error(err))
        threading.Thread(target=worker, daemon=True).start()

    def _ollama_error(self, err, title="Ollama Error"):
        self.ai_timing_lbl.config(text="Error!")
        messagebox.showerror(title, f"Request failed:\n{err}")
        self.status.set(f"{title}.")

    def _accept_ai_to_row(self):
        suggestion = normalize_text(self.ai_result_box.get("1.0", tk.END).strip())
        if not suggestion:
            messagebox.showinfo("Empty", "No suggestion to apply.")
            return
        if self.selected_detection_index is not None:
            idx = self.selected_detection_index
            self._push_undo()
            d = self.detections[idx]
            d.setdefault("ocr_text", d["text"])
            d["text"] = suggestion
            d["status"] = "reviewed"
            self._sync_tree_row(idx)
            self._update_selection_panel()
            self.refresh_preview()
            self.status.set(f"Applied suggestion to detection #{idx} (Ctrl+Z to undo).")
        elif self.selected_bbox is not None:
            self.sel_text_var.set(suggestion)
            self._save_selected_text()
        else:
            messagebox.showinfo("No selection", "Select a row or draw a box.")

    def _accept_ai_to_page(self):
        suggestion = self.ai_result_box.get("1.0", tk.END).strip()
        if not suggestion:
            messagebox.showinfo("Empty", "No suggestion to apply.")
            return
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", normalize_text(suggestion))
        self.status.set("Replaced page text with suggestion.")


# ======================================================================
# MAIN
# ======================================================================
def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    app = OCRApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()