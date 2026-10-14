"""
OCR GUI Tool — Images + PDF, bounding boxes, image info, page navigation,
preprocessing, regex field extraction, pluggable OCR backends, table
recognition, and optional Ollama-assisted correction / re-reading.

Backends:
    tesseract          — baseline, needs tesseract-ocr-nep on Ubuntu
    easyocr            — native Nepali/Hindi, ~2GB torch install
    surya              — best open Devanagari quality, GPU recommended
                         (works with old and new Surya APIs)
    paddleocr          — lightweight dedicated devanagari_PP-OCRv5_server_rec
    paddlestructurev3  — layout + table recognition (returns HTML tables)
    auto               — tries surya -> paddlestructurev3 -> paddleocr -> easyocr -> tesseract

Block labels (custom set, see PADDLE_LABELS):
    text:        title heading subheading paragraph text_line list_item label
                 header footer page_number
    tables:      table table_heading table_footer
    forms:       form form_field checkbox radio_button
    handwriting: handwriting signature handwritten_annotation
                 handwritten_correction handwritten_number
    marks:       logo stamp seal letterhead watermark
    graphics:    image photo diagram chart map illustration
    condition:   crossed_out redacted blurred unclear mixed_text
    OCR-backend / older labels (text, doc_title, paragraph_title, ...) are
    mapped to this set automatically on OCR, load and save.

Automatic annotation (no drawing, no typing labels):
    1. Open a page and click "Nepali" or "English" (top row). The page is then
       annotated automatically (or press F7 / "Auto-annotate page"):
         * every text line is read (Box OCR engine; table rules erased first)
         * tables = 3+ rows of side-by-side cells -> table block + HTML
         * other lines are grouped into blocks (close, aligned, same size);
           a short last line, a first-line indent or a big gap starts a new one
         * blocks are labelled title / heading / subheading / list / form /
           paragraph; header and footer by position and wording
         * logos / stamps / figures from big ink blobs and coloured ink
         * reading order (header first, footer last on export)
       With PaddleOCR installed, PPStructureV3 layout regions are used as well.
    2. Fix wrong guesses in one click: select a box (or ANY line inside a block)
       and press a Quick label button or a key on the image:
         1 Title  2 Heading  3 Subheading  4 Paragraph  5 List  6 Table
         7 Header  8 Footer  9 Form  0 Logo   (+ Stamp, Figure, Page no., Line)
       A line relabels the block it belongs to; a line with no block gets one.
       M (or "Merge") joins the selected block with the next block below it.
       F7 rebuilds every block except the ones you changed in this session -
       blocks loaded from a saved file are regrouped too; saved line boxes are
       kept and OCR only fills in lines that are missing.
       A block without line boxes gets them on export, estimated from the ink.
    Lists with sub-items: ONE list block covers the whole list. If you also draw a
       box around a sub-list (or around one item's text), it is folded into the
       outer list on export - synthetic v2 has no nested blocks. Every item and
       sub-item starts a new line in the text ("1. ...\nक) ...\nख) ...\n2. ..."),
       a wrapped line joins its item with a space. Markers understood: 1. 1) १.
       i. a) क) (क) (१) (a) (iv) and bullets • - * ➢ ✓ ...
    3. Ctrl+S writes the synthetic v2 JSON: '#' prefixes for headings, newlines
       between list items / header & footer lines, spaces inside paragraphs,
       and the page language you picked.

Annotation workflow:
    * Opening an image/PDF auto-loads an existing sidecar annotation file
      (<name>_annotations.jsonl / .json, PaddleOCR-VL JSON/JSONL, simple JSON)
      from the same folder or an "annotations/" subfolder and draws its boxes.
    * You can also open a .json/.jsonl directly — the referenced image is found
      and opened automatically.
    * "Split PDF into page images" (on by default): opening a PDF writes
          <name>_pages/<name>_page001.png, ..._page002.png, manifest.json
      and Ctrl+S writes one annotation per page next to its image:
          <name>_pages/<name>_page001.json  {"image": "<name>_page001.png",
                                            "source_pdf": "../<name>.pdf",
                                            "page": 1, ..., "blocks": [...]}
      Reopening the PDF loads each page file onto its page; opening a page PNG
      (or its JSON) on its own also works and saves back with the PDF link kept.
    * With splitting off, Ctrl+S saves all pages to <name>_annotations.jsonl.
    * Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) undo / redo box edits (per page).
    * Ctrl+A selects all text in any text field.
    * Delete / BackSpace on the canvas or list deletes the selected box.

Bounding boxes on disk are always in the coordinate space of the raw page
render (PDF page at "render_zoom", or the original image), with "width",
"height" and "render_zoom" recorded so they can be rescaled on import.

Exports PaddleOCR-VL JSON / JSONL, annotations JSON / JSONL,
training JSONL (ERNIEKit), CSV, searchable PDF.

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
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
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

from list_tree import (build_list_tree, tree_text, line_paths, flatten_tree,
                       absorb_nested_blocks, distribute_text_to_lines,
                       merge_list_detections, flat_items, items_text,
                       list_block_text)

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

# Per-box OCR ("Box OCR" row): reads ONE drawn box as a whole line/block.
BOX_OCR_ENGINES = [
    "Tesseract",
    "PaddleOCR (rec only)",
    "PaddleOCR (det+rec)",
    "EasyOCR",
    "Surya",
]
DEFAULT_BOX_ENGINE = "Tesseract"
BOX_OCR_MODES = ["line", "block", "word"]
# Tesseract page-segmentation modes: 7 = single text line, 6 = uniform block, 8 = single word
TESS_PSM = {"line": 7, "block": 6, "word": 8}
BOX_MIN_LINE_PX = 48        # small crops are upscaled to about this height
DEFAULT_NEW_BOX_LABEL = "text_line"

AUTO_ORDER = ["surya", "paddlestructurev3", "paddleocr", "easyocr", "tesseract"]

LANG_HINTS = {
    "eng":     ["en"],
    "nep":     ["ne"],
    "eng+nep": ["en", "ne"],
    "nep+eng": ["ne", "en"],
}


def _module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


PADDLE_AVAILABLE = _module_available("paddleocr")


def wants_nepali(lang_str):
    return "ne" in LANG_HINTS.get(lang_str, ["en"])


# ----------------------------------------------------------------------
# Export types: label -> (method, extension, filename suffix)
# Each type gets its own suffix so one export never overwrites another.
# ----------------------------------------------------------------------
EXPORT_TYPES = {
    "Synthetic v2 JSON (training format)": ("export_synth_v2",       ".json",  ""),
    "Ground truth v2 JSONL (benchmark)": ("export_gt_v2_jsonl",      ".jsonl", "_gt"),
    "Annotations JSONL (per-page)":     ("export_jsonl",             ".jsonl", "_annotations"),
    "Annotations JSON (per-page)":      ("export_annotations_json",  ".json",  "_annotations"),
    "Training JSONL (ERNIEKit)":       ("export_training_jsonl",    ".jsonl", "_train"),
    "PaddleOCR-VL JSON":               ("export_paddle_json",       ".json",  "_paddle"),
    "PaddleOCR-VL JSONL (per-page)":   ("export_paddle_jsonl",      ".jsonl", "_paddle"),
    "CSV (detections)":                ("export_csv",               ".csv",   "_detections"),
    "JSON (simple)":                   ("export_json",              ".json",  "_simple"),
    "Searchable PDF":                  ("export_searchable_pdf",    ".pdf",   "_searchable"),
    "CSV (extracted fields)":          ("export_fields_csv",        ".csv",   "_fields"),
}
DEFAULT_EXPORT_TYPE = "Synthetic v2 JSON (training format)"

# ----------------------------------------------------------------------
# Block labels (custom set)
# ----------------------------------------------------------------------
DEFAULT_LABEL = "paragraph"

PADDLE_LABELS = [
    # text
    "title", "heading", "subheading", "paragraph", "text_line", "list", "list_item",
    "form_label", "form_value",
    "label", "header", "footer", "page_number",
    # tables
    "table", "table_heading", "table_footer",
    # forms
    "form", "form_field", "checkbox", "radio_button",
    # handwriting
    "handwriting", "signature", "handwritten_annotation",
    "handwritten_correction", "handwritten_number",
    # marks / branding
    "logo", "stamp", "seal", "letterhead", "watermark",
    # graphics
    "image", "photo", "diagram", "chart", "map", "illustration",
    # condition
    "crossed_out", "redacted", "blurred", "unclear", "mixed_text",
]

# OCR-backend / old labels -> this set (applied on OCR, load and save).
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

# Content is always cleared for these.
EMPTY_CONTENT_LABELS = {
    "image", "photo", "diagram", "chart", "map", "illustration",
    "logo", "seal", "signature", "redacted",
}
# These must have a transcription.
TEXT_REQUIRED_LABELS = {
    "title", "heading", "subheading", "paragraph", "text_line", "list", "list_item",
    "form_label", "form_value",
    "label", "header", "footer", "page_number",
    "table_heading", "table_footer", "form_field",
    "handwriting", "handwritten_annotation", "handwritten_correction",
    "handwritten_number", "mixed_text",
}
# Text optional: table (HTML), form, checkbox, radio_button, stamp,
# letterhead, watermark, crossed_out, blurred, unclear.

# Outside the reading flow: saved with block_order = null + ignore_in_eval.
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
    # text
    "title": (0, 140, 255), "heading": (0, 200, 255), "subheading": (0, 230, 230),
    "paragraph": (200, 200, 200), "text_line": (170, 170, 170),
    "list_item": (180, 200, 140), "list": (160, 210, 120), "label": (140, 180, 220),
    "form_label": (0, 160, 255), "form_value": (60, 200, 60),
    "header": (255, 120, 100), "footer": (255, 120, 100),
    "page_number": (100, 100, 255),
    # tables
    "table": (255, 100, 200), "table_heading": (255, 140, 220),
    "table_footer": (220, 160, 240),
    # forms
    "form": (120, 200, 0), "form_field": (80, 220, 120),
    "checkbox": (40, 180, 40), "radio_button": (40, 160, 90),
    # handwriting
    "handwriting": (200, 0, 200), "signature": (160, 0, 255),
    "handwritten_annotation": (230, 80, 230),
    "handwritten_correction": (130, 0, 200), "handwritten_number": (255, 90, 200),
    # marks
    "logo": (0, 120, 200), "stamp": (60, 60, 230), "seal": (0, 0, 255),
    "letterhead": (0, 90, 160), "watermark": (190, 190, 120),
    # graphics
    "image": (0, 200, 100), "photo": (0, 170, 80), "diagram": (200, 200, 0),
    "chart": (230, 180, 60), "map": (160, 140, 40), "illustration": (60, 220, 160),
    # condition
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


_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})|\\n|\\t|\\r")
_SIMPLE_ESCAPES = {"\\n": "\n", "\\t": "\t", "\\r": "\r"}


def _fix_double_escaped(s):
    """Turn literal escapes from double-escaped JSON or pasted text into real characters:
    '\\u0915' -> 'क', '\\n' (backslash + n) -> newline, '\\t' -> tab, '\\r' -> CR."""
    if not isinstance(s, str) or "\\" not in s:
        return s

    def sub(m):
        if m.group(1) or m.group(2):
            try:
                return chr(int(m.group(1) or m.group(2), 16))
            except ValueError:
                return m.group(0)
        return _SIMPLE_ESCAPES.get(m.group(0), m.group(0))
    return _ESCAPE_RE.sub(sub, s)


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
    """A dot leader containing '…' -> plain dots (each … counts as three)."""
    s = m.group(0)
    return "." * (s.count(".") + 3 * s.count("\u2026"))


def clean_block_content(text, label):
    """
    Safe, loss-free fixes for ground truth. Returns (text, label, fixes).
    """
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
    """Things a human should check against the image (never auto-fixed)."""
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


def qa_page_warnings(dets, width=None, height=None):
    """Page-level checks: reading order, review status, page-covering boxes, per-block warnings."""
    out = []
    if width and height:
        for i, d in enumerate(dets):
            lab = canonical_label(d.get("label"))
            if lab not in LINE_LABELS and is_page_covering(d["bbox"], width, height):
                out.append(f"#{i} ({lab}) covers the whole page - draw one box per region "
                           "(header, paragraph, table, list, footer, ...) instead")
    flow = [d for d in dets if canonical_label(d.get("label")) not in NON_FLOW_LABELS
            and canonical_label(d.get("label")) not in LINE_LABELS]
    orders = sorted(d["order"] for d in flow if isinstance(d.get("order"), int))
    if orders != list(range(1, len(orders) + 1)):
        out.append(f"reading order is not 1..{len(orders)} without gaps: {orders[:15]}")
    missing = sum(1 for d in flow if not isinstance(d.get("order"), int))
    if missing:
        out.append(f"{missing} content block(s) without an order (use Auto Order)")
    unrev = sum(1 for d in dets if d.get("status") != "reviewed")
    if unrev:
        out.append(f"{unrev} block(s) not marked reviewed")
    for i, d in enumerate(dets):
        for w in qa_block_warnings(d):
            out.append(f"#{i} (order {d.get('order')}, {d.get('label')}): {w}")
    return out


def _order_key(d, idx=0):
    o = d.get("order")
    return (o if isinstance(o, (int, float)) else 10**9, idx)


def scale_detections(dets, factor):
    """Return copies of detections with bboxes multiplied by factor."""
    out = []
    for d in dets:
        nd = dict(d)
        x, y, w, h = d["bbox"]
        nd["bbox"] = (int(round(x * factor)), int(round(y * factor)),
                      max(1, int(round(w * factor))), max(1, int(round(h * factor))))
        out.append(nd)
    return out


def reading_order(dets):
    """Top-to-bottom bands of vertically overlapping boxes, left-to-right within a band."""
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
# Rule-based Nepali OCR fixes (OPT-IN).
# ----------------------------------------------------------------------
NEPALI_OCR_FIXES = [
    (r"गाउपालिका", "गाउँपालिका"),   # missing chandrabindu
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
# Table HTML: normalize, repair, validate
# ----------------------------------------------------------------------
_TABLE_TAGS = ("table", "thead", "tbody", "tfoot", "tr", "th", "td")


def normalize_table_html(html):
    """Remove whitespace-only gaps between tags."""
    if not html:
        return html
    return re.sub(r">\s+<", "><", html.strip())


def repair_table_html(html):
    """Normalize and fix the common 'cell without an opening <tr>' mistake."""
    if not html:
        return html
    h = normalize_table_html(html)
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
    """Return a list of human-readable problems ([] if the table is well-formed)."""
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
# Annotation file loading (all supported formats)
# ----------------------------------------------------------------------
def _read_json_any(path):
    """Read a .json (object or list) or .jsonl file into a list of items."""
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
    if rec.get("ocr_text") is not None:
        d["ocr_text"] = normalize_text(rec["ocr_text"])
    return d


# ======================================================================
# SYNTHETIC v2 FORMAT
# ======================================================================
SYNTH_SCHEMA_VERSION = 2

HEADING_LEVELS = {"title": 1, "heading": 2, "subheading": 3, "table_heading": 3}
_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}[ \t]+")


def heading_prefix(level):
    return "#" * max(1, int(level or 1)) + " "


def strip_heading_prefix(text):
    return _HEADING_PREFIX_RE.sub("", text or "", count=1).strip()


def format_heading(text, level):
    t = strip_heading_prefix(text)
    return heading_prefix(level) + t if t else ""


SYNTH_BLOCK_MAP = {
    **{lab: ("heading", {"level": lvl}) for lab, lvl in HEADING_LEVELS.items()},
    "paragraph": ("paragraph", {}),
    "label": ("paragraph", {}),
    "table_footer": ("paragraph", {}),
    "mixed_text": ("paragraph", {}),
    "handwriting": ("paragraph", {"handwritten": True}),
    "handwritten_annotation": ("paragraph", {"handwritten": True}),
    "handwritten_correction": ("paragraph", {"handwritten": True}),
    "handwritten_number": ("paragraph", {"handwritten": True}),
    "list": ("list", {}),
    "list_item": ("list", {}),
    "header": ("header", {}),
    "letterhead": ("header", {}),
    "footer": ("footer", {}),
    "page_number": ("footer", {}),
    "table": ("table", {}),
    "form": ("form_field", {}),
    "form_field": ("form_field", {}),
    "image": ("figure", {}), "photo": ("figure", {}), "diagram": ("figure", {}),
    "chart": ("figure", {}), "map": ("figure", {}), "illustration": ("figure", {}),
    "logo": ("logo", {}),
    "stamp": ("stamp", {}),
    "seal": ("stamp", {}),
}
SYNTH_LINE_ROLES = {"text_line": None, "form_label": "label", "form_value": "value"}
LINE_LABELS = set(SYNTH_LINE_ROLES)
SYNTH_OVERLAYS = {
    "crossed_out": {"crossed_out": True},
    "blurred": {"legible": False},
    "unclear": {"legible": False},
    "redacted": {"legible": False, "redacted": True},
}
SYNTH_EXTRAS = {"signature", "watermark", "checkbox", "radio_button"}
SYNTH_NO_ORDER = {"logo", "stamp"}
SYNTH_FOLD_INTO = {
    "list": {"list", "paragraph"},
    "paragraph": {"paragraph"},
    "heading": {"heading"},
    "header": {"header", "paragraph", "heading"},
    "footer": {"footer", "paragraph"},
    "form_field": {"form_field", "paragraph"},
    "table": {"list", "paragraph", "heading", "form_field"},
}
SYNTH_NO_LINES = {"figure", "logo", "stamp"}
SYNTH_FIRST_TYPES = {"header"}
SYNTH_LAST_TYPES = {"footer"}
PAGE_BOX_FRACTION = 0.90
SYNTH_UNMEASURED_QUALITY = ("downscale", "noise_sigma", "text_contrast", "jpeg_quality")


def is_page_covering(bbox, width, height, frac=PAGE_BOX_FRACTION):
    if not width or not height:
        return False
    return _area_xywh(bbox) >= frac * float(width) * float(height)

SYNTH_TYPE_TO_LABEL = {
    "paragraph": "paragraph", "list": "list", "header": "header", "footer": "footer",
    "table": "table", "form_field": "form_field", "figure": "image",
    "logo": "logo", "stamp": "stamp",
}
SYNTH_LEVEL_TO_LABEL = {1: "title", 2: "heading", 3: "subheading"}

FIELD_TYPE_KEYWORDS = [
    ("name_m", ["grandfather", "father", "husband", "हजुरबुबा", "बाजे", "बाबु", "बुबा", "पति", "पिता"]),
    ("name_f", ["grandmother", "mother", "wife", "हजुरआमा", "आमा", "पत्नी", "माता"]),
    ("phone", ["phone", "mobile", "contact", "tel", "फोन", "मोबाइल", "सम्पर्क"]),
    ("gender", ["gender", "sex", "लिङ्ग", "लिंग"]),
    ("date", ["date", "dob", "birth", "मिति", "जन्म"]),
    ("age", ["age", "उमेर"]),
    ("money", ["amount", "rs", "npr", "salary", "रकम", "रु", "तलब"]),
    ("ward", ["ward", "वडा"]),
    ("district", ["district", "जिल्ला"]),
    ("plot", ["plot", "kitta", "कित्ता"]),
    ("area", ["area", "क्षेत्रफल"]),
    ("post", ["post", "designation", "position", "पद"]),
    ("address", ["address", "ठेगाना"]),
    ("citizenship", ["citizenship", "नागरिकता"]),
    ("name", ["name", "नाम"]),
]

# === FIXED ===
# Before: [०-९]+  —  only matched a SINGLE Devanagari number, so "५.१)" and
# "5.1)" never matched and got list_style="none" (and _list_items() merged
# the sub-items into the previous item). Now the digit class repeats an
# optional ".<digits>" suffix, so "५", "५.१", "५.१.२" and the Latin
# equivalents all match the dev/lat groups.
_LIST_MARKER_RE = re.compile(
    r"^\s*\(?(?:(?P<dev>[०-९]+(?:\.[०-९]+)*)|(?P<lat>[0-9]+(?:\.[0-9]+)*)|"
    r"(?P<roman>[ivxIVX]+)|(?P<alpha>[A-Za-z])|(?P<devalpha>[क-ह]))"
    r"(?:[.)।]|\)\.?)\s"
    r"|^\s*(?P<bullet>[•●○◦▪■□◆◇►▸➢✓✔❖*\-–])\s")

_FORM_SPLIT_RE = re.compile(r"^(?P<label>.*?(?:\s[-–—]|[:：]))(?:\s+(?P<value>.*))?$")
_FORM_SEP_WORDS = {"-", "–", "—", ":", "："}


def _r1(v):
    v = round(float(v), 1)
    return int(v) if v == int(v) else v


def _rect_quad(x0, y0, x1, y1):
    return [[_r1(x0), _r1(y0)], [_r1(x1), _r1(y0)], [_r1(x1), _r1(y1)], [_r1(x0), _r1(y1)]]


def _xyxy_ints(b):
    return [int(round(v)) for v in b]


def _xyxy_out(b):
    return [_r1(v) for v in b]


def _area_xywh(b):
    return max(0, b[2]) * max(0, b[3])


def _inter_xywh(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def det_quad(d):
    x, y, w, h = d["bbox"]
    q, qb = d.get("quad"), d.get("quad_bbox")
    if q and qb and len(q) == 4:
        qx, qy, qw, qh = qb
        sx = w / float(qw) if qw else 1.0
        sy = h / float(qh) if qh else 1.0
        return [[_r1(x + (px - qx) * sx), _r1(y + (py - qy) * sy)] for px, py in q]
    return _rect_quad(x, y, x + w, y + h)


def det_xyxy(d):
    x, y, w, h = d["bbox"]
    sb, qb = d.get("src_xyxy"), d.get("quad_bbox")
    if sb and qb:
        qx, qy, qw, qh = qb
        sx = w / float(qw) if qw else 1.0
        sy = h / float(qh) if qh else 1.0
        return [_r1(x + (sb[0] - qx) * sx), _r1(y + (sb[1] - qy) * sy),
                _r1(x + (sb[2] - qx) * sx), _r1(y + (sb[3] - qy) * sy)]
    if d.get("quad"):
        return _quad_bounds(det_quad(d))
    return [x, y, x + w, y + h]


def _quad_bounds(q):
    xs = [p[0] for p in q]
    ys = [p[1] for p in q]
    return [min(xs), min(ys), max(xs), max(ys)]


def _norm_text(s):
    return re.sub(r"\s+", " ", normalize_text(s or "")).strip()


def infer_field_type(label_text):
    t = (label_text or "").lower()
    words = set(re.findall(r"[a-z]+", t))
    for ftype, kws in FIELD_TYPE_KEYWORDS:
        for kw in kws:
            if kw.isascii():
                if kw in words or (len(kw) > 3 and kw in t):
                    return ftype
            elif kw in t:
                return ftype
    return "text"


def list_style_of(text):
    m = _LIST_MARKER_RE.match(text or "")
    if not m:
        return "none"
    for k in ("dev", "lat", "roman", "alpha", "devalpha", "bullet"):
        if m.group(k):
            return {"lat": "latin", "devalpha": "dev_alpha"}.get(k, k)
    return "none"


def detect_language(text):
    t = re.sub(r"<[^>]+>", " ", text or "")
    dev = len(re.findall("[ऀ-॥॰-ॿ]", t))
    lat = len(re.findall(r"[A-Za-z]", t))
    devd = len(re.findall("[०-९]", t))
    latd = len(re.findall(r"[0-9]", t))
    lang = "ne" if dev > lat else "en"
    if devd and latd:
        digits = "mixed"
    elif devd or latd:
        digits = "dev" if devd > latd else "latin"
    else:
        digits = "dev" if lang == "ne" else "latin"
    return lang, digits


# ---------------------------------------------------------------- tables
class _CellParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.cur, self.depth = [], None, 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            return
        if self.depth > 1:
            return
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            a = dict(attrs)

            def span(k):
                try:
                    return max(1, int(str(a.get(k, "1")).strip() or 1))
                except ValueError:
                    return 1
            self.cur = {"tag": tag, "rowspan": span("rowspan"), "colspan": span("colspan"), "text": []}
            self.rows[-1].append(self.cur)
        elif tag == "br" and self.cur is not None:
            self.cur["text"].append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1
        elif self.depth == 1 and tag in ("td", "th"):
            self.cur = None

    def handle_data(self, data):
        if self.cur is not None:
            self.cur["text"].append(data)


def parse_table_cells(html_text):
    p = _CellParser()
    try:
        p.feed(html_text or "")
        p.close()
    except Exception:
        return [], 0, 0, []
    occ, cells, nrows, ncols = set(), [], 0, 0
    for r, row in enumerate(p.rows):
        c = 0
        for cell in row:
            while (r, c) in occ:
                c += 1
            rs, cs = cell["rowspan"], cell["colspan"]
            for dr in range(rs):
                for dc in range(cs):
                    occ.add((r + dr, c + dc))
            cells.append({"row": r, "col": c, "rowspan": rs, "colspan": cs,
                          "text": _norm_text("".join(cell["text"])),
                          "is_header": cell["tag"] == "th"})
            nrows = max(nrows, r + rs)
            ncols = max(ncols, c + cs)
            c += cs
    nrows = max(nrows, len(p.rows))
    return cells, nrows, ncols, len(p.rows)


def cells_to_html(cells, tr_rows):
    import html as _html
    rows = [[] for _ in range(tr_rows)]
    for c in cells:
        if c["row"] < tr_rows:
            rows[c["row"]].append(c)
    out = ["<table>"]
    for row in rows:
        out.append("<tr>")
        for c in row:
            tag = "th" if c["is_header"] else "td"
            attrs = ""
            if c["rowspan"] > 1:
                attrs += f' rowspan="{c["rowspan"]}"'
            if c["colspan"] > 1:
                attrs += f' colspan="{c["colspan"]}"'
            out.append(f"<{tag}{attrs}>{_html.escape(c['text'], quote=False)}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _rule_components(bw, orient, minlen, max_thick, long_ok=None):
    minlen = int(max(3, minlen))
    ksize = (minlen, 1) if orient == "h" else (1, minlen)
    m = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, ksize))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        lx, ly, lw, lh, a = [int(v) for v in st[i]]
        length = lw if orient == "h" else lh
        if length < minlen or a / float(max(length, 1)) > max_thick:
            continue
        if orient == "h":
            bands = (bw[ly + lh: ly + lh + 2, lx:lx + lw], bw[max(0, ly - 2): ly, lx:lx + lw])
        else:
            bands = (bw[ly:ly + lh, lx + lw: lx + lw + 2].T, bw[ly:ly + lh, max(0, lx - 2): lx].T)
        near = max(float(b.any(axis=0).mean()) if b.size else 0.0 for b in bands)
        if near > 0.3 and (long_ok is None or length < long_ok):
            continue
        out.append((lx, ly, lw, lh, a))
    return out


def detect_ruling_lines(gray, xyxy, pad=6, text_h=15.0):
    H, W = gray.shape[:2]
    x0, y0 = max(0, int(xyxy[0]) - pad), max(0, int(xyxy[1]) - pad)
    x1, y1 = min(W, int(xyxy[2]) + pad), min(H, int(xyxy[3]) + pad)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return [], [], []
    crop = gray[y0:y1, x0:x1]
    bw = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)
    ch, cw = bw.shape
    found = {"h": [], "v": []}
    max_thick = max(5.0, 0.35 * text_h)
    for orient, minlen in (("h", max(15, cw // 5, 0.25 * cw)),
                           ("v", max(10, ch // 6, 0.15 * ch, 2.2 * text_h))):
        long_ok = 0.5 * (cw if orient == "h" else ch)
        for lx, ly, lw, lh, a in _rule_components(bw, orient, minlen, max_thick, long_ok):
            if orient == "h":
                pos = y0 + ly + lh / 2.0
                thick = max(1, int(round(a / float(max(lw, 1)))))
                p0, p1 = [x0 + lx, _r1(pos)], [x0 + lx + lw, _r1(pos)]
            else:
                pos = x0 + lx + lw / 2.0
                thick = max(1, int(round(a / float(max(lh, 1)))))
                p0, p1 = [_r1(pos), y0 + ly], [_r1(pos), y0 + ly + lh]
            found[orient].append({"orientation": orient, "width": thick, "p0": p0, "p1": p1,
                                  "bbox": [x0 + lx, y0 + ly, x0 + lx + lw, y0 + ly + lh],
                                  "_pos": pos})
    entries = []
    pos_out = {}
    for orient in ("h", "v"):
        items = sorted(found[orient], key=lambda e: e["_pos"])
        merged = []
        for e in items:
            if merged and abs(e["_pos"] - merged[-1]["_pos"]) <= 3:
                m = merged[-1]
                m["bbox"] = [min(m["bbox"][0], e["bbox"][0]), min(m["bbox"][1], e["bbox"][1]),
                             max(m["bbox"][2], e["bbox"][2]), max(m["bbox"][3], e["bbox"][3])]
                if orient == "h":
                    m["p0"][0], m["p1"][0] = m["bbox"][0], m["bbox"][2]
                else:
                    m["p0"][1], m["p1"][1] = m["bbox"][1], m["bbox"][3]
            else:
                merged.append(e)
        pos_out[orient] = [e["_pos"] for e in merged]
        for e in merged:
            e.pop("_pos")
            entries.append(e)
    return pos_out["h"], pos_out["v"], entries


def _grid_bounds(n, ext, lo, hi, snaps):
    if n <= 0:
        return [lo, hi]
    b = [None] * (n + 1)
    b[0], b[n] = float(lo), float(hi)
    for i in range(1, n):
        if (i - 1) in ext and i in ext:
            mid = (ext[i - 1][1] + ext[i][0]) / 2.0
            between = [s for s in (snaps or []) if ext[i - 1][1] - 2 <= s <= ext[i][0] + 2]
            b[i] = min(between, key=lambda s: abs(s - mid)) if between else mid
    known = [i for i, v in enumerate(b) if v is not None]
    for i in range(n + 1):
        if b[i] is None:
            a = max(k for k in known if k < i)
            c = min(k for k in known if k > i)
            b[i] = b[a] + (b[c] - b[a]) * (i - a) / float(c - a)
    if snaps:
        steps = sorted(b[i + 1] - b[i] for i in range(n))
        tol = max(3.0, 0.45 * steps[len(steps) // 2])
        snapped = list(b)
        for i in range(n + 1):
            near = min(snaps, key=lambda s: abs(s - b[i]))
            if abs(near - b[i]) <= tol:
                snapped[i] = near
        if all(snapped[i] < snapped[i + 1] for i in range(n)):
            b = snapped
    return b


# ---------------------------------------------------------------- ink fitting
def tighten_to_ink(gray, bbox, fit_angle=True):
    H, W = gray.shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = gray[y0:y1, x0:x1]
    if float(crop.std()) < 6.0:
        return None
    m = 2
    ox0, oy0, ox1, oy1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    _t, bw_out = cv2.threshold(gray[oy0:oy1, ox0:ox1], 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ix, iy = x0 - ox0, y0 - oy0
    bw = np.ascontiguousarray(bw_out[iy:iy + (y1 - y0), ix:ix + (x1 - x0)])
    ch, cw = bw.shape
    for orient, minlen in (("h", 0.8 * cw), ("v", 0.8 * ch)):
        for lx, ly, lw, lh, _a in _rule_components(bw, orient, minlen, max(3.0, 0.3 * ch)):
            if orient == "h":
                bw[max(0, ly - 1):ly + lh + 1, :] = 0
            else:
                bw[:, max(0, lx - 1):lx + lw + 1] = 0
    # cell borders: an edge column that is mostly ink and continues outside the box
    colcov = (bw > 0).mean(axis=0)
    edge_ = min(15, max(3, int(0.05 * cw)))
    for c in sorted(set(list(range(min(edge_, cw))) + list(range(max(0, cw - edge_), cw)))):
        if colcov[c] >= 0.5:
            cx_ = c + ix
            up = iy > 0 and bool(bw_out[max(0, iy - 2):iy, cx_].any())
            down = iy + ch < bw_out.shape[0] and bool(bw_out[iy + ch:iy + ch + 2, cx_].any())
            if up or down:
                bw[:, max(0, c - 1):c + 2] = 0
    # pieces of a horizontal rule: a thin (<= 3 px) wide run of ink with blank
    # rows above and below. Kept if it is the only ink in the box (a dot leader).
    rowink = (bw > 0).sum(axis=1)
    thr = max(2.0, 0.02 * cw)
    runs_, a_ = [], None
    for r_, v_ in enumerate(list(rowink > thr) + [False]):
        if v_ and a_ is None:
            a_ = r_
        elif not v_ and a_ is not None:
            runs_.append((a_, r_ - 1))
            a_ = None
    if len(runs_) > 1:
        for a_, b_ in runs_:
            if b_ - a_ + 1 > 3 or rowink[a_:b_ + 1].max() < 0.15 * cw:
                continue
            above = a_ == 0 or bool((rowink[max(0, a_ - 2):a_] <= thr).all())
            below = b_ == ch - 1 or bool((rowink[b_ + 1:b_ + 3] <= thr).all())
            if above and below:
                bw[max(0, a_ - 1):b_ + 2, :] = 0
    n, lab, st, _c = cv2.connectedComponentsWithStats(bw, 8)

    def cut_at(i, row, out_row):
        if out_row < 0 or out_row >= bw_out.shape[0]:
            return False
        cols = np.nonzero(lab[row] == i)[0]
        return bool(cols.size) and bool(bw_out[out_row, cols + ix].any())

    def continues_v(i):
        """True if the component's ink carries on above or below the box -
        a real table rule does, a danda at the end of a line does not."""
        _rr, cc = np.nonzero(lab == i)
        if not cc.size:
            return True
        c0, c1 = int(cc.min()) + ix, int(cc.max()) + ix + 1
        top, bot = iy, iy + ch
        above = top > 0 and bool(bw_out[max(0, top - 2):top, c0:c1].any())
        below = bot < bw_out.shape[0] and bool(bw_out[bot:bot + 2, c0:c1].any())
        return above or below

    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        lx, ly, lw, lh, a = [int(v) for v in st[i]]
        if a < 3:
            continue
        if lh < 0.35 * ch and ((ly == 0 and cut_at(i, 0, iy - 1)) or
                               (ly + lh == ch and cut_at(i, ch - 1, iy + ch))):
            continue
        rule_h = (lw >= 0.9 * cw and lh <= max(3, 0.2 * ch)) or (lw >= 0.4 * cw and lh <= 2)
        rule_v = ((lh >= 0.9 * ch and lw <= max(3, 0.1 * cw) and (lx <= 1 or lx + lw >= cw - 1))
                  or (lh >= 0.6 * ch and lw <= 2 and (lx <= 2 or lx + lw >= cw - 2)))
        if rule_h or (rule_v and continues_v(i)):
            continue
        keep[i] = True
    if keep.any():
        hs_ = [int(st[i][3]) for i in range(1, n) if keep[i]]
        big_h = max(hs_)
        main = [i for i in range(1, n) if keep[i] and st[i][3] >= 0.35 * big_h]
        if main:
            by0_ = min(int(st[i][1]) for i in main)
            by1_ = max(int(st[i][1] + st[i][3]) for i in main)
            gap_ = 0.25 * (by1_ - by0_)
            for i in range(1, n):
                if keep[i] and st[i][3] < 0.35 * big_h and (
                        st[i][1] + st[i][3] < by0_ - gap_ or st[i][1] > by1_ + gap_):
                    keep[i] = False
    if not keep.any():
        return None
    ys, xs = np.nonzero(keep[lab])
    ix0, ix1, iy0, iy1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    xyxy = [x0 + ix0, y0 + iy0, x0 + ix1, y0 + iy1]
    if fit_angle and len(xs) >= 30 and (ix1 - ix0) >= 4 * (iy1 - iy0):
        import math
        pts = np.column_stack([xs, ys]).astype(np.float32)
        box = cv2.boxPoints(cv2.minAreaRect(pts))
        s, dd = box.sum(axis=1), box[:, 1] - box[:, 0]
        tl, br = box[int(np.argmin(s))], box[int(np.argmax(s))]
        tr, bl = box[int(np.argmin(dd))], box[int(np.argmax(dd))]
        ang = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
        long_side = math.hypot(tr[0] - tl[0], tr[1] - tl[1]) > math.hypot(bl[0] - tl[0], bl[1] - tl[1])
        if long_side and 0.3 <= abs(ang) <= 10:
            quad = [[_r1(x0 + float(p[0])), _r1(y0 + float(p[1]))] for p in (tl, tr, br, bl)]
            qb = _quad_bounds(quad)
            return [int(np.floor(qb[0])), int(np.floor(qb[1])),
                    int(np.ceil(qb[2])), int(np.ceil(qb[3]))], quad, round(ang, 2)
    return xyxy, None, 0.0


# ---------------------------------------------------------------- page builder
def _synth_block_order(flow):
    geo = reading_order(flow)
    final = sorted([b for b in flow if isinstance(b["det"].get("order"), (int, float))],
                   key=lambda b: b["det"]["order"])
    if not final:
        return geo
    placed = {id(b) for b in final}
    for i, b in enumerate(geo):
        if id(b) in placed:
            continue
        j = i - 1
        while j >= 0 and id(geo[j]) not in placed:
            j -= 1
        pos = 0 if j < 0 else final.index(geo[j]) + 1
        final.insert(pos, b)
        placed.add(id(b))
    return final


def _list_items(texts):
    items = []
    for t in texts:
        if not items or _LIST_MARKER_RE.match(t):
            items.append(t)
        else:
            items[-1] += " " + t
    return items


def _make_line(text, xyxy, quad=None, role=None, src=None):
    x0, y0, x1, y1 = xyxy
    return {"text": text, "xyxy": list(xyxy), "bbox": (x0, y0, x1 - x0, y1 - y0),
            "quad": quad or _rect_quad(x0, y0, x1, y1), "role": role,
            "flags": dict((src or {}).get("flags", {})), "ocr_text": (src or {}).get("ocr_text")}


_LEADER_RE = re.compile(r"[.\u2026_]{4,}")


def _split_form_line(L):
    t = L["text"]
    leaders = list(_LEADER_RE.finditer(t))
    if leaders:
        rest = re.sub(r"\s+", " ", _LEADER_RE.sub(" ", t)).strip()
        if any(c.isalnum() for c in rest):
            inside = [False] * len(t)
            for lm in leaders:
                for k in range(lm.start(), lm.end()):
                    inside[k] = True
            idx = [k for k, c in enumerate(t) if not inside[k] and not c.isspace()]
            x0, y0, x1, y1 = L["xyxy"]
            n = float(max(1, len(t)))
            lx0 = x0 + (x1 - x0) * idx[0] / n
            lx1 = x0 + (x1 - x0) * (idx[-1] + 1) / n
            # the dots are the blank to fill in: label only, value stays null
            return [_make_line(rest, [round(lx0, 1), y0, round(lx1, 1), y1],
                               role="label", src=L)], False
        # only dots (+ danda): a blank line to fill in, not a missing separator
        return [dict(L, role="label")], False
    m = _FORM_SPLIT_RE.match(t)
    if not m:
        return [dict(L, role="label")], True
    lab_t, val_t = m.group("label").strip(), (m.group("value") or "").strip()
    x0, y0, x1, y1 = L["xyxy"]
    n = float(max(1, len(t)))
    lab_x1 = x0 + (x1 - x0) * len(m.group("label")) / n
    out = [_make_line(lab_t, [x0, y0, round(lab_x1, 1), y1], role="label", src=L)]
    if val_t:
        val_x0 = x0 + (x1 - x0) * (len(t) - len(val_t)) / n
        out.append(_make_line(val_t, [round(val_x0, 1), y0, x1, y1], role="value", src=L))
    return out, False


def _synth_form(b, bid, warns):
    labels, values = [], []
    for L in b["lines"]:
        if L["role"] is None or (L["role"] == "label" and _LEADER_RE.search(L["text"] or "")):
            parts, nosep = _split_form_line(L)
            if nosep:
                warns.append(f"form block #{bid}: line '{L['text'][:30]}' has no ' -' or ':' separator "
                             "- kept as a label without value (draw form_label/form_value boxes)")
        else:
            parts = [L]
        for P in parts:
            (labels if P["role"] == "label" else values).append(P)
    labels = reading_order(labels)
    pair = {}
    for v in reading_order(values):
        vx0, vy0, vx1, vy1 = v["xyxy"]
        best, best_gap = None, None
        for lb in labels:
            if id(lb) in pair:
                continue
            lx0, ly0, lx1, ly1 = lb["xyxy"]
            ov = min(vy1, ly1) - max(vy0, ly0)
            if ov >= 0.3 * min(vy1 - vy0, ly1 - ly0) and lx0 < vx0:
                gap = vx0 - lx1
                if best is None or gap < best_gap:
                    best, best_gap = lb, gap
        if best is None:
            for lb in labels:
                if id(lb) in pair:
                    continue
                lx0, ly0, lx1, ly1 = lb["xyxy"]
                if ly1 <= vy0 + 2 and min(vx1, lx1) - max(vx0, lx0) > 0:
                    gap = vy0 - ly1
                    if best is None or gap < best_gap:
                        best, best_gap = lb, gap
        if best is None:
            warns.append(f"form block #{bid}: value '{v['text'][:30]}' has no label next to it")
            labels.append(dict(_make_line("", v["xyxy"], role="label"), _orphan_value=v))
        else:
            pair[id(best)] = v
    fields, lines, texts = [], [], []
    for lb in labels:
        v = lb.get("_orphan_value") or pair.get(id(lb))
        if lb.get("_orphan_value") is None:
            lines.append(lb)
        if v is not None:
            lines.append(v)
        fields.append({
            "label": {"text": lb["text"], "bbox": _xyxy_out(lb["xyxy"]), "quad": lb["quad"]},
            "field_type": infer_field_type(lb["text"]),
            "value": ({"text": v["text"], "bbox": _xyxy_out(v["xyxy"]), "quad": v["quad"]}
                      if v is not None else None),
        })
        texts.append((lb["text"] + " " + v["text"]).strip() if v is not None else lb["text"])
    return "\n".join(texts), lines, fields


def _synth_table(b, bid, tindex, gray, use_rulings, warns):
    d = b["det"]
    x, y, w, h = b["bbox"]
    tb = [x, y, x + w, y + h]
    cells, nrows, ncols, tr_rows = parse_table_cells(d.get("text") or "")
    lines = reading_order(b["lines"])
    if not cells:
        warns.append(f"table block #{bid}: no table HTML - run table OCR on it first")
        for L in lines:
            L["cell_id"] = None
        return "", lines, None, []
    html_out = cells_to_html(cells, tr_rows)

    unused = list(lines)
    cell_lines = {}
    pending = []
    for ci, c in enumerate(cells):
        if not c["text"]:
            continue
        hit = next((L for L in unused if _norm_text(L["text"]) == c["text"]), None)
        if hit is not None:
            unused.remove(hit)
            cell_lines[ci] = [hit]
        else:
            pending.append(ci)
    still = []
    for ci in pending:
        t, acc, chosen = cells[ci]["text"], "", []
        for L in unused:
            lt = _norm_text(L["text"])
            if not lt or lt not in t:
                continue
            nxt = (acc + " " + lt).strip()
            if t.startswith(nxt):
                acc, chosen = nxt, chosen + [L]
                if acc == t:
                    break
        if acc == t and chosen:
            for L in chosen:
                unused.remove(L)
            cell_lines[ci] = chosen
        else:
            still.append(ci)

    rext, cext = {}, {}
    for ci, ls in cell_lines.items():
        c = cells[ci]
        lo_y, hi_y = min(L["xyxy"][1] for L in ls), max(L["xyxy"][3] for L in ls)
        lo_x, hi_x = min(L["xyxy"][0] for L in ls), max(L["xyxy"][2] for L in ls)
        if c["rowspan"] == 1:
            e = rext.get(c["row"])
            rext[c["row"]] = (lo_y, hi_y) if e is None else (min(e[0], lo_y), max(e[1], hi_y))
        if c["colspan"] == 1:
            e = cext.get(c["col"])
            cext[c["col"]] = (lo_x, hi_x) if e is None else (min(e[0], lo_x), max(e[1], hi_x))
    hpos, vpos, rules = ([], [], [])
    if gray is not None and use_rulings:
        hs = sorted(L["xyxy"][3] - L["xyxy"][1] for L in lines) or [15.0]
        hpos, vpos, rules = detect_ruling_lines(gray, tb, text_h=hs[len(hs) // 2])
    rb = _grid_bounds(nrows, rext, tb[1], tb[3], hpos)
    cb = _grid_bounds(ncols, cext, tb[0], tb[2], vpos)
    rects = [[cb[c["col"]], rb[c["row"]], cb[c["col"] + c["colspan"]], rb[c["row"] + c["rowspan"]]]
             for c in cells]

    for L in list(unused):
        cx = (L["xyxy"][0] + L["xyxy"][2]) / 2.0
        cy = (L["xyxy"][1] + L["xyxy"][3]) / 2.0
        inside = [ci for ci, r in enumerate(rects) if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]]
        if not inside:
            continue
        ci = inside[0]
        cell_lines.setdefault(ci, []).append(L)
        unused.remove(L)
        warns.append(f"table #{tindex} cell {ci}: line text '{L['text'][:25]}' != HTML "
                     f"'{cells[ci]['text'][:25]}' - fix one of them")
    for L in unused:
        warns.append(f"table #{tindex}: line '{L['text'][:25]}' is in no cell")
    for ci in still:
        if ci not in cell_lines:
            warns.append(f"table #{tindex} cell {ci} '{cells[ci]['text'][:25]}' has no line box")

    ordered = []
    for ci in sorted(cell_lines):
        for L in sorted(cell_lines[ci], key=lambda L: (L["xyxy"][1], L["xyxy"][0])):
            L["cell_id"] = ci
            ordered.append(L)
    for L in unused:
        L["cell_id"] = None
        ordered.append(L)

    header_rows = 0
    for r in range(nrows):
        rc = [c for c in cells if c["row"] == r]
        if rc and all(c["is_header"] for c in rc):
            header_rows += 1
        else:
            break
    has_h = len(hpos) >= 2
    has_v = len(vpos) >= 2
    style = "full" if has_h and has_v else "horizontal" if has_h else "vertical" if has_v else "none"
    bquad = det_quad(d) if d.get("quad") else _rect_quad(*tb)
    out_cells = []
    for ci, (c, r) in enumerate(zip(cells, rects)):
        out_cells.append({
            "id": ci, "row": c["row"], "col": c["col"],
            "rowspan": c["rowspan"], "colspan": c["colspan"],
            "bbox": _xyxy_out(r), "text": c["text"], "is_header": c["is_header"],
            "cell_type": None, "empty": not c["text"], "quad": _rect_quad(*r),
        })
    tinfo = {
        "num_rows": nrows, "num_cols": ncols, "header_rows": header_rows,
        "ruling_style": style, "theme": None, "col_types": None,
        "reading_order": list(range(len(cells))), "html": html_out,
        "bbox": _xyxy_ints(tb), "cells": out_cells, "quad": bquad,
    }
    for e in rules:
        e2 = {"table_index": tindex}
        e2.update(e)
        e.clear()
        e.update(e2)
    return html_out, ordered, tinfo, rules


def estimate_block_lines(gray, bbox, texts):
    H, W = gray.shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = erase_table_rules(gray[y0:y1, x0:x1])
    _t, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (bw > 0).sum(axis=1) > max(1, 0.01 * (x1 - x0))
    runs, start = [], None
    for i, v in enumerate(list(ink) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i - 1])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 < 0.35 * min(r[1] - r[0] + 1,
                                                             merged[-1][1] - merged[-1][0] + 1):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    row_hs = sorted(r[1] - r[0] + 1 for r in merged)
    min_row_h = max(3.0, 0.4 * row_hs[len(row_hs) // 2]) if row_hs else 3.0
    rows = [r for r in merged if r[1] - r[0] + 1 >= min_row_h]
    if not rows or len(rows) > 6:
        return None
    if len(texts) == len(rows):
        parts = texts
    elif len(texts) == 1 and len(rows) > 1:
        words = texts[0].split()
        if len(words) < len(rows):
            return None
        widths = []
        for a, b2 in rows:
            cols = np.nonzero((bw[a:b2 + 1] > 0).any(axis=0))[0]
            widths.append(float(cols.max() - cols.min() + 1) if cols.size else 1.0)
        total, wl = sum(widths), [len(wd) + 1 for wd in words]
        parts, k, acc, sw = [], 0, 0.0, float(sum(wl))
        for ri in range(len(rows)):
            target = sum(widths[:ri + 1]) / total * sw
            start_k = k
            while k < len(words) - (len(rows) - ri - 1) and (acc + wl[k] <= target + 0.5 * wl[k]
                                                               or k == start_k):
                acc += wl[k]
                k += 1
            parts.append(" ".join(words[start_k:k]))
        if k < len(words):
            parts[-1] = (parts[-1] + " " + " ".join(words[k:])).strip()
        if any(not p_ for p_ in parts):
            return None
    else:
        return None
    out = []
    for (a, b2), t in zip(rows, parts):
        cols = np.nonzero((bw[a:b2 + 1] > 0).any(axis=0))[0]
        if not cols.size:
            return None
        bx0, bx1 = x0 + int(cols.min()), x0 + int(cols.max()) + 1
        xyxy, quad = [bx0, y0 + a, bx1, y0 + b2 + 1], None
        fit = tighten_to_ink(gray, (bx0, y0 + a, bx1 - bx0, b2 - a + 1))
        if fit:
            xyxy, quad, _ang = fit
        out.append(_make_line(_norm_text(t), xyxy, quad))
    return out


def build_synth_page(dets, width, height, meta, page_img=None, tighten=True, rulings=True,
                     retighten=False):
    warns = []
    dets, n_merged = merge_list_detections([dict(d) for d in dets])
    if n_merged:
        warns.append(f"{n_merged} list block(s) joined into the list they continue "
                     "(numbering 1., 2. ... / 5.1) under 5.) - press F7 to see it on screen")
    gray = None
    if page_img is not None:
        gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY) if page_img.ndim == 3 else page_img

    blocks, lines, overlays, extras = [], [], [], []
    for d in dets:
        lab = canonical_label(d.get("label"))
        if lab in SYNTH_LINE_ROLES:
            lines.append(d)
        elif lab in SYNTH_OVERLAYS:
            overlays.append(d)
        elif lab in SYNTH_EXTRAS:
            extras.append(d)
        else:
            typ, extra = SYNTH_BLOCK_MAP.get(lab, (None, None))
            if typ is None:
                warns.append(f"label '{lab}' has no synthetic type - exported as paragraph")
                typ, extra = "paragraph", {}
            if lab == "page_number" and height and \
                    d["bbox"][1] + d["bbox"][3] / 2.0 < 0.5 * float(height):
                typ = "header"
            if is_page_covering(d["bbox"], width, height):
                warns.append(f"'{lab}' box {_xyxy_ints(d['bbox'])} covers the whole page - "
                             "not exported (synthetic v2 has no page-level block; draw one "
                             "box per region instead)")
                continue
            blocks.append({"det": d, "type": typ, "extra": dict(extra),
                           "bbox": tuple(d["bbox"]), "lines": []})

    angles = []
    L_all = []
    for d in lines:
        x, y, w, h = d["bbox"]
        xyxy, quad = [x, y, x + w, y + h], None
        loaded = bool(d.get("quad") or d.get("src_xyxy"))
        t = None
        if tighten and gray is not None and (retighten or not loaded):
            t = tighten_to_ink(gray, d["bbox"])
        if t:
            xyxy, quad, ang = t
            if quad:
                angles.append(ang)
        elif loaded:
            quad = det_quad(d)
            xyxy = det_xyxy(d)
        text = _norm_text(d.get("text"))
        L = _make_line(text, xyxy, quad, SYNTH_LINE_ROLES[canonical_label(d.get("label"))])
        if d.get("ocr_text") is not None and _norm_text(d["ocr_text"]) != text:
            L["ocr_text"] = _norm_text(d["ocr_text"])
        if not text:
            warns.append(f"empty line box at {_xyxy_ints(xyxy)}")
        L_all.append(L)

    for L in L_all:
        la = max(1, _area_xywh(L["bbox"]))
        for o in overlays:
            if _inter_xywh(L["bbox"], o["bbox"]) >= 0.5 * la:
                L["flags"].update(SYNTH_OVERLAYS[canonical_label(o.get("label"))])

    dup_lines, rem_lines = set(), []
    for big in L_all:
        if not big["text"]:
            continue
        for small in L_all:
            if small is big or id(small) in dup_lines or not small["text"]:
                continue
            sa = max(1, _area_xywh(small["bbox"]))
            if (_area_xywh(big["bbox"]) > 1.5 * sa
                    and _inter_xywh(small["bbox"], big["bbox"]) >= 0.8 * sa
                    and small["text"] in big["text"]):
                dup_lines.add(id(big))
                # keep the part of the merged line that is NOT the smaller line
                # ("श्री कार्यालय प्रमुखज्यू | मिति......") as its own line box
                bt_, st_, rem, side = big["text"], small["text"], "", None
                if bt_.startswith(st_):
                    rem, side = bt_[len(st_):].strip(), "right"
                elif bt_.endswith(st_):
                    rem, side = bt_[:-len(st_)].strip(), "left"
                note = "dropped as a merged duplicate"
                if rem:
                    bx0, by0, bx1, by1 = big["xyxy"]
                    sx0, _sy0, sx1, _sy1 = small["xyxy"]
                    box = [sx1 + 1, by0, bx1, by1] if side == "right" else [bx0, by0, sx0 - 1, by1]
                    if box[2] - box[0] >= 4:
                        xyxy_r, quad_r = box, None
                        if gray is not None:
                            tt = tighten_to_ink(gray, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
                            if tt:
                                xyxy_r, quad_r, _a = tt
                        rem_lines.append(_make_line(rem, xyxy_r, quad_r, big.get("role")))
                        note = f"split - '{rem[:20]}' kept as its own line"
                warns.append(f"line '{big['text'][:30]}' {_xyxy_ints(big['xyxy'])} contains the "
                             f"line '{small['text'][:20]}' - {note}")
                break
    L_all = [L for L in L_all if id(L) not in dup_lines] + rem_lines

    folded = set()
    for inner in sorted(blocks, key=lambda b: _area_xywh(b["bbox"])):
        for outer in sorted(blocks, key=lambda b: _area_xywh(b["bbox"])):
            if outer is inner or id(outer) in folded:
                continue
            if inner["type"] not in SYNTH_FOLD_INTO.get(outer["type"], ()):
                continue
            ia = max(1, _area_xywh(inner["bbox"]))
            if _area_xywh(outer["bbox"]) > ia and _inter_xywh(inner["bbox"], outer["bbox"]) >= 0.85 * ia:
                folded.add(id(inner))
                ilab = canonical_label(inner["det"].get("label"))
                warns.append(f"'{ilab}' box {_xyxy_ints(_bb(inner['det']))} is inside a "
                             f"'{canonical_label(outer['det'].get('label'))}' box - merged into it")
                break
    blocks = [b for b in blocks if id(b) not in folded]

    wrappers = set()
    for outer in sorted(blocks, key=lambda b: -_area_xywh(b["bbox"])):
        if outer["type"] in ("table", "header", "footer") or outer["type"] in SYNTH_NO_LINES:
            continue
        if outer["det"].get("label_fixed"):
            continue
        oa = _area_xywh(outer["bbox"])
        inner = [b for b in blocks
                 if b is not outer and id(b) not in wrappers
                 and oa > _area_xywh(b["bbox"])
                 and _inter_xywh(b["bbox"], outer["bbox"]) >= 0.85 * max(1, _area_xywh(b["bbox"]))]
        if len(inner) >= 2:
            wrappers.add(id(outer))
            warns.append(f"'{canonical_label(outer['det'].get('label'))}' box "
                         f"{_xyxy_ints(_bb(outer['det']))} wraps {len(inner)} other blocks - "
                         "dropped (layout-model container region)")
    blocks = [b for b in blocks if id(b) not in wrappers]

    textual = [b for b in blocks if b["type"] not in SYNTH_NO_LINES]
    for L in L_all:
        la = max(1, _area_xywh(L["bbox"]))
        best = None
        for b in textual:
            if _inter_xywh(L["bbox"], b["bbox"]) >= 0.5 * la:
                if best is None or _area_xywh(b["bbox"]) < _area_xywh(best["bbox"]):
                    best = b
        if best is None:
            warns.append(f"line '{L['text'][:30]}' is outside every block - exported as its own paragraph")
            x0, y0, x1, y1 = L["xyxy"]
            best = {"det": {"bbox": (x0, y0, x1 - x0, y1 - y0), "order": None, "text": ""},
                    "type": "paragraph", "extra": {}, "bbox": (x0, y0, x1 - x0, y1 - y0), "lines": []}
            blocks.append(best)
        best["lines"].append(L)

    # Line boxes only 2-3 px tall (saved by older versions from underline /
    # rule remnants): join the text to the line on the same row, choosing the
    # word order that matches the block text, or refit the box to the ink.
    hs_all = sorted(L["xyxy"][3] - L["xyxy"][1] for L in L_all if L["text"])
    med_h = float(hs_all[len(hs_all) // 2]) if hs_all else 15.0
    for b in blocks:
        for L in list(b["lines"]):
            x0, y0, x1, y1 = L["xyxy"]
            if y1 - y0 >= max(4.0, 0.3 * med_h) or not L["text"]:
                continue
            cy = (y0 + y1) / 2.0
            others = [o for o in b["lines"] if o is not L
                      and o["xyxy"][1] - 0.5 * med_h <= cy <= o["xyxy"][3] + 0.5 * med_h]
            if others:
                o = min(others, key=lambda o: abs((o["xyxy"][1] + o["xyxy"][3]) / 2.0 - cy))
                want = _norm_text(strip_heading_prefix(b["det"].get("text") or ""))
                a1 = (L["text"] + " " + o["text"]).strip()
                a2 = (o["text"] + " " + L["text"]).strip()
                if a1 in want and a2 not in want:
                    o["text"] = a1
                elif a2 in want:
                    o["text"] = a2
                else:
                    o["text"] = a1 if x0 < o["xyxy"][0] else a2
                nx0, nx1 = min(o["xyxy"][0], x0), max(o["xyxy"][2], x1)
                if (nx0, nx1) != (o["xyxy"][0], o["xyxy"][2]):
                    o["xyxy"] = [nx0, o["xyxy"][1], nx1, o["xyxy"][3]]
                    o["bbox"] = (nx0, o["xyxy"][1], nx1 - nx0, o["xyxy"][3] - o["xyxy"][1])
                    o["quad"] = _rect_quad(*o["xyxy"])
                b["lines"].remove(L)
                warns.append(f"line '{L['text'][:25]}' was {y1 - y0:g} px tall - joined to "
                             f"'{o['text'][:30]}'")
            elif gray is not None:
                ex = [max(0, x0 - med_h), max(0, cy - 0.6 * med_h), x1 + med_h, cy + 0.6 * med_h]
                tt = tighten_to_ink(gray, (ex[0], ex[1], ex[2] - ex[0], ex[3] - ex[1]))
                if tt and tt[0][3] - tt[0][1] > y1 - y0:
                    L["xyxy"] = list(tt[0])
                    L["bbox"] = (tt[0][0], tt[0][1], tt[0][2] - tt[0][0], tt[0][3] - tt[0][1])
                    L["quad"] = tt[1] or _rect_quad(*tt[0])
                    warns.append(f"line '{L['text'][:25]}' was {y1 - y0:g} px tall - refitted "
                                 f"to {_xyxy_ints(L['xyxy'])}")

    heads = [b for b in blocks if b["type"] in SYNTH_FIRST_TYPES]
    feet = [b for b in blocks if b["type"] in SYNTH_LAST_TYPES]
    body = [b for b in blocks if b["type"] not in SYNTH_NO_ORDER
            and b["type"] not in SYNTH_FIRST_TYPES and b["type"] not in SYNTH_LAST_TYPES]
    ordered = (reading_order(heads) + _synth_block_order(body) + reading_order(feet)
               + reading_order([b for b in blocks if b["type"] in SYNTH_NO_ORDER]))

    layout, out_blocks, out_lines, tables, rules_all = [], [], [], [], []
    tindex = 0
    ro = 0
    raw = {"paragraph_chars": 0, "list_items": 0, "form_fields": 0, "table_cells": 0}
    for bid, b in enumerate(ordered):
        typ, d, extra = b["type"], b["det"], b["extra"]
        shrunk = False
        if b["lines"] and typ != "table" and typ not in SYNTH_NO_LINES:
            lx0 = min(L["xyxy"][0] for L in b["lines"])
            ly0 = min(L["xyxy"][1] for L in b["lines"])
            lx1 = max(L["xyxy"][2] for L in b["lines"])
            ly1 = max(L["xyxy"][3] for L in b["lines"])
            lhs = sorted(L["xyxy"][3] - L["xyxy"][1] for L in b["lines"])
            pad = max(2, int(round(0.25 * lhs[len(lhs) // 2])))
            nx0, ny0 = max(0, int(lx0) - pad), max(0, int(ly0) - pad)
            nx1, ny1 = int(round(lx1)) + pad, int(round(ly1)) + pad
            if width:
                nx1 = min(int(width), nx1)
            if height:
                ny1 = min(int(height), ny1)
            b["bbox"] = (nx0, ny0, max(1, nx1 - nx0), max(1, ny1 - ny0))
            shrunk = True
        x, y, w, h = b["bbox"]
        bxyxy = _xyxy_ints([x, y, x + w, y + h])
        bquad = det_quad(d) if d.get("quad") and not shrunk else _rect_quad(*bxyxy)
        ent = {"id": bid, "type": typ, "bbox": bxyxy,
               "reading_order": None if typ in SYNTH_NO_ORDER else ro}
        if typ not in SYNTH_NO_ORDER:
            ro += 1
        if extra.get("handwritten"):
            for L in b["lines"]:
                L["flags"]["handwritten"] = True

        if typ in SYNTH_NO_LINES:
            blines = []
            if b["lines"]:
                warns.append(f"{typ} block #{bid} contains line boxes - they were dropped")
            text = "<figure/>" if typ == "figure" else None
        elif typ == "table":
            text, blines, tinfo, rules = _synth_table(b, bid, tindex, gray, rulings, warns)
            ent["text"] = text
            ent["table_index"] = tindex
            if tinfo:
                tables.append(tinfo)
                raw["table_cells"] += len(tinfo["cells"])
            rules_all.extend(rules)
            tindex += 1
        elif typ == "form_field":
            if b["lines"]:
                text, blines, fields = _synth_form(b, bid, warns)
            else:
                text, blines, fields = (clean_block_content(d.get("text", ""), "form_field")[0], [], [])
                warns.append(f"form block #{bid} has no line boxes - fields list is empty")
            ent["text"] = text
            ent["fields"] = fields
            raw["form_fields"] += len(fields)
        else:
            blines = reading_order(b["lines"])
            if typ == "heading":
                for L in blines:
                    L["text"] = strip_heading_prefix(L["text"])
            texts = [L["text"] for L in blines if L["text"]]
            if not blines:
                src = clean_block_content(d.get("text", ""), d.get("label", "paragraph"))[0]
                texts = [strip_heading_prefix(t) if typ == "heading" else t.strip()
                         for t in src.split("\n") if t.strip()]
                texts = [t for t in texts if t]
                est = estimate_block_lines(gray, b["bbox"], texts) if gray is not None and texts else None
                if est:
                    blines = est
                    warns.append(f"{typ} block #{bid} had no line boxes - {len(est)} estimated "
                                 "from the ink (press F6 on it for OCR'd lines)")
                else:
                    warns.append(f"{typ} block #{bid} has no line boxes"
                                 + (" - text taken from the block" if texts else " and no text"))
            if typ == "paragraph":
                text = paragraph_text(blines, texts)
                raw["paragraph_chars"] += len(text)
            elif typ == "heading":
                text = format_heading(" ".join(texts), extra.get("level", 1))
            elif typ == "list":
                list_items_out, list_lines = None, {}
                if blines:
                    list_items_out, list_lines = flat_items(build_list_tree(blines))
                    text = items_text(list_items_out)
                    raw["list_items"] += len(list_items_out)
                else:
                    items = _list_items(texts)
                    text = "\n".join(items)
                    raw["list_items"] += len(items)
            else:
                text = "\n".join(texts)
        if typ not in ("table", "form_field") and text is not None:
            ent["text"] = text
        if typ == "heading":
            ent["level"] = int(extra.get("level", 1))
        if typ == "list":
            ent["list_style"] = list_style_of(ent.get("text", ""))
            if list_items_out:
                ent["items"] = list_items_out
        if extra.get("handwritten"):
            ent["handwritten"] = True
        ent["quad"] = bquad
        layout.append(ent)
        out_blocks.append({"type": typ, "bbox": bxyxy, "block_id": bid, "font_id": None, "quad": bquad})

        for k, L in enumerate(blines):
            le = {"text": L["text"], "bbox": _xyxy_out(L["xyxy"]), "block_id": bid,
                  "line_in_block": k, "is_heading": typ == "heading",
                  "font_size": None, "font_id": None}
            if typ == "table":
                le["table_index"] = tindex - 1
                le["cell_id"] = L.get("cell_id")
            if typ == "form_field" and L.get("role"):
                le["role"] = L["role"]
            if typ == "list" and id(L) in list_lines:
                le["list_level"] = list_lines[id(L)][1]
                le["item_index"] = list_lines[id(L)][0]
            le["quad"] = L["quad"]
            le.update(L.get("flags") or {})
            if L.get("ocr_text"):
                le["ocr_text"] = L["ocr_text"]
            out_lines.append(le)

    page_text = "\n\n".join(e["text"] for e in layout
                            if e["reading_order"] is not None and e.get("text"))
    lang, digits = detect_language(page_text)
    forced = meta.get("language")
    if forced in ("ne", "en"):
        lang = forced
        if not re.search("[0-9\u0966-\u096f]", re.sub(r"<[^>]+>", " ", page_text)):
            digits = "dev" if forced == "ne" else "latin"
    extra_regions = []
    for d in extras:
        x, y, w, h = d["bbox"]
        extra_regions.append({"type": canonical_label(d.get("label")),
                              "bbox": _xyxy_ints([x, y, x + w, y + h]), "quad": det_quad(d)})
    stamps = [e["bbox"] for e in layout if e["type"] == "stamp"]
    sigs = [e["bbox"] for e in extra_regions if e["type"] == "signature"]

    is_gray = False
    is_bin = False
    if page_img is not None:
        if page_img.ndim == 2:
            is_gray = True
        else:
            if np.array_equal(page_img[..., 0], page_img[..., 1]) and \
               np.array_equal(page_img[..., 0], page_img[..., 2]):
                is_gray = True
        if is_gray:
            g = page_img if page_img.ndim == 2 else cv2.cvtColor(
                page_img, cv2.COLOR_BGR2GRAY)
            sample = g[::8, ::8]
            is_bin = len(np.unique(sample)) <= 2

    quality = {
        "skew_deg":         round(float(np.median(angles)), 2) if angles else 0.0,
        "perspective":      False,
        "is_clean":         True,
        "is_binarized":     bool(is_bin),
        "is_grayscale":     bool(is_gray),
        "has_fold":         False,
        "has_bleedthrough": False,
        "ops":              [],
        "downscale":        None,
        "noise_sigma":      None,
        "text_contrast":    None,
        "jpeg_quality":     None,
        "has_stamp":        bool(stamps),
        "has_signature":    bool(sigs),
        "line_count":       len(out_lines),
        "table_count":      len(tables),
        "figure_count":     sum(1 for e in layout if e["type"] == "figure"),
        "stamp_bboxes":     stamps,
        "signature_bboxes": sigs,
        "raw_lengths":      raw,
    }
    rec = {
        "v": SYNTH_SCHEMA_VERSION,
        "image": meta.get("image"),
        "width": int(width), "height": int(height),
        "document_id": meta.get("document_id"),
        "page_number": meta.get("page_number"),
        "page_count": meta.get("page_count"),
        "language": lang, "digit_style": digits,
        "text": page_text,
        "fonts": [], "font_ids": [],
        "layout": layout, "blocks": out_blocks, "lines": out_lines,
        "tables": tables, "ruling_lines": rules_all, "quality": quality,
    }
    if extra_regions:
        rec["extra_regions"] = extra_regions
    rec["annotation"] = {"source": "real", "tool": "ocr_tool",
                         "status": meta.get("status", "draft")}
    return rec, warns


def synth_record_to_blocks(item):
    pn = _num(item.get("page_number"))
    meta = {
        "page": int(pn) - 1 if pn is not None and pn >= 1 else 0,
        "image": item.get("image"),
        "width": _num(item.get("width")), "height": _num(item.get("height")),
        "zoom": None,
        "doc_meta": {"document_id": item.get("document_id"),
                     "page_number": item.get("page_number"),
                     "page_count": item.get("page_count"),
                     "language": item.get("language")},
    }
    out = []

    def add(text, bb, label, order, quad, extra=None):
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            return
        x0, y0, x1, y1 = [float(v) for v in bb]
        box = (int(round(x0)), int(round(y0)),
               max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))
        d = {"text": normalize_text(text or ""), "conf": 100, "bbox": box,
             "label": label, "order": order, "status": "imported",
             "quad_bbox": box, "src_xyxy": [x0, y0, x1, y1]}
        if isinstance(quad, (list, tuple)) and len(quad) == 4:
            try:
                d["quad"] = [[float(p[0]), float(p[1])] for p in quad]
            except (TypeError, ValueError, IndexError):
                pass
        if extra:
            d.update(extra)
        out.append(dict(meta, det=d))

    for e in item.get("layout") or []:
        if not isinstance(e, dict):
            continue
        typ = e.get("type")
        text = e.get("text") or ""
        if typ == "heading":
            label = SYNTH_LEVEL_TO_LABEL.get(e.get("level"), "heading")
            text = strip_heading_prefix(text)
        else:
            label = SYNTH_TYPE_TO_LABEL.get(typ, typ or "paragraph")
        if typ in SYNTH_NO_LINES:
            text = ""
        r = e.get("reading_order")
        add(text, e.get("bbox"), label, r + 1 if isinstance(r, int) else None, e.get("quad"))
    for ln in item.get("lines") or []:
        if not isinstance(ln, dict):
            continue
        label = {"label": "form_label", "value": "form_value"}.get(ln.get("role"), "text_line")
        extra = {"ocr_text": normalize_text(ln["ocr_text"])} if ln.get("ocr_text") else None
        add(ln.get("text"), ln.get("bbox"), label, None, ln.get("quad"), extra)
    ex = item.get("extra_regions") or []
    for e in ex:
        if isinstance(e, dict):
            add("", e.get("bbox"), e.get("type") or "signature", None, e.get("quad"))
    if not any(isinstance(e, dict) and e.get("type") == "signature" for e in ex):
        for sb in (item.get("quality") or {}).get("signature_bboxes") or []:
            add("", sb, "signature", None, None)
    return out


_JUNK_ALWAYS = set("|[]{}=_\u00a6")
_JUNK_LOWCONF = _JUNK_ALWAYS | set("\u2018\u2019\"'`~\u2014-.,:;!()/\\<>")


def is_junk_token(text, conf=100):
    t = (text or "").strip()
    if not t:
        return True
    if all(c in _JUNK_ALWAYS for c in t):
        return True
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 100.0
    return conf < 50 and all(c in _JUNK_LOWCONF or c.isspace() for c in t)


def erase_table_rules(bgr):
    if bgr is None or bgr.size == 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    H, W = gray.shape[:2]
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)
    out = bgr.copy()
    found = 0
    for orient, minlen in (("h", max(40, W // 12)), ("v", max(40, H // 30))):
        for lx, ly, lw, lh, _a in _rule_components(bw, orient, minlen, 6.0):
            out[max(0, ly - 1):ly + lh + 1, max(0, lx - 1):lx + lw + 1] = 255
            found += 1
    return out if found else bgr


def group_words_into_lines(words, gap_factor=3.0):
    if not words:
        return []
    allh = sorted(w["bbox"][3] for w in words)
    big = 1.8 * allh[len(allh) // 2]
    rows = []
    normal = [w for w in words if w["bbox"][3] <= big]
    tall = [w for w in words if w["bbox"][3] > big]
    for w in sorted(normal, key=lambda w: (w["bbox"][1], w["bbox"][0])):
        x, y, bw, bh = w["bbox"]
        best, best_ov = None, 0
        for r in rows:
            ov = min(r["y1"], y + bh) - max(r["y0"], y)
            if ov >= 0.5 * min(bh, r["y1"] - r["y0"]) and ov > best_ov:
                best, best_ov = r, ov
        if best is None:
            rows.append({"y0": y, "y1": y + bh, "words": [w]})
        else:
            best["words"].append(w)
            best["y0"], best["y1"] = min(best["y0"], y), max(best["y1"], y + bh)
    for w in tall:
        cy = w["bbox"][1] + w["bbox"][3] / 2.0
        inside = [r for r in rows if r["y0"] <= cy <= r["y1"]]
        if inside:
            min(inside, key=lambda r: abs((r["y0"] + r["y1"]) / 2.0 - cy))["words"].append(w)
        else:
            rows.append({"y0": w["bbox"][1], "y1": w["bbox"][1] + w["bbox"][3], "words": [w]})
    lines = []
    for r in sorted(rows, key=lambda r: r["y0"]):
        ws = sorted(r["words"], key=lambda w: w["bbox"][0])
        hs = sorted(w["bbox"][3] for w in ws)
        mh = hs[len(hs) // 2]
        cur = [ws[0]]
        chunks = [cur]
        for a, b in zip(ws, ws[1:]):
            if b["bbox"][0] - (a["bbox"][0] + a["bbox"][2]) > gap_factor * mh:
                cur = [b]
                chunks.append(cur)
            else:
                cur.append(b)
        for ch in chunks:
            lines.append(_line_from_words(ch))
    return lines


def _line_from_words(ws, label="text_line"):
    x0 = min(w["bbox"][0] for w in ws)
    y0 = min(w["bbox"][1] for w in ws)
    x1 = max(w["bbox"][0] + w["bbox"][2] for w in ws)
    y1 = max(w["bbox"][1] + w["bbox"][3] for w in ws)
    cs = [w.get("conf", 100) for w in ws]
    hs = sorted(w["bbox"][3] for w in ws)
    return {"text": " ".join(w["text"] for w in ws if w["text"]),
            "bbox": (int(round(x0)), int(round(y0)),
                     max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))),
            "conf": int(sum(cs) / len(cs)), "label": label, "words": ws,
            "fh": round(float(hs[len(hs) // 2]), 1)}


def split_form_rows(lines):
    out = []
    for ln in lines:
        ws = ln["words"]
        cut = None
        for i, w in enumerate(ws):
            t = w["text"].strip()
            if t in _FORM_SEP_WORDS or t.endswith((":", "：")) or (
                    t.endswith("-") and len(t) > 1 and not t[-2].isdigit()):
                cut = i
                break
        if cut is None and len(ws) > 1:
            gaps = [(ws[i + 1]["bbox"][0] - (ws[i]["bbox"][0] + ws[i]["bbox"][2]), i)
                    for i in range(len(ws) - 1)]
            g, i = max(gaps)
            if g > 1.5 * ln["bbox"][3]:
                cut = i
        if cut is None:
            out.append(ln)
            continue
        out.append(_line_from_words(ws[:cut + 1], "form_label"))
        if ws[cut + 1:]:
            out.append(_line_from_words(ws[cut + 1:], "form_value"))
    return out


# ======================================================================
# AUTO LAYOUT
# ======================================================================
QUICK_LABELS = [
    ("title", "Title", "1"), ("heading", "Heading", "2"), ("subheading", "Subheading", "3"),
    ("paragraph", "Paragraph", "4"), ("list", "List", "5"), ("table", "Table", "6"),
    ("header", "Header", "7"), ("footer", "Footer", "8"), ("form_field", "Form", "9"),
    ("logo", "Logo", "0"), ("stamp", "Stamp", None), ("image", "Figure", None),
    ("page_number", "Page no.", None), ("text_line", "Line", None),
]
PAGE_LANGS = {"ne": ("Nepali", "nep+eng"), "en": ("English", "eng")}
DEFAULT_PAGE_LANG = "ne"
LABEL_MENU = [q[0] for q in QUICK_LABELS] + [l for l in PADDLE_LABELS
                                             if l not in {q[0] for q in QUICK_LABELS}]

HEADER_KEYWORDS = [
    "कार्यालय", "नगरपालिका", "गाउँपालिका", "गाउपालिका", "सरकार", "मन्त्रालय", "विभाग",
    "प्रदेश", "महानगरपालिका", "उपमहानगरपालिका", "वडा कार्यालय", "निर्देशनालय", "आयोग",
    "government", "office", "municipality", "ministry", "department", "directorate",
    "commission", "province",
]
_PAGE_NUM_RE = re.compile(
    r"^[\s\-–—(\[]*(?:page|pg\.?|पृष्ठ|पेज)?\s*[0-9०-९]{1,4}"
    r"(?:\s*(?:/|of|मध्ये)\s*[0-9०-९]{1,4})?[\s\-–—)\]]*$", re.I)

AUTO_HEADER_ZONE = 0.15
AUTO_FOOTER_ZONE = 0.85
AUTO_TITLE_RATIO = 1.6
AUTO_HEADING_RATIO = 1.3
AUTO_SUBHEADING_RATIO = 1.2
AUTO_MAX_HEIGHT_RATIO = 1.7


def _fh(d):
    return float(d.get("fh") or d["bbox"][3])


def body_line_height(lines, default=15.0):
    pairs = sorted((_fh(d), max(1, len(_norm_text(d.get("text"))))) for d in lines)
    if not pairs:
        return float(default)
    half, acc = sum(n for _h, n in pairs) / 2.0, 0
    for h, n in pairs:
        acc += n
        if acc >= half:
            return float(h)
    return float(pairs[-1][0])


def _is_strong_text(d):
    t = _norm_text(d.get("text"))
    try:
        conf = float(d.get("conf", 100))
    except (TypeError, ValueError):
        conf = 100.0
    return conf >= 55 and sum(1 for c in t if c.isalnum()) >= 2


def _bb(d):
    x, y, w, h = d["bbox"]
    return x, y, x + w, y + h


def _median(vals, default=1.0):
    v = sorted(x for x in vals if x)
    return float(v[len(v) // 2]) if v else float(default)


def _center_inside(d, box_xywh):
    x, y, w, h = box_xywh
    cx = d["bbox"][0] + d["bbox"][2] / 2.0
    cy = d["bbox"][1] + d["bbox"][3] / 2.0
    return x <= cx <= x + w and y <= cy <= y + h


def _union_bbox(items, pad=0, W=None, H=None):
    x0 = min(_bb(d)[0] for d in items) - pad
    y0 = min(_bb(d)[1] for d in items) - pad
    x1 = max(_bb(d)[2] for d in items) + pad
    y1 = max(_bb(d)[3] for d in items) + pad
    x0, y0 = max(0, x0), max(0, y0)
    if W:
        x1 = min(W, x1)
    if H:
        y1 = min(H, y1)
    return (int(round(x0)), int(round(y0)),
            max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))


def line_bands(dets):
    items = sorted(dets, key=lambda d: (d["bbox"][1], d["bbox"][0]))
    bands = []
    for d in items:
        x, y, w, h = d["bbox"]
        if bands:
            b = bands[-1]
            if min(b["y1"], y + h) - max(b["y0"], y) >= 0.5 * min(h, b["h"]):
                b["items"].append(d)
                b["y1"] = max(b["y1"], y + h)
                continue
        bands.append({"y0": y, "y1": y + h, "h": h, "items": [d]})
    return [sorted(b["items"], key=lambda d: d["bbox"][0]) for b in bands]


def _is_list_line(t):
    return bool(_LIST_MARKER_RE.match(t or ""))


def _is_form_line(t):
    t = (t or "").strip()
    m = _FORM_SPLIT_RE.match(t)
    if not m:
        return False
    lab = m.group("label").strip()
    return 1 < len(lab) <= 40 and len(lab) <= 0.75 * len(t) and len(t) <= 90


def _is_page_number(t):
    return bool(_PAGE_NUM_RE.match((t or "").strip()))


def table_html_from_lines(lines):
    import html as _html
    bands = line_bands(lines)
    if len(bands) < 2:
        return ""
    widest = max(len(b) for b in bands)
    if widest < 2:
        return ""
    hmed = _median([d["bbox"][3] for d in lines], 10)
    iv = sorted((_bb(d)[0], _bb(d)[2]) for b in bands if len(b) >= widest - (widest > 2)
                for d in b)
    spans = []
    for x0, x1 in iv:
        if spans and x0 <= spans[-1][1] + 0.3 * hmed:
            spans[-1][1] = max(spans[-1][1], x1)
        else:
            spans.append([x0, x1])
    if len(spans) < 2:
        return ""

    def col_of(d):
        cx = (_bb(d)[0] + _bb(d)[2]) / 2.0
        for i, (a, b) in enumerate(spans):
            if a <= cx <= b:
                return i
        return min(range(len(spans)),
                   key=lambda i: min(abs(cx - spans[i][0]), abs(cx - spans[i][1])))

    out = ["<table>"]
    for r, band in enumerate(bands):
        cells = [[] for _ in spans]
        for d in band:
            cells[col_of(d)].append(_norm_text(d.get("text")))
        tag = "th" if r == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{_html.escape(' '.join(c).strip(), quote=False)}</{tag}>"
                                    for c in cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def find_tables(lines, hmed):
    bands = line_bands(lines)
    runs, cur = [], []
    for b in bands:
        top = min(d["bbox"][1] for d in b)
        if len(b) >= 2 and cur and top - max(_bb(d)[3] for d in cur[-1]) <= 3.0 * hmed:
            cur.append(b)
            continue
        if cur:
            runs.append(cur)
        cur = [b] if len(b) >= 2 else []
    if cur:
        runs.append(cur)
    tables, used = [], set()
    for run in runs:
        while run and len(run[0]) < 3:
            run = run[1:]
        while run and len(run[-1]) < 3:
            run = run[:-1]
        if len(run) < 3 or sum(1 for b in run if len(b) >= 3) < 2:
            continue
        tl = [d for b in run for d in b]
        th = _median([d["bbox"][3] for d in tl], hmed)
        for _pass in range(2):
            tx0, ty0, tw, tht = _union_bbox(tl)
            tx1, ty1 = tx0 + tw, ty0 + tht
            ids = {id(d) for d in tl}
            for d in lines:
                if id(d) in used or id(d) in ids:
                    continue
                x0, y0, x1, y1 = _bb(d)
                cx = (x0 + x1) / 2.0
                inside = _center_inside(d, (tx0, ty0, tw, tht))
                above = (tx0 <= cx <= tx1 and (x1 - x0) < 0.5 * tw
                         and y0 < ty0 and -0.5 * th <= ty0 - y1 <= 1.5 * th)
                if inside or above:
                    tl.append(d)
        html = table_html_from_lines(tl)
        if html:
            tables.append((tl, html))
            used.update(id(d) for d in tl)
    return tables, [d for d in lines if id(d) not in used]


def heal_split_lines(lines, hmed):
    lines = list(lines)
    changed = True
    while changed:
        changed = False
        for band in line_bands(lines):
            for a, b in zip(band, band[1:]):
                ax0, ay0, ax1, ay1 = _bb(a)
                bx0, by0, bx1, by1 = _bb(b)
                if bx0 - ax1 > 8 * hmed:
                    continue
                cover = False
                for o in lines:
                    if o is a or o is b:
                        continue
                    ox0, oy0, ox1, oy1 = _bb(o)
                    near = (0 <= oy0 - max(ay1, by1) <= 2.0 * hmed
                            or 0 <= min(ay0, by0) - oy1 <= 2.0 * hmed)
                    if near and ox0 <= ax1 and ox1 >= bx0:
                        cover = True
                        break
                if not cover:
                    continue
                m = dict(a)
                m["text"] = (_norm_text(a.get("text")) + " " + _norm_text(b.get("text"))).strip()
                if a.get("ocr_text") or b.get("ocr_text"):
                    m["ocr_text"] = ((a.get("ocr_text") or a.get("text") or "") + " "
                                     + (b.get("ocr_text") or b.get("text") or "")).strip()
                m["bbox"] = _union_bbox([a, b])
                m["conf"] = min(a.get("conf", 100), b.get("conf", 100))
                m["fh"] = _median([_fh(a), _fh(b)])
                lines = [o for o in lines if o is not a and o is not b] + [m]
                changed = True
                break
            if changed:
                break
    return lines


def is_graphic_noise(d, hmed):
    if d.get("page_number"):
        return False
    t = _norm_text(d.get("text"))
    conf = d.get("conf", 100)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 100.0
    if len(t) <= 2 and conf < 40:
        return True
    return len(t) <= 4 and conf < 50 and _fh(d) > 2.5 * hmed


def _centered_pair(a, b):
    ax0, _ay0, ax1, _ay1 = _bb(a)
    bx0, _by0, bx1, _by1 = _bb(b)
    f = max(1.0, min(_fh(a), _fh(b)))
    return (abs((ax0 + ax1) / 2.0 - (bx0 + bx1) / 2.0) <= max(0.6 * f, 12)
            and abs(ax0 - bx0) > 1.0 * f)


def _split_paragraphs(chain, hmed):
    if len(chain) < 2:
        return [chain]
    is_list = _is_list_line(chain[0].get("text"))
    widths = [d["bbox"][2] for d in chain]
    maxw = max(widths)
    gaps = [_bb(b)[1] - _bb(a)[3] for a, b in zip(chain, chain[1:])]
    gs = sorted(max(0.0, g) for g in gaps)
    medgap = gs[len(gs) // 2] if len(gs) > 1 else None
    out = [[chain[0]]]
    for k, (a, b) in enumerate(zip(chain, chain[1:])):
        brk = False
        if not is_list and not _centered_pair(a, b):
            if a["bbox"][2] < 0.7 * maxw and b["bbox"][2] >= 0.8 * maxw:
                brk = True
            if b["bbox"][0] - a["bbox"][0] > 0.8 * hmed and b["bbox"][2] >= 0.8 * maxw \
                    and a["bbox"][2] >= 0.8 * maxw:
                brk = True
        if medgap is not None and gaps[k] > max(2.0 * medgap, medgap + 0.8 * hmed):
            brk = True
        if brk:
            out.append([b])
        else:
            out[-1].append(b)
    return out


def _starts_paragraph(b, items):
    bx0, by0, bx1, by1 = _bb(b)
    f = _fh(b)
    for c in items:
        if c is b:
            continue
        cx0, cy0, cx1, cy1 = _bb(c)
        if 0 <= cy0 - by1 <= 1.5 * f and abs(cx0 - bx0) <= 0.5 * f and (cx1 - cx0) >= 0.7 * (bx1 - bx0):
            return True
    return False


def chain_lines(lines, hmed):
    items = sorted(lines, key=lambda d: (d["bbox"][1], d["bbox"][0]))
    tall = 1.3 * hmed
    n = len(items)
    pairs = []
    for j in range(n):
        bx0, by0, bx1, by1 = _bb(items[j])
        bh, bf = by1 - by0, _fh(items[j])
        for i in range(j):
            ax0, ay0, ax1, ay1 = _bb(items[i])
            ah, af = ay1 - ay0, _fh(items[i])
            mh = max(1.0, min(ah, bh))
            gap = by0 - ay1
            if gap < -0.5 * mh or gap > max(2.0 * min(af, bf), 1.0 * hmed):
                continue
            mf = max(1.0, min(af, bf))
            centered = _centered_pair(items[i], items[j]) and not _starts_paragraph(items[j], items)
            if not centered and (max(af, bf) > AUTO_MAX_HEIGHT_RATIO * mf
                                 or ((af >= tall) != (bf >= tall))):
                continue
            if centered and max(af, bf) > 3.0 * mf:
                continue
            if not centered and af >= tall and bf >= tall and max(af, bf) > 1.2 * mf:
                continue
            ov = min(ax1, bx1) - max(ax0, bx0)
            if ov < 0.3 * max(1.0, min(ax1 - ax0, bx1 - bx0)):
                continue
            pairs.append((gap, -ov, i, j))
    pairs.sort(key=lambda t: (t[3], t[0], t[1]))
    down, up = {}, {}
    for _g, _o, i, j in pairs:
        if i in down or j in up:
            continue
        top = i
        while top in up:
            top = up[top]
        if _is_list_line(items[j].get("text")) and not _is_list_line(items[top].get("text")):
            continue
        down[i], up[j] = j, i
    blocks = []
    for k in range(n):
        if k in up:
            continue
        chain = [items[k]]
        while k in down:
            k = down[k]
            chain.append(items[k])
        blocks.extend(_split_paragraphs(chain, hmed))
    return blocks


def classify_text_block(blines, hmed):
    texts = [_norm_text(d.get("text")) for d in blines if _norm_text(d.get("text"))]
    if not texts:
        return "paragraph", 1.0
    rs = sorted(_fh(d) / max(1.0, hmed) for d in blines)
    r = rs[-1]
    r_typ = rs[(len(rs) - 1) // 2]
    n = len(texts)
    whole = " ".join(texts).strip()
    caption = whole[:1] in "([{" and whole[-1:] in ")]}"
    if r_typ >= AUTO_HEADING_RATIO and n <= 3 and len(whole) <= 120 and not caption:
        return "heading", r
    if _is_list_line(texts[0]) or sum(map(_is_list_line, texts)) * 2 >= n + (n > 1):
        return "list", r
    if n >= 2 and sum(map(_is_form_line, texts)) >= 0.6 * n:
        return "form_field", r
    t = texts[0]
    if (n == 1 and r >= AUTO_SUBHEADING_RATIO and len(t) <= 60 and not caption
            and not t.rstrip().endswith(("।", ".", ",", ";"))):
        return "subheading", r
    return "paragraph", r


def stacked_centered(blines):
    """Short lines centred under each other (cover pages, addresses, signature
    blocks) - not a running paragraph, so their line breaks are kept."""
    bl = [d for d in blines if _norm_text(d.get("text"))]
    if len(bl) < 2:
        return False
    xs = [(_bb(d)[0], _bb(d)[2]) for d in bl]
    left, right = min(a for a, _ in xs), max(b for _, b in xs)
    width = max(1.0, right - left)
    centers = [(a + b) / 2.0 for a, b in xs]
    lefts = [a for a, _ in xs]
    widths = [b - a for a, b in xs]
    return (max(centers) - min(centers) <= 0.08 * width
            and max(lefts) - min(lefts) >= 0.1 * width
            and min(widths) <= 0.8 * max(widths))


def paragraph_text(blines, texts):
    return ("\n" if stacked_centered(blines) else " ").join(texts)


def promote_page_title(dets, H):
    """No title yet: the heading whose text is clearly the biggest on the page,
    in its top half, becomes the title (level 1). Judged against ALL text lines,
    so it also works on cover pages with little body text. Labels the
    annotator set by hand are never changed."""
    labs = [canonical_label(d.get("label")) for d in dets]
    if "title" in labs:
        return None
    lines = [d for d, l in zip(dets, labs) if l in LINE_LABELS and _norm_text(d.get("text"))]
    heads = [d for d, l in zip(dets, labs) if l == "heading"]
    if not heads or not lines:
        return None

    def inside(L, b):
        x, y, w, h = b["bbox"]
        cx, cy = L["bbox"][0] + L["bbox"][2] / 2.0, L["bbox"][1] + L["bbox"][3] / 2.0
        return x <= cx <= x + w and y <= cy <= y + h

    def text_h(b):
        hs = sorted(L["bbox"][3] for L in lines if inside(L, b))
        return float(hs[len(hs) // 2]) if hs else float(b["bbox"][3])

    sizes = [(text_h(b), b) for b in heads]
    big_h, big = max(sizes, key=lambda t: (t[0], -t[1]["bbox"][1]))
    others = [h for h, b in sizes if b is not big]
    body = sorted(L["bbox"][3] for L in lines if not inside(L, big))
    if not body:
        return None
    body_h = body[(len(body) - 1) // 2]
    if (big_h >= 1.3 * body_h and all(big_h >= 1.2 * h for h in others)
            and big["bbox"][1] < 0.45 * H and not big.get("label_fixed")):
        big["label"] = "title"
        return big
    return None


def block_text_from_lines(label, blines):
    texts = [_norm_text(d.get("text")) for d in reading_order(blines)]
    texts = [t for t in texts if t]
    typ = SYNTH_BLOCK_MAP.get(canonical_label(label), ("paragraph", {}))[0]
    if typ == "heading":
        return " ".join(strip_heading_prefix(t) for t in texts)
    if typ == "list":
        return list_block_text([d for d in blines if _norm_text(d.get("text"))])
    if typ in ("header", "footer", "form_field"):
        return "\n".join(texts)
    return paragraph_text(blines, texts)


def detect_graphic_regions(gray, text_boxes, W, H, hmed, exclude=()):
    if gray is None:
        return []
    gray = erase_table_rules(gray)
    _t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = bw.copy()
    for (x, y, w, h) in list(text_boxes) + list(exclude):
        x0, y0 = max(0, int(x) - 2), max(0, int(y) - 2)
        m[y0:int(y + h) + 3, x0:int(x + w) + 3] = 0
    k = max(3, int(round(0.6 * hmed)))
    dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(dil, 8)
    min_side = max(2.5 * hmed, 0.03 * min(W, H))
    out = []
    for i in range(1, n):
        x, y, w, h, _a = [int(v) for v in st[i]]
        if w < min_side or h < min_side or w > 0.95 * W or h > 0.9 * H:
            continue
        if not (0.2 <= w / float(h) <= 5.0):
            continue
        fill = float(np.count_nonzero(m[y:y + h, x:x + w])) / float(w * h)
        if not (0.06 <= fill <= 0.9):
            continue
        out.append((_graphic_label((x, y, w, h), W, H, text_boxes), (x, y, w, h)))
    return out


def detect_color_marks(bgr, W, H, hmed):
    if bgr is None or bgr.ndim != 3:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    mask = ((sat > 90) & (val > 50) & (val < 250)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    k = max(3, int(round(0.8 * hmed)))
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(dil, 8)
    min_side = max(2.0 * hmed, 0.025 * min(W, H))
    out = []
    for i in range(1, n):
        x, y, w, h, _a = [int(v) for v in st[i]]
        if w < min_side or h < min_side or w > 0.95 * W or h > 0.9 * H:
            continue
        if not (0.2 <= w / float(h) <= 5.0):
            continue
        if np.count_nonzero(mask[y:y + h, x:x + w]) < 0.03 * w * h:
            continue
        out.append((x, y, w, h))
    return out


def detect_picture_regions(gray, bgr, text_boxes, W, H, hmed):
    if gray is None:
        return []
    src = bgr if (bgr is not None and bgr.ndim == 3) else gray
    src = erase_table_rules(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
    bgr = src if src.ndim == 3 else None
    mid = (gray > 60) & (gray < 200)
    if bgr is not None and bgr.ndim == 3:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mid |= (hsv[..., 1] > 60) & (hsv[..., 2] > 50)
    m = mid.astype(np.uint8) * 255
    for (x, y, w, h) in text_boxes:
        m[max(0, int(y) - 2):int(y + h) + 3, max(0, int(x) - 2):int(x + w) + 3] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = max(9, int(0.025 * min(W, H)))
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(closed, 8)
    min_side = max(3.0 * hmed, 0.06 * min(W, H))
    out = []
    for i in range(1, n):
        x, y, w, h, _a = [int(v) for v in st[i]]
        if w < min_side or h < min_side or w > 0.98 * W or h > 0.95 * H:
            continue
        if np.count_nonzero(m[y:y + h, x:x + w]) < 0.2 * w * h:
            continue
        out.append((x, y, w, h))
    return out


def _graphic_label(box, W, H, text_boxes=()):
    x, y, w, h = box
    compact = 0.5 <= w / float(h) <= 2.0 and max(w, h) <= 0.35 * min(W, H)
    if not compact:
        return "image"
    over_text = any(_inter_xywh(box, tb) > 0.1 * _area_xywh(tb) for tb in text_boxes)
    return "stamp" if over_text else "logo"


def _iou_xywh(a, b):
    inter = _inter_xywh(a, b)
    union = _area_xywh(a) + _area_xywh(b) - inter
    return inter / float(union) if union else 0.0


def assign_reading_order(dets):
    ordered = reading_order(dets)
    n = 0
    for d in ordered:
        if canonical_label(d.get("label")) in NON_FLOW_LABELS | LINE_LABELS:
            d["order"] = None
        else:
            n += 1
            d["order"] = n
    return ordered


def _new_block(label, blines, W, H, hmed, text=None):
    pad = max(2, min(10, int(round(0.25 * _median([_fh(d) for d in blines], hmed)))))
    return {"text": text if text is not None else block_text_from_lines(label, blines),
            "conf": int(_median([d.get("conf", 100) for d in blines], 100)),
            "bbox": _union_bbox(blines, pad, W, H), "label": label,
            "order": None, "status": "auto"}


def auto_annotate_page(lines, W, H, gray=None, regions=None, find_graphics=True, image=None):
    regions = [dict(r) for r in (regions or [])]
    L = []
    for d in lines:
        t = _norm_text(d.get("text"))
        if not t:
            continue
        lab = canonical_label(d.get("label") or "text_line")
        nd = dict(d, text=t, label=lab if lab in LINE_LABELS else "text_line", order=None)
        nd["bbox"] = tuple(int(round(v)) for v in d["bbox"])
        nd.setdefault("status", "auto")
        nd.setdefault("conf", 100)
        nd.setdefault("ocr_text", t)
        L.append(nd)
    ratios = sorted(d["fh"] / float(d["bbox"][3]) for d in L if d.get("fh") and d["bbox"][3] > 0)
    k = ratios[len(ratios) // 2] if len(ratios) >= 3 else 1.0
    for d in L:
        if not d.get("fh"):
            d["fh"] = round(d["bbox"][3] * k, 1)
    hmed_all = body_line_height(L, 15)
    L = [d for d in L if d.get("status") != "auto" or not is_graphic_noise(d, hmed_all)]

    pics = []
    if find_graphics and gray is not None:
        strong = [d["bbox"] for d in L if d.get("status") != "auto" or _is_strong_text(d)]
        excl = [r["bbox"] for r in regions
                if SYNTH_BLOCK_MAP.get(canonical_label(r.get("label")), ("", {}))[0] in SYNTH_NO_LINES]
        pics = [(_graphic_label(b, W, H, strong), b)
                for b in detect_picture_regions(gray, image, strong, W, H, hmed_all)
                if not any(_iou_xywh(b, e) > 0.3 for e in excl)]

        def _inside_pic(b):
            return any(_inter_xywh(b, p) >= 0.7 * _area_xywh(b) for _l, p in pics)
        every = [d["bbox"] for d in L]
        pics += [(lab, b) for lab, b in detect_graphic_regions(gray, every, W, H, hmed_all, excl)
                 if not _inside_pic(b) and not any(_iou_xywh(b, p) > 0.3 for _l, p in pics)]
        for box in detect_color_marks(image, W, H, hmed_all):
            if _inside_pic(box):
                continue
            if not any(_iou_xywh(box, b) > 0.3 for _l, b in pics) and \
                    not any(_iou_xywh(box, e) > 0.3 for e in excl):
                pics.append((_graphic_label(box, W, H, strong), box))
        pics = [(lab, b) for lab, b in pics
                if lab == "stamp" or sum(_inter_xywh(b, sb) for sb in strong) < 0.35 * _area_xywh(b)]
        L = [d for d in L if d.get("status") != "auto" or _is_strong_text(d)
             or not any(_inter_xywh(d["bbox"], b) >= 0.5 * _area_xywh(d["bbox"]) for _l, b in pics)]

    def _rlab(r):
        return canonical_label(r.get("label"))
    hosts = [r for r in regions if _rlab(r) not in LINE_LABELS | set(SYNTH_OVERLAYS) | SYNTH_EXTRAS
             and SYNTH_BLOCK_MAP.get(_rlab(r), ("paragraph", {}))[0] not in SYNTH_NO_LINES]
    graphics = [r for r in regions if SYNTH_BLOCK_MAP.get(_rlab(r), ("", {}))[0] in SYNTH_NO_LINES]
    per, free, kept_lines = {}, [], []
    for d in L:
        inside = [r for r in hosts if _center_inside(d, r["bbox"])]
        if inside:
            r = min(inside, key=lambda r: r["bbox"][2] * r["bbox"][3])
            per.setdefault(id(r), []).append(d)
            kept_lines.append(d)
        elif any(_center_inside(d, g["bbox"]) for g in graphics):
            continue
        else:
            free.append(d)
    for r in hosts:
        bl = per.get(id(r), [])
        if r.get("status") == "auto" and bl:
            lab = _rlab(r)
            if lab == "paragraph":
                new, _r = classify_text_block(bl, hmed_all)
                lab = "paragraph" if new == "subheading" and len(bl) > 1 else new
                r["label"] = lab
            if lab == "table" and not _looks_like_html(r.get("text") or ""):
                r["text"] = table_html_from_lines(bl)
            elif lab != "table":
                r["text"] = block_text_from_lines(lab, bl)

    hmed = _median([d["bbox"][3] for d in free], hmed_all)
    tables, rest = find_tables(free, hmed)
    hmed = body_line_height(rest, hmed)
    rest = heal_split_lines(rest, hmed)
    free = [d for tl, _h in tables for d in tl] + rest
    blocks = []
    for chain in chain_lines(rest, hmed):
        lab, r = classify_text_block(chain, hmed)
        blocks.append({"lines": chain, "label": lab, "r": r})
    blocks = absorb_nested_blocks(blocks, hmed)

    heads = [b for b in blocks if b["label"] == "heading"]
    if heads and "title" not in {canonical_label(r.get("label")) for r in regions}:
        big = max(heads, key=lambda b: (b["r"], -_bb(b["lines"][0])[1]))
        if big["r"] >= AUTO_TITLE_RATIO and _bb(big["lines"][0])[1] < 0.5 * H:
            big["label"] = "title"

    have = {_rlab(r) for r in regions}

    def y0(b):
        return min(_bb(d)[1] for d in b["lines"])

    def y1(b):
        return max(_bb(d)[3] for d in b["lines"])

    def text_of(b):
        return " ".join(_norm_text(d.get("text")) for d in b["lines"])

    if not have & {"header", "letterhead"} and blocks:
        first = min(blocks, key=y0)
        top = [b for b in blocks if y1(b) <= AUTO_HEADER_ZONE * H
               and y0(b) <= y1(first) and y1(b) >= y0(first)]
        below = [b for b in blocks if b not in top] or tables
        if top and below:
            txt = " ".join(text_of(b) for b in top).lower()
            side = any(i < j and min(y1(a), y1(c)) - max(y0(a), y0(c)) > 0
                       for i, a in enumerate(top) for j, c in enumerate(top))
            if any(k in txt for k in HEADER_KEYWORDS) or side:
                merged = [d for b in top for d in b["lines"]]
                blocks = [b for b in blocks if b not in top]
                blocks.append({"lines": merged, "label": "header", "r": 1.0})
            else:
                for b in top:
                    if len(b["lines"]) == 1 and _is_page_number(text_of(b)):
                        b["label"] = "page_number"
    if not have & {"footer", "page_number"}:
        others = lambda b: [o for o in blocks if o is not b]
        cand = [b for b in blocks if len(b["lines"]) <= 3 and (
            y0(b) >= AUTO_FOOTER_ZONE * H
            or (y0(b) >= 0.8 * H and _is_page_number(text_of(b))))]
        cand = [b for b in cand
                if all(y1(o) <= y0(b) + 2 or o in cand for o in others(b))]
        tables_below = any(max(_bb(d)[3] for d in tl) > min(y0(b) for b in cand) for tl, _h in tables) \
            if cand else False
        if cand and not tables_below and (any(_is_page_number(text_of(b)) for b in cand)
                                          or min(y0(b) for b in cand) >= 0.88 * H):
            merged = [d for b in cand for d in b["lines"]]
            blocks = [b for b in blocks if b not in cand]
            blocks.append({"lines": merged, "label": "footer", "r": 1.0})

    out = []
    for r in regions:
        out.append(r)
    for tl, html in tables:
        out.append(_new_block("table", tl, W, H, hmed, text=html))
    new_blocks = []
    for b in blocks:
        nb = _new_block(b["label"], b["lines"], W, H, hmed)
        new_blocks.append(nb)
        out.append(nb)
    summary = Counter(canonical_label(d["label"]) for d in out)

    if pics:
        tbl = [d["bbox"] for d in out if canonical_label(d["label"]) == "table"]
        found = [(lab, b) for lab, b in pics
                 if not any(_inter_xywh(b, t) > 0.5 * _area_xywh(b) for t in tbl)]
        for lab, box in found:
            out.append({"text": "", "conf": 100, "bbox": tuple(box), "label": lab,
                        "order": None, "status": "auto"})
            summary[lab] += 1
            if lab == "logo":
                for hb in new_blocks:
                    if hb["label"] == "header":
                        hx0, hy0, hx1, hy1 = _bb(hb)
                        gx0, gy0, gx1, gy1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
                        if min(hy1, gy1) - max(hy0, gy0) > 0:
                            nx0, ny0 = min(hx0, gx0), min(hy0, gy0)
                            hb["bbox"] = (nx0, ny0, max(hx1, gx1) - nx0, max(hy1, gy1) - ny0)

    out.extend(kept_lines)
    out.extend(free)
    out, _n = merge_list_detections(out)
    summary = Counter(canonical_label(d["label"]) for d in out
                      if canonical_label(d["label"]) not in LINE_LABELS)
    promote_page_title(out, H)
    out = assign_reading_order(out)
    return out, summary


def layout_regions_from_structure(dets):
    out = []
    for d in dets:
        lab = canonical_label(d.get("label"))
        if lab in LINE_LABELS:
            continue
        nd = dict(d, label=lab, order=None, status="auto")
        nd["bbox"] = tuple(int(round(v)) for v in d["bbox"])
        if lab in EMPTY_CONTENT_LABELS:
            nd["text"] = ""
        out.append(nd)
    return out


def parse_annotation_items(items):
    blocks, kinds = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("res"), dict):
            item = item["res"]

        if isinstance(item.get("layout"), list) and isinstance(item.get("lines"), list):
            kinds.add("synthetic_v2")
            blocks.extend(synth_record_to_blocks(item))
            continue

        if "parsing_res_list" in item or "layout_det_res" in item:
            kinds.add("paddle")
            page = _num(item.get("page_index"))
            meta = {
                "page": int(page) if page is not None else 0,
                "image": item.get("input_path"),
                "width": _num(item.get("width")),
                "height": _num(item.get("height")),
                "zoom": _num(item.get("render_zoom")),
            }
            layout = item.get("layout_det_res") or {}
            score_by = {}
            for lb in (layout.get("boxes") or []):
                c = lb.get("coordinate") or []
                if len(c) == 4:
                    s = _num(lb.get("score"))
                    score_by[tuple(int(v) for v in c)] = s if s is not None else 1.0
            plist = item.get("parsing_res_list") or []
            if not plist:
                plist = [{"block_label": b.get("label", "text"), "block_content": "",
                          "block_bbox": b.get("coordinate"), "block_order": b.get("order")}
                         for b in (layout.get("boxes") or [])]
            for b in plist:
                if not isinstance(b, dict):
                    continue
                xyxy = _xyxy_from_record(b)
                if not xyxy:
                    continue
                score = score_by.get(tuple(int(v) for v in xyxy), 1.0)
                blocks.append(dict(meta, det=_make_det(b, xyxy, score * 100)))

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


def split_pages_dir(pdf_path):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(pdf_path)), stem + "_pages")


def split_page_stem(pdf_path, page):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return f"{stem}_page{page + 1:03d}"


SPLIT_MANIFEST = "manifest.json"

_SIDECAR_SUFFIXES = ("_annotations", "_paddle", "_simple", "_detections", "_train")


def find_sidecar_annotations(src_path):
    d = os.path.dirname(os.path.abspath(src_path))
    stem = os.path.splitext(os.path.basename(src_path))[0]
    dirs = [d, os.path.join(d, "annotations")]
    if os.path.basename(d).lower() == "images":
        dirs.append(os.path.dirname(d))
    preferred = [stem + s for s in (
        "_annotations.jsonl", "_annotations.json", "_paddle.jsonl",
        "_paddle.json", ".jsonl", ".json", "_simple.json")]
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
                })
        full_text = pytesseract.image_to_string(img, lang=lang_str)
        return detections, full_text

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
            detections.append({
                "text": text, "conf": int(conf * 100),
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "label": "text_line", "order": i + 1,
            })
        full_text = "\n".join(d["text"] for d in detections)
        return detections, full_text

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
                })
        full_text = "\n".join(d["text"] for d in detections)
        return detections, full_text

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
                detections.append({
                    "text": text.strip(), "conf": conf,
                    "bbox": (x0, y0, x1 - x0, y1 - y0),
                    "label": "text", "order": order,
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
                })
        full_text = "\n".join(d["text"] for d in detections)
        return detections, full_text

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

    @staticmethod
    def _group_into_lines(dets):
        if not dets:
            return []
        heights = sorted(d["bbox"][3] for d in dets)
        tol = max(4, int(heights[len(heights) // 2] * 0.5))
        dets = sorted(dets, key=lambda d: d["bbox"][1] + d["bbox"][3] / 2)
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
            out.append(" ".join(d["text"] for d in items if d["text"]))
        return out

    @staticmethod
    def _prep_box_crop(crop_bgr, mode):
        h = crop_bgr.shape[0]
        target = BOX_MIN_LINE_PX if mode != "block" else BOX_MIN_LINE_PX // 2
        if 0 < h < target:
            f = min(4.0, target / float(h))
            crop_bgr = cv2.resize(crop_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        pad = max(8, crop_bgr.shape[0] // 4)
        return _pad_white(crop_bgr, pad)

    def _get_paddle_rec(self, lang_str):
        key = "rec_nep" if wants_nepali(lang_str) else "rec_en"
        if key not in self._paddle:
            from paddleocr import TextRecognition
            names = ([DEVANAGARI_REC_MODEL] if key == "rec_nep"
                     else ["en_PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"])
            last = None
            for n in names:
                try:
                    self._paddle[key] = TextRecognition(model_name=n)
                    break
                except Exception as e:
                    last = e
            if key not in self._paddle:
                raise RuntimeError(f"Could not load PaddleOCR TextRecognition: {last}")
        return self._paddle[key]

    @staticmethod
    def _res_get(res, k):
        try:
            v = res.get(k)
        except AttributeError:
            v = None
        if v is None:
            try:
                v = (res.json or {}).get("res", {}).get(k)
            except Exception:
                v = None
        return v

    def read_region(self, crop_bgr, engine, lang_str, mode="line"):
        crop = self._prep_box_crop(crop_bgr, mode)
        text, conf = "", None

        if engine == "Tesseract":
            psm = TESS_PSM.get(mode, 7)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
            text = pytesseract.image_to_string(gray, lang=lang_str, config=cfg)
            data = pytesseract.image_to_data(gray, lang=lang_str, config=cfg,
                                             output_type=Output.DICT)
            cs = [float(c) for c, t in zip(data["conf"], data["text"])
                  if str(c) not in ("-1", "") and float(c) >= 0 and t.strip()]
            conf = int(sum(cs) / len(cs)) if cs else None

        elif engine == "PaddleOCR (rec only)" and mode != "block":
            try:
                model = self._get_paddle_rec(lang_str)
                out = list(model.predict(input=crop))
                parts, scores = [], []
                for r in out:
                    t = self._res_get(r, "rec_text")
                    sc = self._res_get(r, "rec_score")
                    if t:
                        parts.append(str(t))
                    if sc is not None:
                        scores.append(float(sc))
                text = " ".join(parts)
                conf = int(100 * sum(scores) / len(scores)) if scores else None
            except ImportError:
                return self.read_region(crop_bgr, "PaddleOCR (det+rec)", lang_str, mode)

        elif engine in ("PaddleOCR (det+rec)", "PaddleOCR (rec only)"):
            ocr = self._get_paddle(lang_str)
            if hasattr(ocr, "predict"):
                dets = self._parse_paddle_results(ocr.predict(crop), 0)
            else:
                dets, _ = self.run_paddle(crop, lang_str, 0)
            lines = self._group_into_lines(dets)
            text = ("\n" if mode == "block" else " ").join(lines)
            conf = int(sum(d["conf"] for d in dets) / len(dets)) if dets else None

        elif engine == "EasyOCR":
            reader = self._get_easyocr(LANG_HINTS.get(lang_str, ["en"]))
            dets = []
            for box, t, c in reader.readtext(crop):
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                dets.append({"text": t, "conf": int(c * 100),
                             "bbox": (int(min(xs)), int(min(ys)),
                                      int(max(xs) - min(xs)), int(max(ys) - min(ys)))})
            lines = self._group_into_lines(dets)
            text = ("\n" if mode == "block" else " ").join(lines)
            conf = int(sum(d["conf"] for d in dets) / len(dets)) if dets else None

        elif engine == "Surya":
            kind = self._get_surya()
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            preds = None
            if kind[0] == "new" and mode != "block":
                _, rec, _det = kind
                try:
                    params = set(inspect.signature(rec.__call__).parameters)
                except (TypeError, ValueError):
                    params = set()
                if "bboxes" in params:
                    w, h = pil.size
                    extra = ({"langs": [LANG_HINTS.get(lang_str, ["en"])]}
                             if "langs" in params else {})
                    preds = rec([pil], bboxes=[[[0, 0, w, h]]], **extra)
            if preds is not None:
                tl = [l for p in preds for l in (getattr(p, "text_lines", None) or [])]
                text = " ".join((getattr(l, "text", "") or "").strip() for l in tl)
                cs = [getattr(l, "confidence", None) for l in tl]
                cs = [c for c in cs if c is not None]
                conf = int(100 * sum(cs) / len(cs)) if cs else None
            else:
                dets, _ = self.run_surya(crop, lang_str, 0)
                lines = self._group_into_lines(dets)
                text = ("\n" if mode == "block" else " ").join(lines)
                conf = int(sum(d["conf"] for d in dets) / len(dets)) if dets else None
        else:
            raise ValueError(f"Unknown box OCR engine: {engine}")

        text = (text or "").strip()
        if mode == "block":
            text = "\n".join(re.sub(r"[ \t]+", " ", ln).strip()
                             for ln in text.splitlines() if ln.strip())
        else:
            text = re.sub(r"\s+", " ", text)
        if conf is not None:
            conf = max(0, min(100, conf))
        return normalize_text(self._maybe_fix(text, lang_str)), conf

    _PN_WHITELIST = "0123456789\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f-/()."

    def _read_page_number(self, crop, engine, lang_str):
        tries = []
        try:
            tries.append(self.read_region(crop, "Tesseract", lang_str, "line")[0])
        except Exception:
            pass
        if engine != "Tesseract":
            try:
                tries.append(self.read_region(crop, engine, lang_str, "line")[0])
            except Exception:
                pass
        try:
            c = self._prep_box_crop(crop, "line")
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            cfg = f"--oem 1 --psm 7 -c tessedit_char_whitelist={self._PN_WHITELIST}"
            tries.append(pytesseract.image_to_string(g, lang=lang_str, config=cfg))
        except Exception:
            pass
        for t in tries:
            t = re.sub(r"\s+", " ", normalize_text(t or "")).strip()
            t = re.sub(r"^(पृष्ठ|पेज|page|pg\.?)(?=[0-9\u0966-\u096f])", r"\1 ", t, flags=re.I)
            if t and _is_page_number(t):
                return t
        return None

    def find_page_numbers(self, img, engine, lang_str, lines):
        H, W = img.shape[:2]
        gray = erase_table_rules(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        _t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hs = sorted(int(l["bbox"][3]) for l in lines) or [20]
        bodyh = max(8, hs[len(hs) // 2])
        top, bot = int(0.12 * H), int(0.86 * H)
        m = bw.copy()
        m[top:bot, :] = 0
        short = []
        for l in lines:
            x, y, w, h = [int(v) for v in l["bbox"]]
            in_margin = y + h <= top or y >= bot
            if in_margin and len(_norm_text(l.get("text"))) <= 10 and not _is_page_number(l.get("text")):
                short.append(l)
            else:
                m[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = 0
        alltext = " ".join(_norm_text(l.get("text")) for l in lines)
        n_dev = len(re.findall("[\u0966-\u096f]", alltext))
        n_lat = len(re.findall("[0-9]", alltext))
        to_dev = n_dev >= 3 and n_dev > 3 * n_lat
        k = max(3, int(0.9 * bodyh))
        dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
        n, _lab, st, _c = cv2.connectedComponentsWithStats(dil, 8)
        found, replaced = [], []
        for i in range(1, n):
            x, y, w, h, _a = [int(v) for v in st[i]]
            if not (0.3 * bodyh <= h <= 3.0 * bodyh) or w > 0.3 * W or w < 3:
                continue
            ink = np.nonzero(m[y:y + h, x:x + w])
            if ink[0].size < 6:
                continue
            y0, y1 = y + int(ink[0].min()), y + int(ink[0].max()) + 1
            x0, x1 = x + int(ink[1].min()), x + int(ink[1].max()) + 1
            pad = max(4, (y1 - y0) // 2)
            crop = img[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
            t = self._read_page_number(crop, engine, lang_str)
            if not t:
                continue
            if to_dev:
                t = t.translate(str.maketrans("0123456789", "\u0966\u0967\u0968\u0969\u096a"
                                                          "\u096b\u096c\u096d\u096e\u096f"))
            box = (x0, y0, x1 - x0, y1 - y0)
            for l in short:
                if _inter_xywh(l["bbox"], box) > 0.3 * min(_area_xywh(l["bbox"]), _area_xywh(box)):
                    replaced.append(l)
            found.append({"text": t, "conf": 80, "bbox": box, "label": "text_line",
                          "status": "auto", "line_src": "page", "fh": float(y1 - y0),
                          "ocr_text": t, "page_number": True})
        return found, replaced

    def _words_for_lines(self, img, engine, lang_str, sparse, psm=None):
        if engine == "Tesseract":
            cfg = f"--oem 1 --psm {psm or (11 if sparse else 6)}"
            data = pytesseract.image_to_data(img, lang=lang_str, config=cfg, output_type=Output.DICT)
            words = []
            for i, t in enumerate(data["text"]):
                t = (t or "").strip()
                try:
                    c = float(data["conf"][i])
                except (TypeError, ValueError):
                    c = -1
                if not t or c < 0:
                    continue
                words.append({"text": t, "conf": int(c),
                              "bbox": (data["left"][i], data["top"][i],
                                       data["width"][i], data["height"][i])})
            return words
        if engine.startswith("PaddleOCR"):
            ocr = self._get_paddle(lang_str)
            if hasattr(ocr, "predict"):
                dets = self._parse_paddle_results(ocr.predict(img), 0)
            else:
                dets, _ = self.run_paddle(img, lang_str, 0)
        elif engine == "EasyOCR":
            reader = self._get_easyocr(LANG_HINTS.get(lang_str, ["en"]))
            dets = []
            for box, t, c in reader.readtext(img):
                xs = [q[0] for q in box]
                ys = [q[1] for q in box]
                dets.append({"text": t, "conf": int(c * 100),
                             "bbox": (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))})
        elif engine == "Surya":
            dets, _ = self.run_surya(img, lang_str, 0)
        else:
            raise ValueError(f"Unknown engine: {engine}")
        return [{"text": d["text"], "conf": int(d.get("conf", 100)), "bbox": tuple(d["bbox"])}
                for d in dets if (d.get("text") or "").strip()]

    def detect_lines(self, crop_bgr, engine, lang_str, block_type="paragraph"):
        h0, w0 = crop_bgr.shape[:2]
        f = min(3.0, 64.0 / h0) if 0 < h0 < 64 else 1.0
        img = cv2.resize(crop_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1 else crop_bgr
        if block_type in ("page", "table"):
            img = erase_table_rules(img)
        pad = 10
        img = _pad_white(img, pad)
        words = []
        for wd in self._words_for_lines(img, engine, lang_str, sparse=(block_type == "table"),
                                        psm=3 if block_type == "page" else None):
            x, y, w, h = [float(v) for v in wd["bbox"]]
            x0, y0 = max(0.0, (x - pad) / f), max(0.0, (y - pad) / f)
            x1, y1 = min(float(w0), (x + w - pad) / f), min(float(h0), (y + h - pad) / f)
            text = normalize_text(self._maybe_fix(wd["text"], lang_str)).strip()
            if x1 - x0 < 1 or y1 - y0 < 1 or not text or is_junk_token(text, wd.get("conf", 100)):
                continue
            words.append({"text": text, "conf": wd["conf"], "bbox": (x0, y0, x1 - x0, y1 - y0)})
        if not words:
            return []
        gap = {"table": 0.9, "form_field": 1e9, "page": 1.2}.get(block_type, 3.0)
        lines = group_words_into_lines(words, gap)
        if block_type == "form_field":
            lines = split_form_rows(lines)
        for ln in lines:
            ln.pop("words", None)
        return lines

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

        self._history = {}
        self._dirty = False
        self._annotation_path = None
        self._annotation_kind = None
        self._split_meta = None
        self._last_autofix = 0
        self._last_new_label = DEFAULT_NEW_BOX_LABEL
        self._box_ocr_raw = None
        self._box_ocr_job = 0
        self._synth_meta = None
        self._lines_busy = False
        self.page_lang = {}
        self._last_page_lang = DEFAULT_PAGE_LANG
        self._auto_busy = False

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
        export_menu.add_command(
            label="Synthetic v2 JSON  (training format)  ★",
            command=self.export_synth_v2)
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
        import_menu.add_command(label="Annotations / PaddleOCR-VL (JSON or JSONL)...",
                                command=self.import_annotations)
        import_menu.add_command(label="Reload sidecar annotations",
                                command=self.reload_sidecar)

        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=3)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(toolbar, text="Min Conf:").pack(side=tk.LEFT)
        self.conf_var = tk.IntVar(value=60)
        ttk.Spinbox(toolbar, from_=0, to=100, textvariable=self.conf_var,
                    width=5).pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Lang:").pack(side=tk.LEFT, padx=(8, 2))
        self.lang_var = tk.StringVar(value="eng+nep")
        ttk.Combobox(toolbar, textvariable=self.lang_var, width=8, state="readonly",
                     values=["eng", "nep", "eng+nep", "nep+eng", "fra", "deu", "spa", "ita", "por",
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

        auto = ttk.Frame(self.root, padding=(6, 3))
        auto.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(auto, text="Page language:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.page_lang_var = tk.StringVar(value=DEFAULT_PAGE_LANG)
        for code, (txt, _l) in PAGE_LANGS.items():
            ttk.Radiobutton(auto, text=f"  {txt}  ", value=code, variable=self.page_lang_var,
                            style="Toolbutton", command=self._on_page_lang).pack(side=tk.LEFT, padx=2)
        self.auto_btn = ttk.Button(auto, text="\u26a1 Auto-annotate page (F7)",
                                   command=self.auto_annotate_current)
        self.auto_btn.pack(side=tk.LEFT, padx=(10, 4))
        self.auto_on_lang_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto, text="Run when language is picked",
                        variable=self.auto_on_lang_var).pack(side=tk.LEFT, padx=4)
        self.auto_layout_var = tk.BooleanVar(value=PADDLE_AVAILABLE)
        ttk.Checkbutton(auto, text="Layout model (PPStructureV3)", variable=self.auto_layout_var,
                        state=tk.NORMAL if PADDLE_AVAILABLE else tk.DISABLED
                        ).pack(side=tk.LEFT, padx=4)
        self.auto_gfx_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto, text="Find logos / stamps / figures",
                        variable=self.auto_gfx_var).pack(side=tk.LEFT, padx=4)

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
        ttk.Button(prep, text="Auto Order",
                   command=self.auto_order).pack(side=tk.RIGHT, padx=4)

        hint = ttk.Frame(self.root, padding=(6, 0))
        hint.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(hint, foreground="#666",
                  text=("Pick the page language -> blocks are found automatically (F7 re-runs) · "
                        "click a box + 1-9 / 0 or a Quick label button to fix a guess · "
                        "Drag: new box · Ctrl+Z / Ctrl+Y: undo / redo · Ctrl+S: save · "
                        "Del: delete box · Esc: deselect")
                  ).pack(side=tk.LEFT)

        exp = ttk.Frame(self.root, padding=(6, 2))
        exp.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(exp, text="Export:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))

        self.export_type_var = tk.StringVar(value=DEFAULT_EXPORT_TYPE)
        self.export_type_combo = ttk.Combobox(
            exp, textvariable=self.export_type_var, width=32, state="readonly",
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

        quick = ttk.Frame(left, padding=(4, 2, 4, 0))
        quick.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(quick, text="Quick label:", font=("Segoe UI", 9, "bold")
                  ).grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 6))
        per_row = (len(QUICK_LABELS) + 1) // 2
        for i, (lab, txt, key) in enumerate(QUICK_LABELS):
            b, g, r = label_color(lab)
            r, g, b = [int(c + (255 - c) * 0.55) for c in (r, g, b)]
            colr = f"#{r:02x}{g:02x}{b:02x}"
            tk.Button(quick, text=f"{txt}  {key}" if key else txt, bg=colr, fg="#000",
                      activebackground=colr, relief=tk.GROOVE, bd=1, padx=4, pady=0,
                      command=lambda l=lab: self.apply_quick_label(l)
                      ).grid(row=i // per_row, column=1 + i % per_row, sticky="we", padx=1, pady=1)
        tk.Button(quick, text="Merge \u2193  M", relief=tk.GROOVE, bd=1, padx=4, pady=0,
                  command=self.merge_selected_block
                  ).grid(row=0, column=per_row + 1, rowspan=2, sticky="nswe", padx=(6, 1), pady=1)
        for c in range(1, per_row + 1):
            quick.columnconfigure(c, weight=1)

        ttk.Label(sel_frame, text="Label:").pack(side=tk.LEFT)
        self.sel_label_var = tk.StringVar(value=DEFAULT_LABEL)
        self.sel_label_combo = ttk.Combobox(
            sel_frame, textvariable=self.sel_label_var, width=24, height=20,
            state="readonly", values=LABEL_MENU,
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

        box_row = ttk.Frame(left, padding=(4, 2))
        box_row.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(box_row, text="Box OCR:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.box_engine_var = tk.StringVar(value=DEFAULT_BOX_ENGINE)
        ttk.Combobox(box_row, textvariable=self.box_engine_var, width=20,
                     state="readonly", values=BOX_OCR_ENGINES).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Label(box_row, text="as").pack(side=tk.LEFT)
        self.box_mode_var = tk.StringVar(value="line")
        ttk.Combobox(box_row, textvariable=self.box_mode_var, width=6,
                     state="readonly", values=BOX_OCR_MODES).pack(side=tk.LEFT, padx=(4, 6))
        self.box_ocr_btn = ttk.Button(box_row, text="▶ OCR this box (F5)",
                                      command=self.ocr_selected_box)
        self.box_ocr_btn.pack(side=tk.LEFT, padx=2)
        self.box_auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box_row, text="Auto-OCR new boxes",
                        variable=self.box_auto_var).pack(side=tk.LEFT, padx=6)
        self.box_ocr_lbl = ttk.Label(box_row, text="", foreground="#666")
        self.box_ocr_lbl.pack(side=tk.LEFT, padx=6)

        syn_row = ttk.Frame(left, padding=(4, 2))
        syn_row.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(syn_row, text="Lines:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.lines_block_btn = ttk.Button(syn_row, text="\u2261 Lines in block (F6)",
                                          command=self.lines_in_block)
        self.lines_block_btn.pack(side=tk.LEFT, padx=(4, 2))
        self.lines_all_btn = ttk.Button(syn_row, text="\u2261 Lines in all blocks",
                                        command=self.lines_in_all_blocks)
        self.lines_all_btn.pack(side=tk.LEFT, padx=2)
        self.synth_tight_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(syn_row, text="Tight line boxes on export",
                        variable=self.synth_tight_var).pack(side=tk.LEFT, padx=6)
        self.synth_rules_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(syn_row, text="Detect table rules",
                        variable=self.synth_rules_var).pack(side=tk.LEFT, padx=2)
        self.synth_retight_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(syn_row, text="Re-tighten loaded lines",
                        variable=self.synth_retight_var).pack(side=tk.LEFT, padx=2)

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
        for lab, _txt, key in QUICK_LABELS:
            if key:
                self.canvas.bind(f"<Key-{key}>", lambda e, l=lab: (self.apply_quick_label(l), "break")[1])
        for k in ("m", "M"):
            self.canvas.bind(f"<Key-{k}>", lambda e: (self.merge_selected_block(), "break")[1])

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
                  text="Only rows with status 'reviewed' are exported to Training JSONL.",
                  foreground="#666").pack(side=tk.BOTTOM, anchor="w", pady=(4, 0))

        cols = ("label", "order", "status", "text", "conf", "x", "y", "w", "h")
        self.tree = ttk.Treeview(det_tab, columns=cols, show="headings", height=15)
        widths = {"label": 130, "order": 45, "status": 80, "text": 170, "conf": 45,
                  "x": 45, "y": 45, "w": 45, "h": 45}
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
        r.bind_all("<F5>", lambda e: (self.ocr_selected_box(), "break")[1])
        r.bind_all("<F6>", lambda e: (self.lines_in_block(), "break")[1])
        r.bind_all("<F7>", lambda e: (self.auto_annotate_current(), "break")[1])
        r.bind_all("<Control-r>", lambda e: (self.ocr_selected_box(), "break")[1])

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
        if key == "Synthetic v2 JSON (training format)":
            p = self._default_synth_path()
            if p:
                return p
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
        if key != "Synthetic v2 JSON (training format)":
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
        if self.box_auto_var.get():
            self.ocr_selected_box()

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
        return (
            d.get("label", DEFAULT_LABEL),
            "" if d.get("order") is None else d["order"],
            d.get("status", ""),
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
        self.detections[idx]["label_fixed"] = True
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
            self.sel_info_lbl.config(text=info)
        else:
            self.sel_text_var.set("")
            self.sel_label_var.set(self._last_new_label)
            self.sel_order_var.set(str(len(self.detections) + 1))
            self.sel_save_btn.config(state=tk.NORMAL, text="Save as New")
            self.sel_review_btn.config(state=tk.DISABLED)
            self.sel_delete_btn.config(state=tk.DISABLED)
            self.sel_info_lbl.config(
                text=f"New box {w}x{h} @ ({x},{y}) — type text and press Enter/Save."
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
            raw = self._take_box_ocr_raw()
            if raw is not None:
                d["ocr_text"] = raw
            if text != d["text"] or raw is not None:
                d.setdefault("ocr_text", d["text"])
                d["status"] = "reviewed"
            if label != canonical_label(d.get("label")) or new_bbox != tuple(d["bbox"]):
                d["label_fixed"] = True
            text_changed = text != d["text"]
            d["text"] = text
            d["bbox"] = new_bbox
            d["label"] = label
            d["order"] = order
            self._sync_tree_row(idx)
            note = self._sync_parent_child_text(idx) if text_changed else ""
            self.status.set("Updated detection." + note
                            + (f" Auto-fixed: {'; '.join(fixes)}." if fixes else ""))
        else:
            needs_text = label in TEXT_REQUIRED_LABELS
            if needs_text and not text.strip():
                messagebox.showinfo("No text",
                                    f"'{label}' boxes need a transcription. Type the text, "
                                    "or pick a label that allows empty text "
                                    "(e.g. image, photo, logo, signature, seal, table, checkbox).")
                return
            self._push_undo()
            self._last_new_label = label
            raw = self._take_box_ocr_raw()
            new_det = {
                "text": text, "conf": 100, "bbox": tuple(self.selected_bbox),
                "label": label, "order": order, "status": "reviewed", "label_fixed": True,
            }
            if raw is not None:
                new_det["ocr_text"] = raw
            self.detections.append(new_det)
            idx = len(self.detections) - 1
            self.tree.insert("", tk.END, values=self._tree_values(self.detections[idx]))
            self.selected_detection_index = idx
            self._stash_current_page()
            self.status.set(f"Added new detection ({label}).")
        self._update_selection_panel()
        self.refresh_preview()

    def _sync_parent_child_text(self, idx):
        """After a text edit: a block pushes its words down to its line boxes,
        a line pushes its change up into the block that contains it."""
        d = self.detections[idx]
        lab = canonical_label(d.get("label"))
        if lab in LINE_LABELS:
            p = self._parent_block_index(idx)
            if p is None:
                return ""
            blk = self.detections[p]
            plab = canonical_label(blk.get("label"))
            if plab == "table":
                return ""
            kids = self._child_lines(p)
            if plab == "list":
                blk["text"] = list_block_text(kids)
            else:
                blk["text"] = block_text_from_lines(plab, kids)
            self._sync_tree_row(p)
            return f" Block #{p} text rebuilt from its lines."
        if lab == "table" or not self._is_text_block(d):
            return ""
        kids = self._child_lines(idx)
        if not kids:
            return ""
        src = strip_heading_prefix(d["text"]) if lab in HEADING_LEVELS else d["text"]
        new, ratio = distribute_text_to_lines(src, kids)
        if new is None:
            return (f" Could not match the new text to the {len(kids)} line box(es) "
                    f"(match {ratio:.0%}) - lines left unchanged.")
        changed = 0
        for k, t in zip(kids, new):
            if t and t != k.get("text"):
                k.setdefault("ocr_text", k.get("text"))
                k["text"] = t
                k["status"] = "reviewed"
                changed += 1
        for j, o in enumerate(self.detections):
            if any(o is k for k in kids):
                self._sync_tree_row(j)
        return f" Pushed to {changed} of {len(kids)} line box(es) (match {ratio:.0%})."

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
        todo = [d for d in self.detections if d.get("status") != "reviewed"]
        if not todo:
            self.status.set("All detections on this page are already reviewed.")
            return
        self._push_undo()
        for d in todo:
            d["status"] = "reviewed"
        keep = self.selected_detection_index
        self._rebuild_tree()
        if keep is not None:
            self._select_detection(keep)
        self._update_selection_panel()
        self.status.set(f"Marked {len(todo)} detection(s) reviewed on this page.")

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
    # PAGE-LEVEL TOOLS
    # ==================================================================
    def auto_order(self):
        if not self.detections:
            return
        self._push_undo()
        self._set_detections(assign_reading_order(self.detections))
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
        rw, rh = self._raw_page_size(self.current_page)
        cs = self._coord_scale or 1.0
        warns = [w for w in qa_page_warnings(self.detections, rw * cs, rh * cs) if not w.split(": ", 1)[-1].startswith("table:")]

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
    # BOX OCR
    # ==================================================================
    def ocr_selected_box(self):
        region = self._current_region()
        if region is None:
            self.status.set("Box OCR: draw a box or select one first.")
            return
        label = self.sel_label_var.get()
        if label in EMPTY_CONTENT_LABELS:
            self.status.set(f"Box OCR skipped: '{label}' boxes have no text.")
            return
        if label == "table":
            self._run_table_on_selection()
            return
        crop = self._crop(*region)
        engine = self.box_engine_var.get()
        mode = self.box_mode_var.get()
        lang = self.lang_var.get()
        self.backends.apply_fixes = self.apply_fixes_var.get()
        self._box_ocr_job += 1
        job = self._box_ocr_job
        bbox = tuple(region)
        idx = self.selected_detection_index
        self.box_ocr_lbl.config(text=f"{engine}: reading...", foreground="#666")
        self.status.set(f"Box OCR with {engine} ({mode})... "
                        "first use of an engine loads its model.")

        def worker():
            t0 = time.time()
            try:
                text, conf = self.backends.read_region(crop, engine, lang, mode)
                el = time.time() - t0
                self.root.after(0, lambda: self._box_ocr_done(job, bbox, idx, text, conf, el, engine))
            except pytesseract.TesseractNotFoundError:
                self.root.after(0, lambda: self._box_ocr_fail(
                    job, "Tesseract not installed (sudo apt install tesseract-ocr tesseract-ocr-nep)"))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._box_ocr_fail(job, f"{engine} failed: {err}"))

        threading.Thread(target=worker, daemon=True).start()

    def _box_ocr_done(self, job, bbox, idx, text, conf, elapsed, engine):
        if job != self._box_ocr_job:
            return
        if (self.selected_bbox is None or tuple(self.selected_bbox) != bbox
                or self.selected_detection_index != idx):
            self.box_ocr_lbl.config(text="(selection changed — result dropped)")
            return
        ctxt = "?" if conf is None else f"{conf}%"
        color = "#a33" if (conf is not None and conf < 60) else "#2a7"
        self.box_ocr_lbl.config(text=f"{engine}: conf {ctxt} · {elapsed:.1f}s", foreground=color)
        if not text:
            self.status.set("Box OCR found no text — type it manually, or try another engine/mode.")
            return
        self._box_ocr_raw = (bbox, text)
        self.sel_text_var.set(text)
        self.sel_entry.focus_set()
        self.sel_entry.icursor(tk.END)
        self.sel_entry.xview_moveto(0)
        self.status.set("Box OCR done — correct the text, then press Enter to save "
                        "(F5 re-runs with the selected engine).")

    def _box_ocr_fail(self, job, msg):
        if job != self._box_ocr_job:
            return
        self.box_ocr_lbl.config(text="error", foreground="#a33")
        self.status.set(msg)
        print(f"[ocr-tool] {msg}")

    def _take_box_ocr_raw(self):
        raw, self._box_ocr_raw = self._box_ocr_raw, None
        if raw and self.selected_bbox is not None and tuple(self.selected_bbox) == raw[0]:
            return raw[1]
        return None

    # ==================================================================
    # LINES
    # ==================================================================
    def _is_text_block(self, d):
        lab = canonical_label(d.get("label"))
        if lab in LINE_LABELS or lab in SYNTH_OVERLAYS or lab in SYNTH_EXTRAS:
            return False
        return SYNTH_BLOCK_MAP.get(lab, ("paragraph", {}))[0] not in SYNTH_NO_LINES

    @staticmethod
    def _center_in(o, bbox):
        x, y, w, h = bbox
        ox, oy, ow, oh = o["bbox"]
        cx, cy = ox + ow / 2.0, oy + oh / 2.0
        return x <= cx <= x + w and y <= cy <= y + h

    def _block_has_lines(self, d):
        return any(canonical_label(o.get("label")) in LINE_LABELS and self._center_in(o, d["bbox"])
                   for o in self.detections)

    def lines_in_block(self):
        idx = self.selected_detection_index
        if idx is None or idx >= len(self.detections):
            self.status.set("Lines: select a block first (paragraph, table, form, heading...).")
            return
        d = self.detections[idx]
        if not self._is_text_block(d):
            self.status.set(f"Lines: '{canonical_label(d.get('label'))}' is not a text block "
                            "- select the block box, not a line or a figure.")
            return
        self._run_line_detection([idx])

    def lines_in_all_blocks(self):
        idxs = [i for i, d in enumerate(self.detections)
                if self._is_text_block(d) and not self._block_has_lines(d)]
        if not idxs:
            self.status.set("Lines: every text block on this page already has line boxes.")
            return
        self._run_line_detection(idxs)

    def _set_lines_buttons(self, enabled):
        st = tk.NORMAL if enabled else tk.DISABLED
        for b in (self.lines_block_btn, self.lines_all_btn):
            b.config(state=st)

    def _run_line_detection(self, idxs):
        if self._lines_busy or self.original_image is None:
            return
        engine = self.box_engine_var.get()
        lang = self.lang_var.get()
        self.backends.apply_fixes = self.apply_fixes_var.get()
        jobs = []
        for i in idxs:
            d = self.detections[i]
            x, y, w, h = [int(v) for v in d["bbox"]]
            if w < 4 or h < 4:
                continue
            typ = SYNTH_BLOCK_MAP.get(canonical_label(d.get("label")), ("paragraph", {}))[0]
            jobs.append(((x, y, w, h), typ, self._crop(x, y, w, h)))
        if not jobs:
            return
        page = self.current_page
        self._lines_busy = True
        self._set_lines_buttons(False)
        self.status.set(f"Finding text lines with {engine} in {len(jobs)} block(s)...")

        def worker():
            results, errors = [], []
            for n, (bbox, typ, crop) in enumerate(jobs, 1):
                try:
                    results.append((bbox, self.backends.detect_lines(crop, engine, lang, typ)))
                except pytesseract.TesseractNotFoundError:
                    errors.append("Tesseract not installed (sudo apt install tesseract-ocr tesseract-ocr-nep)")
                    break
                except Exception as e:
                    errors.append(f"{engine}: {e}")
                self.root.after(0, lambda n=n: self.status.set(
                    f"Finding text lines... block {n}/{len(jobs)}"))
            self.root.after(0, lambda: self._lines_done(page, results, errors))

        threading.Thread(target=worker, daemon=True).start()

    def _lines_done(self, page, results, errors):
        self._lines_busy = False
        self._set_lines_buttons(True)
        if page != self.current_page:
            self.status.set("Lines: page changed while detecting - result dropped, run it again.")
            return
        if not results:
            self.status.set("Lines: " + ("; ".join(errors[:2]) if errors else "nothing found."))
            return
        self._push_undo()
        dets = list(self.detections)
        added = removed = 0
        for bbox, lines in results:
            bx, by = bbox[0], bbox[1]
            keep = [o for o in dets if not (canonical_label(o.get("label")) in LINE_LABELS
                                            and self._center_in(o, bbox))]
            removed += len(dets) - len(keep)
            dets = keep
            new = []
            for ln in lines:
                lx, ly, lw, lh = ln["bbox"]
                new.append({"text": ln["text"], "conf": ln["conf"],
                            "bbox": (int(round(bx + lx)), int(round(by + ly)),
                                     max(1, int(round(lw))), max(1, int(round(lh)))),
                            "label": ln["label"], "order": None, "status": "auto",
                            "line_src": "block", "ocr_text": ln["text"]})
            dets.extend(new)
            added += len(new)
            for o in dets:
                lab = canonical_label(o.get("label"))
                if lab in LINE_LABELS or tuple(o["bbox"]) != tuple(bbox) or not new:
                    continue
                if lab == "table":
                    if not _looks_like_html(o.get("text") or ""):
                        o["text"] = table_html_from_lines(new)
                elif not (o.get("text") or "").strip():
                    o["text"] = block_text_from_lines(lab, new)
        self._set_detections(dets)
        msg = f"Lines: added {added} line box(es) in {len(results)} block(s)"
        if removed:
            msg += f", replaced {removed} old one(s)"
        msg += ". Check each line: click it, F5 re-reads it, fix the text, Enter saves. Ctrl+Z undoes."
        if errors:
            msg += f" Errors: {errors[0]}"
        self.status.set(msg)

    # ==================================================================
    # PAGE LANGUAGE + AUTO-ANNOTATE + QUICK LABELS
    # ==================================================================
    def _sync_page_lang_ui(self):
        if not hasattr(self, "page_lang_var"):
            return
        code = self.page_lang.get(self.current_page, self._last_page_lang)
        if code not in PAGE_LANGS:
            code = DEFAULT_PAGE_LANG
        self.page_lang_var.set(code)
        if self.lang_var.get() in ("eng", "nep", "eng+nep", "nep+eng"):
            self.lang_var.set(PAGE_LANGS[code][1])

    def _on_page_lang(self):
        code = self.page_lang_var.get()
        if code not in PAGE_LANGS:
            return
        changed = self.page_lang.get(self.current_page) != code
        self.page_lang[self.current_page] = code
        self._last_page_lang = code
        self.lang_var.set(PAGE_LANGS[code][1])
        if self.original_image is None:
            return
        if changed and self.detections:
            self._set_dirty(True)
        has_blocks = any(canonical_label(d.get("label")) not in LINE_LABELS for d in self.detections)
        if self.auto_on_lang_var.get() and not has_blocks:
            self.auto_annotate_current()
        else:
            self.status.set(f"Page {self.current_page + 1}: {PAGE_LANGS[code][0]}. "
                            "F7 (Auto-annotate page) rebuilds the blocks; boxes you fixed are kept.")

    def auto_annotate_current(self):
        if self.original_image is None:
            self.status.set("Open an image or PDF first.")
            return
        if self._auto_busy:
            return
        self._autosave_pending_new()
        code = self.page_lang_var.get()
        if code not in PAGE_LANGS:
            code = DEFAULT_PAGE_LANG
        self.page_lang[self.current_page] = code
        self._last_page_lang = code
        lang = PAGE_LANGS[code][1]
        self.lang_var.set(lang)
        engine = self.box_engine_var.get()
        self.backends.apply_fixes = self.apply_fixes_var.get()

        dets = copy.deepcopy(self.detections)
        old_lines = [d for d in dets if canonical_label(d.get("label")) in LINE_LABELS]
        keep = [d for d in dets if canonical_label(d.get("label")) not in LINE_LABELS
                and d.get("label_fixed")]
        rebuilt = sum(1 for d in dets if canonical_label(d.get("label")) not in LINE_LABELS
                      and not d.get("label_fixed"))
        good = [d for d in old_lines if d.get("line_src") in ("page", "block")
                or d.get("status") != "auto"]
        reuse = good if any(d.get("line_src") == "page" for d in good) else []
        use_layout = bool(self.auto_layout_var.get()) and PADDLE_AVAILABLE and not keep
        find_gfx = bool(self.auto_gfx_var.get())
        img = self.original_image.copy()
        page = self.current_page
        self._auto_busy = True
        self.auto_btn.config(state=tk.DISABLED)
        self.status.set(f"Auto-annotating page {page + 1} ({PAGE_LANGS[code][0]}): "
                        + ("reusing line boxes" if reuse else f"reading lines with {engine}")
                        + (" + layout model" if use_layout else "")
                        + "... (first use of an engine loads its model)")

        def worker():
            notes = []
            try:
                if reuse:
                    lines = reuse
                else:
                    raw = self.backends.detect_lines(img, engine, lang, "page")
                    new = [{"text": ln["text"], "conf": ln.get("conf", 100),
                            "bbox": tuple(int(round(v)) for v in ln["bbox"]),
                            "label": "text_line", "status": "auto", "line_src": "page",
                            "fh": ln.get("fh"), "ocr_text": ln["text"]} for ln in raw]
                    try:
                        pns, wrong = self.backends.find_page_numbers(img, engine, lang, new + list(good))
                        ids = {id(w_) for w_ in wrong}
                        new = [n_ for n_ in new if id(n_) not in ids] + pns
                        if pns:
                            notes.append("page number: " + ", ".join(p_["text"] for p_ in pns))
                    except Exception as e:
                        notes.append(f"page-number check skipped ({e})")
                    lines = list(good) + [
                        n for n in new
                        if not any(_inter_xywh(n["bbox"], g["bbox"])
                                   >= 0.4 * min(_area_xywh(n["bbox"]), _area_xywh(g["bbox"]))
                                   for g in good)]
                regions = list(keep)
                if use_layout:
                    try:
                        sd, _t = self.backends.run_paddle_structure(img, lang, 0)
                        regions += layout_regions_from_structure(sd)
                    except Exception as e:
                        notes.append(f"layout model skipped ({e})")
                H, W = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                out, summ = auto_annotate_page(lines, W, H, gray, regions, find_gfx, image=img)
                if rebuilt:
                    notes.append(f"{rebuilt} earlier block(s) were rebuilt (Ctrl+Z restores them)")
                self.root.after(0, lambda: self._auto_done(page, code, out, summ, notes, len(lines)))
            except pytesseract.TesseractNotFoundError:
                self.root.after(0, lambda: self._auto_fail(
                    "Tesseract not installed (sudo apt install tesseract-ocr tesseract-ocr-nep), "
                    "or pick another engine in the Box OCR row."))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                self.root.after(0, lambda: self._auto_fail(f"Auto-annotate failed: {err}"))

        threading.Thread(target=worker, daemon=True).start()

    def _auto_fail(self, msg):
        self._auto_busy = False
        self.auto_btn.config(state=tk.NORMAL)
        self.status.set(msg)
        print(f"[auto] {msg}")

    def _auto_done(self, page, code, out, summ, notes, nlines):
        self._auto_busy = False
        self.auto_btn.config(state=tk.NORMAL)
        if page != self.current_page:
            self.status.set("Auto-annotate: page changed while working - result dropped, press F7 again.")
            return
        if not out:
            self.status.set("Auto-annotate found no text on this page. Try another engine in the "
                            "Box OCR row, or check the page language.")
            return
        self._push_undo()
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", self._text_from_dets(
            [d for d in out if canonical_label(d.get("label")) not in LINE_LABELS]))
        self._set_detections(out)
        parts = ", ".join(f"{n} {lab}" for lab, n in summ.most_common())
        msg = (f"Auto-annotated ({PAGE_LANGS[code][0]}): {parts or 'no blocks'}; {nlines} line(s). "
               "Wrong guess? Click the box (or any line in it) and press 1-9 / 0 or a "
               "Quick label button. Ctrl+S saves synthetic v2.")
        if notes:
            msg += " Note: " + "; ".join(notes)
        self.status.set(msg)

    def _parent_block_index(self, idx):
        d = self.detections[idx]
        best = None
        for j, o in enumerate(self.detections):
            if j == idx:
                continue
            lab = canonical_label(o.get("label"))
            if lab in LINE_LABELS or lab in SYNTH_OVERLAYS or lab in SYNTH_EXTRAS:
                continue
            if not _center_inside(d, o["bbox"]):
                continue
            textual = SYNTH_BLOCK_MAP.get(lab, ("paragraph", {}))[0] not in SYNTH_NO_LINES
            key = (not textual, o["bbox"][2] * o["bbox"][3])
            if best is None or key < best[0]:
                best = (key, j)
        return None if best is None else best[1]

    def _child_lines(self, bidx):
        return [o for j, o in enumerate(self.detections)
                if canonical_label(o.get("label")) in LINE_LABELS
                and self._parent_block_index(j) == bidx]

    def apply_quick_label(self, label):
        if self.original_image is None:
            return
        label = canonical_label(label)
        idx = self.selected_detection_index
        if idx is None or idx >= len(self.detections):
            if self.selected_bbox is not None:
                self._quick_label_new_box(label)
            else:
                self.status.set("Click a box (or a line inside a block) first, then a quick label.")
            return
        d = self.detections[idx]
        cur = canonical_label(d.get("label"))
        target = idx
        if cur in LINE_LABELS and label not in LINE_LABELS:
            target = self._parent_block_index(idx)
        if target is not None and canonical_label(self.detections[target].get("label")) == label:
            self.status.set(f"Already '{label}'.")
            return
        self._push_undo()
        note = ""
        if target is None:
            x, y, w, h = d["bbox"]
            H, W = self.original_image.shape[:2]
            pad = max(2, int(h) // 5)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            self.detections.append({"text": "", "conf": 100, "bbox": (x0, y0, x1 - x0, y1 - y0),
                                    "label": label, "order": None, "status": "auto"})
            target = len(self.detections) - 1
            note = " (new block around the line)"
        elif target != idx:
            note = " (the block this line belongs to)"
        blk = self.detections[target]
        kids = self._child_lines(target)
        blk["label"] = label
        blk["label_fixed"] = True
        if label in EMPTY_CONTENT_LABELS:
            blk["text"] = ""
            if kids:
                ids = {id(k) for k in kids}
                self.detections = [o for o in self.detections if id(o) not in ids]
                note += f", removed {len(kids)} line box(es) inside it"
        elif label == "table":
            if not _looks_like_html(blk.get("text") or ""):
                blk["text"] = table_html_from_lines(kids) if kids else ""
                if not blk["text"]:
                    note += " - no grid found: use 'Table OCR on selection' or type the HTML"
        elif label not in LINE_LABELS:
            if kids:
                blk["text"] = block_text_from_lines(label, kids)
            elif _looks_like_html(blk.get("text") or ""):
                blk["text"] = re.sub(r"\s+", " ", _strip_html_tags(blk["text"])).strip()
            else:
                blk["text"] = strip_heading_prefix(blk.get("text") or "") \
                    if label in HEADING_LEVELS else (blk.get("text") or "")
        assign_reading_order(self.detections)
        new_idx = next(i for i, o in enumerate(self.detections) if o is blk)
        self._stash_current_page()
        self._rebuild_tree()
        self._select_detection(new_idx)
        self.status.set(f"Box #{new_idx} -> {label}{note}. Ctrl+Z undoes.")
        if (label not in LINE_LABELS and label not in EMPTY_CONTENT_LABELS
                and self._is_text_block(blk) and not self._block_has_lines(blk)):
            self._run_line_detection([new_idx])

    def merge_selected_block(self):
        idx = self.selected_detection_index
        if idx is None or idx >= len(self.detections):
            self.status.set("Merge: click a block (or a line in it) first.")
            return
        if canonical_label(self.detections[idx].get("label")) in LINE_LABELS:
            idx = self._parent_block_index(idx)
            if idx is None:
                self.status.set("Merge: this line is not in a block - give it a label first.")
                return
        blk = self.detections[idx]
        bx0, by0, bx1, by1 = _bb(blk)

        def ok(o):
            lab = canonical_label(o.get("label"))
            if o is blk or lab in LINE_LABELS or lab in SYNTH_OVERLAYS or lab in SYNTH_EXTRAS:
                return False
            if SYNTH_BLOCK_MAP.get(lab, ("paragraph", {}))[0] in SYNTH_NO_LINES:
                return False
            ox0, _oy0, ox1, _oy1 = _bb(o)
            return min(bx1, ox1) - max(bx0, ox0) > 0.2 * min(bx1 - bx0, ox1 - ox0)

        cands = [o for o in self.detections if ok(o)]
        below = [o for o in cands if _bb(o)[1] >= by0 + 2]
        above = [o for o in cands if _bb(o)[1] < by0 + 2]
        other = (min(below, key=lambda o: _bb(o)[1] - by1) if below
                 else max(above, key=lambda o: _bb(o)[3]) if above else None)
        if other is None:
            self.status.set("Merge: no block above or below that overlaps this one.")
            return
        self._push_undo()
        ox0, oy0, ox1, oy1 = _bb(other)
        nx0, ny0 = min(bx0, ox0), min(by0, oy0)
        blk["bbox"] = (nx0, ny0, max(bx1, ox1) - nx0, max(by1, oy1) - ny0)
        blk["label_fixed"] = True
        self.detections = [o for o in self.detections if o is not other]
        new_idx = next(i for i, o in enumerate(self.detections) if o is blk)
        kids = self._child_lines(new_idx)
        lab = canonical_label(blk.get("label"))
        if lab == "table":
            if kids:
                blk["text"] = table_html_from_lines(kids)
        elif kids:
            blk["text"] = block_text_from_lines(lab, kids)
        else:
            blk["text"] = " ".join(t for t in (blk.get("text"), other.get("text")) if t)
        assign_reading_order(self.detections)
        new_idx = next(i for i, o in enumerate(self.detections) if o is blk)
        self._stash_current_page()
        self._rebuild_tree()
        self._select_detection(new_idx)
        self.status.set(f"Merged into one {lab} block with {len(kids)} line(s). "
                        "Press M again to add the next block. Ctrl+Z undoes.")

    def _quick_label_new_box(self, label):
        bbox = tuple(self.selected_bbox)
        text, label, _fx = clean_block_content(self.sel_text_var.get(), label)
        self._push_undo()
        det = {"text": text, "conf": 100, "bbox": bbox, "label": label, "order": None,
               "status": "reviewed" if text.strip() else "auto", "label_fixed": True}
        raw = self._take_box_ocr_raw()
        if raw is not None:
            det["ocr_text"] = raw
        self.detections.append(det)
        self._last_new_label = label
        assign_reading_order(self.detections)
        idx = next(i for i, o in enumerate(self.detections) if o is det)
        self._stash_current_page()
        self._rebuild_tree()
        self._select_detection(idx)
        self.status.set(f"Added {label} box.")
        if (label not in LINE_LABELS and label not in EMPTY_CONTENT_LABELS
                and self._is_text_block(det) and not self._block_has_lines(det)):
            self._run_line_detection([idx])

    # ==================================================================
    # SYNTHETIC v2 EXPORT
    # ==================================================================
    def _synth_page_image(self, page, w, h):
        try:
            img = self._render_page_image(page) if self.pdf_doc is not None else self.raw_page_image
        except Exception as e:
            print(f"[synth] could not render page {page}: {e}")
            return None
        if img is None:
            return None
        if w and h and (img.shape[1], img.shape[0]) != (int(w), int(h)):
            img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        return img

    def _synth_page_id(self, page):
        if self.pdf_doc is not None:
            return split_page_stem(self.image_path, page)
        return os.path.splitext(os.path.basename(self.image_path))[0]

    def _default_synth_path(self):
        if not self.image_path:
            return None
        pid0 = self._synth_page_id(0)
        if self.pdf_doc is not None:
            return os.path.join(split_pages_dir(self.image_path), pid0 + ".json")
        return os.path.join(
            os.path.dirname(os.path.abspath(self.image_path)), pid0 + ".json")

    def export_synth_v2(self, path=None, write_images=True, quiet=False):
        if self.image_path is None or self.original_image is None:
            if not quiet:
                messagebox.showinfo("No image", "Open an image/PDF first.")
            return False
        self._autosave_pending_new()
        self._last_autofix = self._clean_all_pages()
        entries = [e for e in self._page_entries() if e[1]]
        if not entries:
            if not quiet:
                messagebox.showinfo("No data", "No boxes to export.")
            return False
        if not path:
            path = self._ask_path(None, ".json",
                                  [("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return False
        out_dir = os.path.dirname(os.path.abspath(path))
        img_dir = os.path.join(out_dir, "images")
        sm = self._synth_meta if self._annotation_kind == "synthetic_v2" else None
        split = self._split_meta or {}
        is_pdf = self.pdf_doc is not None
        written, warns_all = [], []
        try:
            for p, dets, w, h in entries:
                pid = self._synth_page_id(p)
                default = self._default_synth_path()
                per_page = len(entries) > 1 or (is_pdf and default and os.path.normcase(
                    os.path.abspath(path)) == os.path.normcase(os.path.abspath(default)))
                # a PDF page always goes to its own <name>_pageNNN.json - never into page001's file
                jpath = os.path.join(out_dir, pid + ".json") if per_page else path
                page_img = self._synth_page_image(p, w, h)
                if write_images:
                    os.makedirs(img_dir, exist_ok=True)
                    ipath = os.path.join(img_dir, pid + ".jpg")
                    src = self.image_path
                    same = (not is_pdf and page_img is not None and self.raw_page_image is not None
                            and page_img.shape == self.raw_page_image.shape)
                    if same and src.lower().endswith((".jpg", ".jpeg")):
                        if os.path.abspath(src) != os.path.abspath(ipath):
                            shutil.copyfile(src, ipath)
                    elif page_img is not None:
                        ok, buf = cv2.imencode(".jpg", page_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not ok:
                            raise RuntimeError("JPEG encoding failed")
                        buf.tofile(ipath)
                    image_ref = "images/" + pid + ".jpg"
                else:
                    image_ref = self._rel(self.image_path, out_dir)
                statuses = {d.get("status") for d in dets}
                src_stem = os.path.splitext(os.path.basename(split.get("source") or self.image_path))[0]
                # PDF: the page number is simply the page's position in the PDF. The
                # loaded file's details belong to ONE page, so never copy its number.
                page_no = (p + 1 if is_pdf else
                           ((sm or {}).get("page_number") or split.get("page") or 1))
                fname_page = None if is_pdf else re.search(
                    r"_page(\d{1,4})$", os.path.splitext(os.path.basename(self.image_path))[0])
                if page_no == 1 and fname_page:
                    page_no = int(fname_page.group(1))
                page_cnt = (self.page_count if is_pdf else
                            ((sm or {}).get("page_count") or split.get("page_count") or 1))
                page_cnt = max(int(page_cnt), int(page_no))
                doc_id = (sm or {}).get("document_id") or re.sub(r"_page\d{1,4}$", "", src_stem)
                meta = {
                    "image": image_ref,
                    "document_id": doc_id,
                    "page_number": page_no,
                    "page_count": page_cnt,
                    "status": "reviewed" if statuses == {"reviewed"} else "draft",
                    "language": self.page_lang.get(p),
                }
                rec, warns = build_synth_page(dets, w, h, meta, page_img,
                                              tighten=self.synth_tight_var.get(),
                                              rulings=self.synth_rules_var.get(),
                                              retighten=self.synth_retight_var.get())
                with open(jpath, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                written.append(jpath)
                warns_all.extend(f"{pid}: {x}" for x in warns)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write synthetic v2 JSON:\n{e}")
            return False
        report = None
        if warns_all:
            report = os.path.splitext(path)[0] + "_synth_warnings.txt"
            with open(report, "w", encoding="utf-8") as f:
                f.write("\n".join(warns_all) + "\n")
            for x in warns_all:
                print("[synth]", x)
        if quiet:
            self._mark_saved(path, "synthetic_v2")
            self.status.set(f"Saved {os.path.basename(path)} (synthetic v2)"
                            + (f" - {len(warns_all)} warning(s), see {os.path.basename(report)}"
                               if report else ""))
            return True
        msg = (f"Wrote {len(written)} page JSON(s) to {out_dir}"
               + (" and images/" if write_images else "") + ".")
        if warns_all:
            msg += (f"\n\n{len(warns_all)} warning(s) (also in {os.path.basename(report)}):\n  "
                    + "\n  ".join(warns_all[:12]) + ("\n  ..." if len(warns_all) > 12 else ""))
        self.status.set(msg.split("\n")[0])
        messagebox.showinfo("Synthetic v2 export", msg)
        return True

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
            "bbox": (x, y, w, h), "label": "table", "order": None, "status": "auto",
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
        self.current_page = 0
        self.selected_bbox = None
        self.selected_detection_index = None
        self._history = {}
        self._annotation_path = None
        self._annotation_kind = None
        self._split_meta = None
        self._synth_meta = None
        self.page_lang = {}
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

    def _after_load(self):
        self._sync_page_lang_ui()
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
    # ANNOTATION LOADING
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

        self._synth_meta = next((b.get("doc_meta") for b in sel if b.get("doc_meta")), None)
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
            lg = next(((b.get("doc_meta") or {}).get("language") for b in bl
                       if (b.get("doc_meta") or {}).get("language") in PAGE_LANGS), None)
            if lg:
                self.page_lang[page] = lg

        if self.current_page in self.page_ocr_cache:
            self._show_cached_page(self.current_page)
        self._sync_page_lang_ui()
        self._update_undo_buttons()
        return loaded, skipped

    def _kind_for(self, kinds):
        if self._split_meta:
            return "split_page"
        return "annotations" if kinds == {"annotations"} else next(iter(kinds))

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
            cand = os.path.join(jdir, str(img).replace("\\", "/"))
            if os.path.isfile(cand):
                return cand
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
        msg = f"Opened {os.path.basename(img)} with {n} box(es) from {os.path.basename(path)}."
        if skipped:
            msg += f" {skipped} page(s) out of range were skipped."
        self.status.set(msg)

    def import_annotations(self):
        path = filedialog.askopenfilename(
            title="Import annotations (annotations JSON/JSONL, PaddleOCR-VL JSON/JSONL)",
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
            lab = canonical_label(d.get("label"))
            box_color = label_color(lab)
            thick = 1 if lab in LINE_LABELS else 2
            if d.get("quad"):
                pts = np.array(det_quad(d), dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], True, box_color, thick)
            else:
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), box_color, thick)

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
            if label in LINE_LABELS:
                continue
            display = self._clean_display_text(d, max_len=60)
            cap = f"[{label}] {display}"
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
                        detections.append({
                            "text": word, "conf": 100,
                            "bbox": (int(x0 * z), int(y0 * z),
                                     int((x1 - x0) * z), int((y1 - y0) * z)),
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
        self._rebuild_tree()
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", normalize_text(full_text))
        self._drop_selection()
        self._stash_current_page()
        self._set_dirty(True)
        self.status.set(f"OCR complete ({backend_used}): "
                        f"{len(self.detections)} detections. Ctrl+Z restores the previous boxes.")

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
        self._set_dirty(True)
        self.run_all_btn.config(state=tk.NORMAL)
        self.run_btn.config(state=tk.NORMAL)
        if self.current_page in self.page_ocr_cache:
            self._show_cached_page(self.current_page)
        self._update_undo_buttons()
        self.status.set(f"OCR All Pages complete. Showing page {self.current_page + 1}.")

    # ==================================================================
    # EXPORT HELPERS
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
            self._rebuild_tree()
            self.refresh_preview()
        return changed

    def _qa_summary(self, entries):
        lines = []
        for p, dets, _w, _h in entries:
            for w in qa_page_warnings(dets, _w, _h):
                lines.append(f"page {p + 1}: {w}")
        for ln in lines:
            print(f"[qa] {ln}")
        return len(lines), lines

    def quick_save(self):
        if not self.image_path or self.original_image is None:
            return False

        self._autosave_pending_new()
        self._last_autofix = self._clean_all_pages()

        p, kind = self._annotation_path, self._annotation_kind

        if p and os.path.isfile(p) and kind == "paddle":
            if p.lower().endswith(".jsonl"):
                return bool(self.export_paddle_jsonl(path=p))
            return bool(self.export_paddle_json(path=p))

        target = None
        if p and p.lower().endswith(".json") \
                and kind in ("synthetic_v2", "split_page", None):
            target = p
        if not target:
            target = self._default_synth_path()
        if not target:
            messagebox.showerror("Save Error", "No image/PDF is open.")
            return False

        folder = os.path.dirname(target)
        if folder and not os.path.isdir(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as e:
                messagebox.showerror("Save Error",
                                     f"Could not create folder:\n{folder}\n\n{e}")
                return False

        return bool(self.export_synth_v2(path=target,
                                         write_images=True,
                                         quiet=True))

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
            polygon = [
                [float(x0), float(y0)], [float(x1), float(y0)],
                [float(x1), float(y1)], [float(x0), float(y1)],
            ]
            parsing_res_list.append({
                "block_label": label,
                "block_content": content,
                "block_bbox": [x0, y0, x1, y1],
                "block_id": i,
                "block_order": order,
                "group_id": i,
                "block_polygon_points": polygon,
                "status": d.get("status", "auto"),
            })
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
                writer.writerow(["label", "order", "status", "text", "ocr_text",
                                 "confidence", "x", "y", "width", "height"])
                for d in entries[0][1]:
                    x, y, w, h = d["bbox"]
                    writer.writerow([
                        canonical_label(d.get("label")),
                        "" if d.get("order") is None else d["order"],
                        d.get("status", ""),
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
            export_data.append({
                "text": d["text"],
                "confidence": d.get("conf", 100),
                "label": canonical_label(d.get("label")),
                "order": d.get("order"),
                "status": d.get("status"),
                "bbox": {"x": x, "y": y, "width": w, "height": h},
            })
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
            records.append(rec)
        return records

    @staticmethod
    def _blocks_payload(dets):
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
                "block_order": order,
                "confidence": d.get("conf", 100),
                "status": d.get("status", "auto"),
            }
            if label in NON_FLOW_LABELS:
                b["ignore_in_eval"] = True
            if "ocr_text" in d:
                b["ocr_text"] = d["ocr_text"]
            blocks.append(b)
        return blocks

    # ==================================================================
    # SPLIT PDF
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
        nbox = 0
        try:
            for p, dets, w, h in entries:
                rec = self._split_page_record(p, dets, w, h, d)
                path = os.path.join(d, split_page_stem(self.image_path, p) + ".json")
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                nbox += len(dets)
        except OSError as e:
            messagebox.showerror("Save Error", f"Failed to write page annotations:\n{e}")
            return False
        self._mark_saved(d, "split")
        msg = (f"Saved {nbox} box(es) on {len(entries)} page(s) to "
               f"{os.path.basename(d)}/ (one PNG + one JSON per page)")
        self.status.set(msg + self._save_suffix(entries))
        return True

    def _save_suffix(self, entries):
        parts = []
        fixed = getattr(self, "_last_autofix", 0)
        self._last_autofix = 0
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
        key = m.get("source_key", "source_pdf")
        rec = {
            "image": os.path.basename(self.image_path),
            key: m.get("source", ""),
            "page": m["page"],
            "page_count": m.get("page_count"),
            "width": w,
            "height": h,
            "render_zoom": m.get("render_zoom") or 1.0,
            "blocks": self._blocks_payload(dets),
        }
        stem = os.path.splitext(os.path.abspath(self.image_path))[0]
        path = stem + ".json"
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        except OSError as e:
            messagebox.showerror("Save Error", f"Failed to write:\n{e}")
            return False
        self._mark_saved(path, "split_page")
        src = os.path.basename(m.get("source") or "") or "source"
        self.status.set(f"Saved {len(dets)} box(es) to {os.path.basename(path)} "
                        f"(page {m['page']} of {src})" + self._save_suffix(entries))
        return True

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
    # GROUND TRUTH v2
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
        self._history = {}
        self._annotation_path = None
        self._annotation_kind = None
        self._split_meta = None
        self._coord_scale = 1.0
        self.page_lang = {}
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
