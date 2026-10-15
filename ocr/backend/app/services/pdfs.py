"""Split a PDF into one image per page, with a best-effort text-layer check."""
from __future__ import annotations

from io import BytesIO
from typing import Iterator

from pypdf import PdfReader

LEGACY_FONT_MARKERS = ("preeti", "kantipur", "sagarmatha")


def pdf_page_count(data: bytes) -> int:
    return len(PdfReader(BytesIO(data)).pages)


def is_legacy_font(text: str) -> str | None:
    low = text.lower()
    for name in LEGACY_FONT_MARKERS:
        if name in low:
            return name
    return None


def iter_pdf_pages(
    data: bytes, dpi: int, image_format: str = "JPEG",
) -> Iterator[tuple[int, bytes, str, str | None]]:
    """Yield (page_number, image_bytes, text_layer, legacy_font_name) per page.

    image_format "PNG" gives lossless pages (written with fast compression;
    the stored copy is re-compressed once).

    Uses pypdfium2 — pip-only, no system binaries. Output is JPEG: encoding
    a 300-DPI page as PNG is roughly 10× slower than JPEG and the visual
    difference for scanned text is negligible.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise RuntimeError(
            "PDF ingestion requires pypdfium2 (pip install pypdfium2)"
        ) from e

    pdf = pdfium.PdfDocument(data)
    reader = PdfReader(BytesIO(data))
    scale = dpi / 72.0

    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")

        buf = BytesIO()
        if image_format.upper() == "PNG":
            pil_image.save(buf, format="PNG", compress_level=1)
        else:
            pil_image.save(buf, format="JPEG", quality=90, optimize=False)
        jpeg_bytes = buf.getvalue()

        try:
            text_layer = reader.pages[i].extract_text() or ""
        except Exception:
            text_layer = ""

        legacy = None
        try:
            fonts = reader.pages[i].get("/Resources", {}).get("/Font", {})
            names = []
            for f in fonts.values():
                obj = f.get_object()
                bf = obj.get("/BaseFont")
                if bf is not None:
                    names.append(str(bf))
            legacy = is_legacy_font(" ".join(names))
        except Exception:
            pass

        yield i + 1, jpeg_bytes, text_layer, legacy
        page.close()
    pdf.close()