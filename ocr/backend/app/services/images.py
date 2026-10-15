"""Read, orient, validate and store a single uploaded image."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from app.storage import get_storage

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
LOSSLESS_EXTS = {".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class PreparedImage:
    image: Image.Image          # already EXIF-corrected, ready to save
    save_ext: str               # "jpg" or "png"
    width: int
    height: int
    sha256: str
    original_bytes: bytes
    source_ext: str


class ImageError(Exception):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_image(
    *,
    data: bytes,
    source_ext: str,
    max_long_edge: int | None,
    min_side: int = 200,
) -> PreparedImage:
    if source_ext.lower() not in ALLOWED_EXTS:
        raise ImageError(f"unsupported extension {source_ext!r}")

    try:
        img = Image.open(BytesIO(data))
        img.load()
    except UnidentifiedImageError as e:
        raise ImageError("not a decodable image") from e
    except Exception as e:
        raise ImageError(f"corrupt image: {e}") from e

    img = ImageOps.exif_transpose(img) or img   # honours EXIF orientation

    # Optional downscale on the long edge.
    if max_long_edge and max(img.size) > max_long_edge:
        scale = max_long_edge / max(img.size)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    if min(img.size) < min_side:
        raise ImageError(f"image too small: {img.size[0]}×{img.size[1]} (min side {min_side})")

    # Save as PNG for lossless inputs, JPEG otherwise.
    save_ext = "png" if source_ext.lower() in LOSSLESS_EXTS else "jpg"
    if save_ext == "jpg" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    return PreparedImage(
        image=img,
        save_ext=save_ext,
        width=img.width,
        height=img.height,
        sha256=_sha256(data),
        original_bytes=data,
        source_ext=source_ext,
    )


def store_image_at(key_base: str, prepared: PreparedImage) -> str:
    """Write the page image as <key_base>.<ext> and return its storage key,
    e.g. manual_annotation/pdf_001/pdf_001_page1.png"""
    storage = get_storage()
    image_key = f"{key_base}.{prepared.save_ext}"
    buf = BytesIO()
    if prepared.save_ext == "jpg":
        prepared.image.save(buf, format="JPEG", quality=92, optimize=True)
        mime = "image/jpeg"
    else:
        # no optimize=True: on a 300-DPI page it multiplies the save time
        prepared.image.save(buf, format="PNG", compress_level=6)
        mime = "image/png"
    storage.put(image_key, buf.getvalue(), content_type=mime)
    return image_key


def store_image(
    n: int, project_id: int, prepared: PreparedImage,
) -> str:
    """Write the corrected image to storage. Returns the image key.

    The original upload is deliberately NOT stored — only the
    EXIF-corrected, optionally downscaled copy that all coordinates refer to.
    To re-enable originals for audit, set DEVALABEL_KEEP_ORIGINALS=true.
    """
    import os
    from app.services.numbers import format_image_path

    storage = get_storage()

    image_key = f"projects/{project_id}/{format_image_path(n, prepared.save_ext)}"

    buf = BytesIO()
    if prepared.save_ext == "jpg":
        prepared.image.save(buf, format="JPEG", quality=92, optimize=True)
        mime = "image/jpeg"
    else:
        prepared.image.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    storage.put(image_key, buf.getvalue(), content_type=mime)

    if os.environ.get("DEVALABEL_KEEP_ORIGINALS", "").lower() in ("1", "true", "yes"):
        from app.services.numbers import format_original_path
        original_key = f"projects/{project_id}/{format_original_path(n, prepared.source_ext)}"
        storage.put(
            original_key,
            prepared.original_bytes,
            content_type="application/octet-stream",
        )

    return image_key