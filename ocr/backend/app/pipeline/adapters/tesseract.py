"""Tesseract recognizer — CPU-only fallback for Devanagari + English.

Requires the system binary ``tesseract`` plus the language packs for
``nep``, ``hin``, and ``eng``.
"""
from __future__ import annotations

from PIL.Image import Image


class TesseractRecognizer:
    name = "tesseract"
    version = "5.x"
    family = "tesseract"

    def __init__(self, device: str = "cpu") -> None:
        if device != "cpu":
            raise RuntimeError(
                f"{self.name} is CPU-only and cannot use device={device!r}. "
                "Pick a CUDA-capable family (surya, paddle, tatr) or select 'cpu'."
            )
        self.device = "cpu"

    def recognize(
        self, crops: list[Image], lang_hint: str
    ) -> list[tuple[str, float]]:
        try:
            import pytesseract
        except ImportError as e:
            raise RuntimeError(
                "tesseract adapter requires pytesseract + the system tesseract binary. "
                "Install: pip install pytesseract && apt install tesseract-ocr "
                "tesseract-ocr-nep tesseract-ocr-hin tesseract-ocr-eng"
            ) from e

        # Map our internal language code to Tesseract's lang string.
        lang = {
            "ne": "nep+eng",
            "hi": "hin+eng",
            "en": "eng",
            "mixed": "nep+hin+eng",
        }.get(lang_hint, "nep+hin+eng")

        out: list[tuple[str, float]] = []
        for crop in crops:
            try:
                # ``image_to_data`` gives per-word confidence; average it.
                data = pytesseract.image_to_data(
                    crop, lang=lang, output_type=pytesseract.Output.DICT
                )
                texts = [t for t in data["text"] if t.strip()]
                confs = [
                    int(c) / 100.0
                    for c, t in zip(data["conf"], data["text"])
                    if t.strip() and int(c) >= 0
                ]
                text = " ".join(texts)
                conf = sum(confs) / len(confs) if confs else 0.0
                out.append((text, conf))
            except Exception as e:  # noqa: BLE001
                out.append(("", 0.0))
        return out