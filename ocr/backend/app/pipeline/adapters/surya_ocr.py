"""Surya layout, text detection and recognition.

Surya supports Devanagari (Hindi, Nepali) out of the box and works on CPU
or CUDA. Models are downloaded to the HuggingFace cache on first use.
"""
from __future__ import annotations

from typing import Any

from PIL.Image import Image


class SuryaLayout:
    name = "surya-layout"
    version = "0.4"
    family = "surya"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from surya.layout import LayoutPredictor
        except ImportError as e:
            raise RuntimeError(
                "Surya adapter requires `pip install surya-ocr`"
            ) from e
        self._model = LayoutPredictor(device=self.device)
        return self._model

    def detect(self, img: Image) -> list[dict[str, Any]]:
        model = self._load()
        results = model([img])
        out: list[dict[str, Any]] = []
        for page in results:
            for box in page.bboxes:
                out.append({
                    "type": _map_label(box.label),
                    "score": float(box.confidence),
                    "quad": [
                        [box.bbox[0], box.bbox[1]],
                        [box.bbox[2], box.bbox[1]],
                        [box.bbox[2], box.bbox[3]],
                        [box.bbox[0], box.bbox[3]],
                    ],
                })
        return out


class SuryaTextDetector:
    name = "surya-det"
    version = "0.4"
    family = "surya"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from surya.detection import DetectionPredictor
        except ImportError as e:
            raise RuntimeError("Surya adapter requires `pip install surya-ocr`") from e
        self._model = DetectionPredictor(device=self.device)
        return self._model

    def detect(self, img: Image) -> list[list[list[float]]]:
        model = self._load()
        results = model([img])
        quads: list[list[list[float]]] = []
        for page in results:
            for line in page.bboxes:
                x0, y0, x1, y1 = line.bbox
                quads.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return quads


class SuryaRecognizer:
    name = "surya-rec"
    version = "0.4"
    family = "surya"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from surya.recognition import RecognitionPredictor
        except ImportError as e:
            raise RuntimeError("Surya adapter requires `pip install surya-ocr`") from e
        self._model = RecognitionPredictor(device=self.device)
        return self._model

    def recognize(
        self, crops: list[Image], lang_hint: str
    ) -> list[tuple[str, float]]:
        model = self._load()
        langs = {
            "ne": ["ne", "en"],
            "hi": ["hi", "en"],
            "en": ["en"],
            "mixed": ["ne", "hi", "en"],
        }.get(lang_hint, ["ne", "en"])
        try:
            results = model(crops, [langs] * len(crops))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"surya recognition failed: {e}") from e
        out: list[tuple[str, float]] = []
        for page in results:
            if page.text_lines:
                tl = page.text_lines[0]
                out.append((tl.text, float(getattr(tl, "confidence", 0.8))))
            else:
                out.append(("", 0.0))
        return out


_LABEL_MAP = {
    "SectionHeader": "heading",
    "Title": "heading",
    "Text": "paragraph",
    "ListItem": "list",
    "Table": "table",
    "Figure": "figure",
    "Picture": "figure",
    "Caption": "paragraph",
    "PageHeader": "header",
    "PageFooter": "footer",
    "Footnote": "footer",
}


def _map_label(raw: str) -> str:
    return _LABEL_MAP.get(raw, "paragraph")