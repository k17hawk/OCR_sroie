"""Adapter protocols. Every adapter implements one of these and nothing else."""
from __future__ import annotations

from typing import Any, Protocol

from PIL.Image import Image


class Region(Protocol):
    type: str
    quad: list[list[float]]
    score: float


class LayoutDetector(Protocol):
    name: str
    version: str
    def detect(self, img: Image) -> list[dict[str, Any]]: ...


class TextDetector(Protocol):
    name: str
    version: str
    def detect(self, img: Image) -> list[list[list[float]]]: ...


class TextRecognizer(Protocol):
    name: str
    version: str
    def recognize(
        self, crops: list[Image], lang_hint: str
    ) -> list[tuple[str, float]]: ...


class TableRecognizer(Protocol):
    name: str
    version: str
    def recognize(self, img: Image, table_quad: list[list[float]]) -> dict[str, Any]: ...


class ObjectDetector(Protocol):
    name: str
    version: str
    def detect(self, img: Image) -> list[dict[str, Any]]: ...
