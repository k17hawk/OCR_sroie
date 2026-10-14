"""Deterministic mock adapters so the whole preannotate flow runs without models."""
from __future__ import annotations

from typing import Any

from PIL.Image import Image


class MockLayout:
    name, version = "mock-layout", "0.0.1"

    def detect(self, img: Image) -> list[dict[str, Any]]:
        # Return one header + one paragraph, both plausible for an A4 scan.
        w, h = img.size
        return [
            {"type": "header", "score": 0.9, "quad":
                [[20, 20], [w - 20, 20], [w - 20, 80], [20, 80]]},
            {"type": "paragraph", "score": 0.8, "quad":
                [[20, 100], [w - 20, 100], [w - 20, 220], [20, 220]]},
        ]


class MockText:
    name, version = "mock-text", "0.0.1"

    def detect(self, img: Image) -> list[list[list[float]]]:
        w, _ = img.size
        return [
            [[30, 30], [w - 30, 30], [w - 30, 60], [30, 60]],
            [[30, 110], [w - 30, 110], [w - 30, 140], [30, 140]],
        ]


class MockRec:
    name, version = "mock-rec", "0.0.1"

    def recognize(self, crops, lang_hint):
        return [("[mock]", 0.5) for _ in crops]


class MockTable:
    name, version = "mock-table", "0.0.1"

    def recognize(self, img, table_quad):
        return {"rows": 2, "cols": 2, "cells": []}


class MockObjects:
    name, version = "mock-objects", "0.0.1"

    def detect(self, img):
        return []
