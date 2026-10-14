"""Adapter registry.

``load(config)`` returns a dict of instantiated adapters. The caller may
request any family name; unsupported families raise a clear error. Missing
optional dependencies are reported by the adapter itself, not here, so the
error names the exact pip package.
"""
from __future__ import annotations

from typing import Any

from app.pipeline.adapters import mock


# ---------------------------------------------------------------------------
# Capability matrix — used by /system/models
# ---------------------------------------------------------------------------

_FAMILY_CAPABILITIES: dict[str, dict[str, bool]] = {
    "mock":      {"supports_cuda": True},   # accepts anything, no-op
    "tesseract": {"supports_cuda": False},  # CPU-only
    "surya":     {"supports_cuda": True},
    "paddle":    {"supports_cuda": True},
    "tatr":      {"supports_cuda": True},
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _family_registry(family: str, device: str) -> dict[str, Any]:
    if family == "mock":
        return {
            "layout":  mock.MockLayout(),
            "text":    mock.MockText(),
            "rec":     mock.MockRec(),
            "table":   mock.MockTable(),
            "objects": mock.MockObjects(),
        }

    if family == "tesseract":
        from app.pipeline.adapters.tesseract import TesseractRecognizer
        return {
            "layout":  mock.MockLayout(),
            "text":    mock.MockText(),
            "rec":     TesseractRecognizer(device=device),
            "table":   mock.MockTable(),
            "objects": mock.MockObjects(),
        }

    if family == "surya":
        from app.pipeline.adapters.surya_ocr import (
            SuryaLayout,
            SuryaTextDetector,
            SuryaRecognizer,
        )
        return {
            "layout":  SuryaLayout(device=device),
            "text":    SuryaTextDetector(device=device),
            "rec":     SuryaRecognizer(device=device),
            "table":   mock.MockTable(),   # Surya has no table module
            "objects": mock.MockObjects(),
        }

    if family == "paddle":
        from app.pipeline.adapters.paddle import (
            PaddleLayout,
            PaddleTextDetector,
            PaddleRecognizer,
            PaddleTable,
        )
        return {
            "layout":  PaddleLayout(device=device),
            "text":    PaddleTextDetector(device=device),
            "rec":     PaddleRecognizer(device=device),
            "table":   PaddleTable(device=device),
            "objects": mock.MockObjects(),
        }

    if family == "tatr":
        from app.pipeline.adapters.tatr import TableTransformer
        return {
            "layout":  mock.MockLayout(),
            "text":    mock.MockText(),
            "rec":     mock.MockRec(),
            "table":   TableTransformer(device=device),
            "objects": mock.MockObjects(),
        }

    raise ValueError(f"unknown adapter family {family!r}")


def load(config: dict[str, Any]) -> dict[str, Any]:
    """Instantiate the adapters for the given config.

    ``config`` accepts:
      * ``family`` — one of the registered family names
      * ``device`` — "cpu" | "cuda" | "cuda:N" | "gpu:N"

    Raises ``ValueError`` for an unknown family, and the adapter itself
    raises ``RuntimeError`` for missing dependencies or an unsupported
    device.
    """
    family = config.get("family", "mock")
    device = config.get("device", "cpu")
    return _family_registry(family, device)


# ---------------------------------------------------------------------------
# System introspection — reports availability without loading any model
# ---------------------------------------------------------------------------

def list_families() -> list[dict[str, Any]]:
    """Report availability for each family."""
    import importlib.util as iu

    def _caps(family: str) -> bool:
        return _FAMILY_CAPABILITIES.get(family, {}).get("supports_cuda", False)

    out: list[dict[str, Any]] = [
        {
            "family": "mock",
            "requires": [],
            "available": True,
            "supports_cuda": _caps("mock"),
            "description": "Deterministic stub. No models, runs anywhere.",
        },
    ]

    def _check(family: str, modules: list[str], desc: str) -> None:
        avail = all(iu.find_spec(m) is not None for m in modules)
        out.append({
            "family": family,
            "requires": modules,
            "available": avail,
            "supports_cuda": _caps(family),
            "description": desc,
        })

    _check("tesseract", ["pytesseract"],
           "CPU-only recognizer (nep+hin+eng). No CUDA.")
    _check("surya", ["surya"],
           "Layout + text + recognition. Devanagari native. CPU or CUDA.")
    _check("paddle", ["paddleocr", "paddle"],
           "PP-DocLayout + DB + dev recognizer. CPU or CUDA.")
    _check("tatr", ["transformers", "torch"],
           "Table Transformer structure recognition. CPU or CUDA.")

    return out