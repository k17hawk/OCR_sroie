"""PaddleOCR adapters (layout, text detection, Devanagari recognition, tables).

Targets PaddleOCR 2.x and 3.x. The table module moved between releases, so
``PaddleTable`` tries each known import path and falls back to a uniform
grid if none is available.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL.Image import Image


# ---------------------------------------------------------------------------
# Device normalisation
# ---------------------------------------------------------------------------

def _normalize_device(device: str) -> str:
    """PaddleOCR expects ``cpu`` or ``gpu:N``."""
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda"):
        idx = device.split(":", 1)[1] if ":" in device else "0"
        return f"gpu:{idx}"
    if device.startswith("gpu"):
        return device
    return device


def _check_paddle() -> None:
    """Raise a clear RuntimeError if the PaddleOCR stack isn't importable."""
    import importlib.util as iu
    missing = [m for m in ("paddle", "paddleocr") if iu.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            "PaddleOCR adapter requires: pip install paddlepaddle paddleocr "
            f"(missing: {', '.join(missing)})"
        )


# ---------------------------------------------------------------------------
# PaddleOCR constructor — handles both 2.x and 3.x signatures
# ---------------------------------------------------------------------------

def _make_ocr(device: str, lang: str = "devanagari") -> Any:
    _check_paddle()
    from paddleocr import PaddleOCR  # type: ignore

    kwargs: dict[str, Any] = dict(use_angle_cls=True, lang=lang, show_log=False)

    # PaddleOCR 3.x uses `device`; 2.x uses `use_gpu`.
    try:
        return PaddleOCR(device=device, **kwargs)
    except TypeError:
        return PaddleOCR(use_gpu=device.startswith("gpu"), **kwargs)


# ---------------------------------------------------------------------------
# Layout — uses OCR blocks as regions
# ---------------------------------------------------------------------------

class PaddleLayout:
    name = "pp-doclayout"
    version = "3.x"
    family = "paddle"

    def __init__(self, device: str = "cpu") -> None:
        self.device = _normalize_device(device)
        self._model = None

    def _load(self) -> Any:
        if self._model is None:
            self._model = _make_ocr(self.device)
        return self._model

    def detect(self, img: Image) -> list[dict[str, Any]]:
        model = self._load()
        arr = np.asarray(img.convert("RGB"))
        try:
            raw = model.ocr(arr, cls=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"paddle layout failed: {e}") from e

        out: list[dict[str, Any]] = []
        for page in raw or []:
            for item in page or []:
                try:
                    quad, (_, conf) = item[0], item[1]
                except Exception:
                    continue
                out.append({
                    "type": "paragraph",
                    "score": float(conf),
                    "quad": [[float(p[0]), float(p[1])] for p in quad],
                })
        return out


# ---------------------------------------------------------------------------
# Text detection
# ---------------------------------------------------------------------------

class PaddleTextDetector:
    name = "paddle-db"
    version = "3.x"
    family = "paddle"

    def __init__(self, device: str = "cpu") -> None:
        self.device = _normalize_device(device)
        self._paddle_layout = PaddleLayout(device)

    def detect(self, img: Image) -> list[list[list[float]]]:
        return [r["quad"] for r in self._paddle_layout.detect(img)]


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

class PaddleRecognizer:
    name = "paddle-dev"
    version = "3.x"
    family = "paddle"

    def __init__(self, device: str = "cpu") -> None:
        self.device = _normalize_device(device)
        self._model = None

    def _load(self) -> Any:
        if self._model is None:
            self._model = _make_ocr(self.device)
        return self._model

    def recognize(
        self, crops: list[Image], lang_hint: str
    ) -> list[tuple[str, float]]:
        model = self._load()
        out: list[tuple[str, float]] = []
        for crop in crops:
            arr = np.asarray(crop.convert("RGB"))
            try:
                raw = model.ocr(arr, cls=True)
            except Exception:  # noqa: BLE001
                out.append(("", 0.0))
                continue
            best_text, best_conf = "", 0.0
            for page in raw or []:
                for item in page or []:
                    try:
                        text, conf = item[1][0], float(item[1][1])
                    except Exception:
                        continue
                    if conf > best_conf:
                        best_text, best_conf = text, conf
            out.append((best_text, best_conf))
        return out


# ---------------------------------------------------------------------------
# Table structure — tries every known import path, falls back to a grid
# ---------------------------------------------------------------------------

class PaddleTable:
    name = "pp-structure-table"
    version = "3.x"
    family = "paddle"

    def __init__(self, device: str = "cpu") -> None:
        self.device = _normalize_device(device)
        self._engine: Any = None
        self._engine_kind: str = "unavailable"

    def _load(self) -> None:
        if self._engine is not None:
            return

        # Path 1 — PaddleOCR 2.x top-level
        try:
            from paddleocr import PPStructure  # type: ignore
            self._engine = PPStructure(show_log=False, layout=False)
            self._engine_kind = "ppstructure"
            return
        except Exception:
            pass

        # Path 2 — PaddleOCR 3.x table submodule
        try:
            from paddleocr import TableRecognitionPipeline  # type: ignore
            self._engine = TableRecognitionPipeline(
                device=self.device, show_log=False
            )
            self._engine_kind = "pipeline"
            return
        except Exception:
            pass

        # Path 3 — PaddleX wrapper (some installs ship it inside paddleocr)
        try:
            from paddleocr.ppstructure.table.predict_table import (  # type: ignore
                TableSystem,
            )
            self._engine = TableSystem()
            self._engine_kind = "tablesystem"
            return
        except Exception:
            pass

        # Nothing available. Record that; recognize() will fall back.
        self._engine_kind = "unavailable"

    def recognize(
        self, img: Image, table_quad: list[list[float]]
    ) -> dict[str, Any]:
        self._load()

        # Engine missing: report "no structure". table_ocr then builds the
        # grid from ruling lines / text. (A made-up 2x2 grid here used to
        # override that and lose every other <td>.)
        if self._engine_kind == "unavailable":
            return dict(_NO_TABLE)

        arr = np.asarray(img.convert("RGB"))

        try:
            if self._engine_kind == "ppstructure":
                result = self._engine(arr)
                for item in result or []:
                    if item.get("type") == "table":
                        html = (item.get("res") or {}).get("html", "")
                        if html:
                            return {"html": html, "rows": 0, "cols": 0, "cells": []}
                # No table found — fall back
                return dict(_NO_TABLE)

            if self._engine_kind == "pipeline":
                result = self._engine.predict(arr)
                for item in result or []:
                    html = item.get("html") or item.get("pred_html") or ""
                    if html:
                        return {"html": html, "rows": 0, "cols": 0, "cells": []}
                return dict(_NO_TABLE)

            if self._engine_kind == "tablesystem":
                result = self._engine(arr)
                html = (
                    (result or {}).get("html")
                    if isinstance(result, dict)
                    else ""
                ) or ""
                if html:
                    return {"html": html, "rows": 0, "cols": 0, "cells": []}
                return dict(_NO_TABLE)

        except Exception as e:  # noqa: BLE001
            # Don't crash the whole OCR pipeline on a table-detection failure.
            return {**_NO_TABLE, "_error": f"paddle table failed: {e}"}

        return dict(_NO_TABLE)


# ---------------------------------------------------------------------------
# Fallback grid
# ---------------------------------------------------------------------------

_NO_TABLE: dict[str, Any] = {"rows": 0, "cols": 0, "cells": []}


def _uniform_grid(
    quad: list[list[float]], *, rows: int, cols: int
) -> dict[str, Any]:
    """Divide the quad's bbox into a rows×cols grid of empty cells."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))

    cells: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            cx0 = x0 + (x1 - x0) * c // cols
            cx1 = x0 + (x1 - x0) * (c + 1) // cols
            cy0 = y0 + (y1 - y0) * r // rows
            cy1 = y0 + (y1 - y0) * (r + 1) // rows
            cells.append({
                "row": r, "col": c,
                "rowspan": 1, "colspan": 1,
                "bbox": [cx0, cy0, cx1, cy1],
            })
    return {"rows": rows, "cols": cols, "cells": cells}