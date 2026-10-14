"""Microsoft Table Transformer for table structure recognition.

Uses HuggingFace ``transformers``. Weights download on first use and cache
under ``~/.cache/huggingface``.
"""
from __future__ import annotations

from typing import Any

from PIL.Image import Image


class TableTransformer:
    name = "microsoft-table-transformer"
    version = "1.1"
    family = "tatr"

    HF_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._processor, self._model

        import importlib.util as iu
        missing = [m for m in ("transformers", "torch") if iu.find_spec(m) is None]
        if missing:
            raise RuntimeError(
                "Table Transformer requires: pip install transformers torch "
                f"(missing: {', '.join(missing)})"
            )

        from transformers import (  # type: ignore
            AutoImageProcessor,
            TableTransformerForObjectDetection,
        )

        self._processor = AutoImageProcessor.from_pretrained(self.HF_MODEL)
        self._model = TableTransformerForObjectDetection.from_pretrained(self.HF_MODEL)
        self._model.to(self.device)
        self._model.eval()
        return self._processor, self._model

    def recognize(
        self, img: Image, table_quad: list[list[float]]
    ) -> dict[str, Any]:
        processor, model = self._load()
        import torch  # type: ignore

        inputs = processor(images=img.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=0.6, target_sizes=target_sizes
        )[0]

        rows_det: list[dict[str, Any]] = []
        cols_det: list[dict[str, Any]] = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            kind = model.config.id2label[int(label)]
            item = {"kind": kind, "score": float(score),
                    "bbox": [int(x0), int(y0), int(x1), int(y1)]}
            if kind == "table row":
                rows_det.append(item)
            elif kind == "table column":
                cols_det.append(item)

        # Cross product -> cell bboxes
        cells: list[dict[str, Any]] = []
        rows_det.sort(key=lambda c: c["bbox"][1])
        cols_det.sort(key=lambda c: c["bbox"][0])
        for r, rb in enumerate(rows_det):
            for c, cb in enumerate(cols_det):
                cx0 = max(rb["bbox"][0], cb["bbox"][0])
                cy0 = max(rb["bbox"][1], cb["bbox"][1])
                cx1 = min(rb["bbox"][2], cb["bbox"][2])
                cy1 = min(rb["bbox"][3], cb["bbox"][3])
                if cx1 > cx0 and cy1 > cy0:
                    cells.append({
                        "row": r, "col": c,
                        "rowspan": 1, "colspan": 1,
                        "bbox": [cx0, cy0, cx1, cy1],
                    })

        return {
            "rows": len(rows_det),
            "cols": len(cols_det),
            "cells": cells,
        }