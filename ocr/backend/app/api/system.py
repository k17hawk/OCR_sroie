"""System introspection endpoints."""
from __future__ import annotations

import shutil
from typing import Any

from fastapi import APIRouter

from app import __version__
from app.pipeline import registry

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/models")
def list_models() -> dict[str, Any]:
    return {"families": registry.list_families()}


@router.get("/devices")
def list_devices() -> dict[str, Any]:
    """Report CPU and, if present, available CUDA devices."""
    devices: list[dict[str, Any]] = [{"id": "cpu", "kind": "cpu", "free_mb": None}]

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                devices.append({
                    "id": f"cuda:{i}",
                    "kind": "cuda",
                    "name": torch.cuda.get_device_name(i),
                    "free_mb": free // (1024 * 1024),
                    "total_mb": total // (1024 * 1024),
                })
    except ImportError:
        pass

    return {"devices": devices, "tesseract": shutil.which("tesseract") is not None,
            "version": __version__}