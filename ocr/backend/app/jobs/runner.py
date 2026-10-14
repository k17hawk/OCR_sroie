"""In-memory job runner for pre-annotation. No database."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from app.core.finalize import finalize
from app.pipeline.preannotate import run as run_preannotate
from app.services import documents as doc_service
from app.storage import get_storage


_jobs: dict[int, dict[str, Any]] = {}
_next_id = 0
_lock = Lock()
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="devalabel-job")


def enqueue_preannotate(*, document_id: str, family: str, device: str) -> dict[str, Any]:
    global _next_id
    with _lock:
        _next_id += 1
        jid = _next_id
        job: dict[str, Any] = {
            "id": jid,
            "document_id": document_id,
            "kind": "preannotate",
            "status": "queued",
            "progress": 0.0,
            "device": device,
            "family": family,
            "result": None,
            "error": None,
        }
        _jobs[jid] = job
    _pool.submit(_run, jid)
    return job


def get(job_id: int) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def _run(job_id: int) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    job["status"] = "running"
    job["progress"] = 0.05
    try:
        row = doc_service.get(job["document_id"])
        if row is None:
            job["status"] = "failed"
            job["error"] = "document missing"
            return

        image_bytes = get_storage().get(row.image_key)
        new_doc, stats = run_preannotate(
            doc=row.doc,
            image_bytes=image_bytes,
            config={"family": job["family"]},
            device=job["device"],
        )
        new_doc = finalize(new_doc)
        row.doc = new_doc
        row.revision += 1
        doc_service.save(row)

        job["status"] = "succeeded"
        job["progress"] = 1.0
        job["result"] = stats
    except Exception as e:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(e)