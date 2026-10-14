"""Job status endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.jobs import runner as job_runner
from app.schemas.api import JobRead

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: int) -> JobRead:
    job = job_runner.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JobRead(**{
        "id": job["id"],
        "document_id": job["document_id"],
        "kind": job["kind"],
        "status": job["status"],
        "progress": job["progress"],
        "device": job["device"],
        "result": job.get("result"),
        "error": job.get("error"),
    })