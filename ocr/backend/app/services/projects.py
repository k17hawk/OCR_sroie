"""Project storage backed by a JSON file at data/_db/projects.json."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.storage.doc_store import _atomic_write
from app.schemas.doc_v2 import REGION_TYPES


DEFAULT_DOC_TYPES: list[str] = [
    "bank_account_form", "citizenship", "national_id", "passport",
    "driving_license", "pan_card", "voter_id", "invoice", "receipt",
    "bank_cheque", "bank_statement", "marksheet", "certificate",
    "land_ownership", "application_letter", "government_notice", "government_form",
    "other",
]

DEFAULT_FIELD_TYPES: list[str] = [
    "text", "name", "name_m", "name_f", "date", "date_bs", "date_ad",
    "checkbox", "citizenship", "national_id", "pan", "passport_no",
    "district", "municipality", "ward", "address", "phone", "email",
    "money", "account_number", "number", "signature_ref", "other",
]

DEFAULT_LAYOUT_TYPES: list[str] = [
    # the region types of the v2 schema (doc_v2.REGION_TYPES); older type
    # names are mapped on load by core/migrate.py
    "title", "heading", "subheading", "paragraph", "list", "sublist", "form_field",
    "table", "table_heading", "header", "footer", "page_number", "figure", "logo", "stamp",
    "signature", "handwriting", "caption",
]


@dataclass
class Project:
    id: int
    name: str
    doc_types: list[str] = field(default_factory=lambda: list(DEFAULT_DOC_TYPES))
    field_types: list[str] = field(default_factory=lambda: list(DEFAULT_FIELD_TYPES))
    layout_types: list[str] = field(default_factory=lambda: list(DEFAULT_LAYOUT_TYPES))
    label_map: dict[str, Any] = field(default_factory=dict)
    model_settings: dict[str, Any] = field(default_factory=dict)
    max_long_edge: int | None = None
    default_dpi: int = 200
    default_language: str = "ne"
    max_upload_mb: int = 100
    require_doc_type_on_submit: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _path() -> Path:
    root = Path(get_settings().storage_root) / "_db"
    root.mkdir(parents=True, exist_ok=True)
    return root / "projects.json"


def _read_all() -> list[Project]:
    p = _path()
    if not p.exists():
        return []
    projects = [Project(**item) for item in json.loads(p.read_text(encoding="utf-8"))]
    for pr in projects:
        # a project saved before the type list changed may still list old
        # region types (text_line, list_item, …): offer only current ones
        kept = [t for t in pr.layout_types if t in REGION_TYPES]
        if "list" in kept and "sublist" not in kept:
            kept.insert(kept.index("list") + 1, "sublist")
        if "page_number" not in kept:
            kept.insert(kept.index("footer") + 1 if "footer" in kept else len(kept), "page_number")
        pr.layout_types = kept or list(DEFAULT_LAYOUT_TYPES)
    return projects


def _write_all(projects: list[Project]) -> None:
    _atomic_write(
        _path(),
        json.dumps([asdict(p) for p in projects], indent=2, ensure_ascii=False) + "\n",
    )


def create(name: str, **kwargs: Any) -> Project:
    projects = _read_all()
    if any(p.name == name for p in projects):
        raise ValueError(f"project name {name!r} already exists")
    next_id = max((p.id for p in projects), default=0) + 1
    project = Project(id=next_id, name=name, **kwargs)
    projects.append(project)
    _write_all(projects)
    return project


def get(project_id: int) -> Project | None:
    for p in _read_all():
        if p.id == project_id:
            return p
    return None


def get_by_name(name: str) -> Project | None:
    for p in _read_all():
        if p.name == name:
            return p
    return None


def list_all() -> list[Project]:
    return sorted(_read_all(), key=lambda p: p.id)


def update(project: Project, patch: dict[str, Any]) -> Project:
    projects = _read_all()
    for i, p in enumerate(projects):
        if p.id != project.id:
            continue
        for key, value in patch.items():
            if key == "max_long_edge":
                setattr(p, key, value)      # None is meaningful here
            elif value is None:
                continue
            else:
                setattr(p, key, value)
        p.updated_at = datetime.now(timezone.utc).isoformat()
        projects[i] = p
        _write_all(projects)
        return p
    raise KeyError(f"project {project.id} not found")

DEFAULT_PROJECT_ID = 1


def ensure_default_project() -> Project:
    """Make sure project id=1 (the one the frontend uses) exists. Safe to call repeatedly."""
    existing = get(DEFAULT_PROJECT_ID)
    if existing is not None:
        return existing
    projects = _read_all()
    taken = {p.name for p in projects}
    name = "local" if "local" not in taken else f"local-{DEFAULT_PROJECT_ID}"
    project = Project(
        id=DEFAULT_PROJECT_ID,
        name=name,
        model_settings={"family": "mock", "device": "cpu"},
        default_dpi=300,
        default_language="ne",
    )
    projects.append(project)
    _write_all(sorted(projects, key=lambda p: p.id))
    return project