"""Create the default project (id=1) in data/_db/projects.json.

Safe to run repeatedly. The backend also does this automatically at startup.
"""
from __future__ import annotations

from app.services import projects as project_store


def seed() -> None:
    existed = project_store.get(project_store.DEFAULT_PROJECT_ID) is not None
    p = project_store.ensure_default_project()
    if existed:
        print(f"project id={p.id} already exists: name={p.name!r}")
    else:
        print(f"created project id={p.id} name={p.name!r}")


if __name__ == "__main__":
    seed()