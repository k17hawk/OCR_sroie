"""Legacy module. DevaLabel now uses file-backed storage.

Documents live under ``data/_db/documents/``; projects live in
``data/_db/projects.json``. This module exists only so that any code path
still importing ``get_session`` or ``Base`` doesn't crash.
"""
from __future__ import annotations


def get_session():
    """Always yields ``None``. Any endpoint that still declares
    ``db: Session = Depends(get_session)`` will get ``None`` and should be
    removed — the file-backed services don't take a session."""
    yield None


class Base:
    """Placeholder for ``Base.metadata`` references in dead code."""
    metadata = None