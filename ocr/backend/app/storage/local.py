"""Local-disk storage. Reads/writes under a configured root directory."""
from __future__ import annotations

from pathlib import Path

from app.storage.base import Storage


class LocalStorage(Storage):
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------
    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root) + "/") and p != self.root:
            raise ValueError(f"path traversal: {key!r}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Storage API
    # ------------------------------------------------------------------
    def put(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        self._p(key).write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).is_file()

    def delete(self, key: str) -> None:
        p = self._p(key)
        if p.is_file():
            p.unlink()

    def url(self, key: str, expires: int = 3600) -> str:
        # Served by the API at /files/{key} in main.py.
        return f"/files/{key}"