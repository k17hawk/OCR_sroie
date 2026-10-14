"""Storage backend selection.

``get_storage()`` always returns a working backend; it never returns ``None``.
The choice between local and S3 is driven by ``settings.storage_backend``.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.storage.base import Storage


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    s = get_settings()

    if s.storage_backend == "s3":
        if not (
            s.s3_endpoint
            and s.s3_bucket
            and s.s3_access_key
            and s.s3_secret_key
        ):
            raise RuntimeError(
                "storage_backend='s3' but one of DEVALABEL_S3_ENDPOINT, "
                "DEVALABEL_S3_BUCKET, DEVALABEL_S3_ACCESS_KEY, "
                "DEVALABEL_S3_SECRET_KEY is unset"
            )
        from app.storage.s3 import S3Storage

        return S3Storage(
            endpoint=s.s3_endpoint,
            bucket=s.s3_bucket,
            access_key=s.s3_access_key,
            secret_key=s.s3_secret_key,
            region=s.s3_region,
        )

    # Default: local disk.
    from app.storage.local import LocalStorage

    return LocalStorage(s.storage_root)