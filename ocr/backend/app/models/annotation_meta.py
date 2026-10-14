from __future__ import annotations
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class AnnotationMeta(Base):
    __tablename__ = "annotation_meta"

    document_pk: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    annotator_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewer_id:  Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_comment: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    document: Mapped["Document"] = relationship(back_populates="annotation_meta")  # noqa: F821
