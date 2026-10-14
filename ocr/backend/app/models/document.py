from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

if TYPE_CHECKING:
    from app.models.annotation_meta import AnnotationMeta


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("document_id", name="uq_documents_document_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE")
    )
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int] = mapped_column(Integer)
    image_key: Mapped[str] = mapped_column(String(512))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    doc_number: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    source_file_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    original_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    sha256:        Mapped[str] = mapped_column(String(64), index=True)
    text_layer:    Mapped[str | None] = mapped_column(String(1_000_000), nullable=True)
    legacy_font:   Mapped[str | None] = mapped_column(String(32), nullable=True)
    
    status: Mapped[str] = mapped_column(String(16), default="pre_annotated")
    doc_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    source_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    annotation_meta: Mapped["AnnotationMeta | None"] = relationship(
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )
