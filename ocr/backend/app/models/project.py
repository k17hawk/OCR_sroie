from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


# ---------------------------------------------------------------------------
# Defaults used by the seed script and by new-project creation.
# Never hard-code these into application logic — always read them from the
# project row so an admin can edit them without a code change.
# ---------------------------------------------------------------------------

DEFAULT_DOC_TYPES: list[str] = [
    "bank_account_form", "citizenship", "national_id", "passport",
    "driving_license", "pan_card", "voter_id", "invoice", "receipt",
    "bank_cheque", "bank_statement", "marksheet", "certificate",
    "land_ownership", "application_letter", "government_notice", "other",
]

DEFAULT_FIELD_TYPES: list[str] = [
    "text", "name", "name_m", "name_f", "date", "date_bs", "date_ad",
    "checkbox", "citizenship", "national_id", "pan", "passport_no",
    "district", "municipality", "ward", "address", "phone", "email",
    "money", "account_number", "number", "signature_ref", "other",
]

DEFAULT_LAYOUT_TYPES: list[str] = [
    # structural
    "header", "title", "heading", "subheading", "paragraph",
    "text_line", "footer", "page_number", "signature_caption",
    # blocks
    "figure", "table", "table_heading", "table_footer", "list", "list_item",
    # forms
    "form_field", "form_label", "form_value", "label", "options",
    # marks
    "logo", "signature", "stamp",
    # handwriting
    "handwriting", "handwritten_annotation",
    "handwritten_correction", "handwritten_number",
    # misc
    "mixed_text",
]


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)

    # Per-project vocabularies. Editable by an admin without a code change.
    doc_types: Mapped[list[str]] = mapped_column(
        JSON, default=lambda: list(DEFAULT_DOC_TYPES)
    )
    field_types: Mapped[list[str]] = mapped_column(
        JSON, default=lambda: list(DEFAULT_FIELD_TYPES)
    )
    layout_types: Mapped[list[str]] = mapped_column(
        JSON, default=lambda: list(DEFAULT_LAYOUT_TYPES)
    )

    label_map: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    model_settings: Mapped[dict[str, Any]] = mapped_column(
        "model_config", JSON, default=dict
    )

    # Ingestion settings
    max_long_edge: Mapped[int | None] = mapped_column(Integer, nullable=True)
    default_dpi: Mapped[int] = mapped_column(Integer, default=200)
    default_language: Mapped[str] = mapped_column(String(8), default="ne")
    max_upload_mb: Mapped[int] = mapped_column(Integer, default=100)
    require_doc_type_on_submit: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r}>"