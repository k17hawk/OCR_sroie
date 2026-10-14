from __future__ import annotations
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class DocumentCounter(Base):
    """Single-row table that hands out monotonic document numbers."""
    __tablename__ = "document_counter"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    next_number: Mapped[int] = mapped_column(Integer, default=1)
