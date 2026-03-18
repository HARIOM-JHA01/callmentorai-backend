import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.connection import Base
import enum


class SessionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    call_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    client_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    call_date: Mapped[str | None] = mapped_column(String(50), nullable=True)
    audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    rubric_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("pending", "processing", "completed", "failed", name="session_status"),
        default="pending",
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    transcript: Mapped["Transcript | None"] = relationship("Transcript", back_populates="session", uselist=False)
    rubric: Mapped["Rubric | None"] = relationship("Rubric", back_populates="session", uselist=False)
    analysis: Mapped["Analysis | None"] = relationship("Analysis", back_populates="session", uselist=False)
    report: Mapped["Report | None"] = relationship("Report", back_populates="session", uselist=False)


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    utterances: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    session: Mapped["Session"] = relationship("Session", back_populates="transcript")


class Rubric(Base):
    __tablename__ = "rubrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    criteria: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    session: Mapped["Session"] = relationship("Session", back_populates="rubric")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    scores: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    strengths: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    improvements: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    key_moments: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    session: Mapped["Session"] = relationship("Session", back_populates="analysis")


# Import Report here to avoid circular imports — it lives in report.py
from app.models.report import Report  # noqa: E402, F401
