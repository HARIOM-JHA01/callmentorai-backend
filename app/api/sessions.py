import os
import uuid
import logging
import aiofiles
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database.connection import get_db
from app.models.session import Session, Transcript, Analysis
from app.pipelines.analysis_pipeline import run_analysis_pipeline
from app.services.progress import get_progress

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/session", tags=["sessions"])

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}
ALLOWED_RUBRIC_EXTENSIONS = {".pdf"}


def _validate_extension(filename: str, allowed: set[str]) -> str:
    """Return the lowercase extension or raise HTTPException."""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' has unsupported extension '{ext}'. Allowed: {sorted(allowed)}",
        )
    return ext


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_session(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Call audio file (.mp3, .wav, .m4a, .ogg)"),
    rubric_pdf: UploadFile = File(..., description="Evaluation rubric PDF"),
    agent_name: Optional[str] = Form(None),
    client_name: Optional[str] = Form(None),
    call_title: Optional[str] = Form(None),
    call_date: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a call audio file and rubric PDF to create a new analysis session.
    Triggers the full analysis pipeline as a background task.
    """
    # Validate file extensions
    audio_ext = _validate_extension(audio_file.filename or "file.unknown", ALLOWED_AUDIO_EXTENSIONS)
    _validate_extension(rubric_pdf.filename or "file.unknown", ALLOWED_RUBRIC_EXTENSIONS)

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(settings.UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save audio file
    audio_filename = f"audio{audio_ext}"
    audio_path = os.path.join(session_dir, audio_filename)
    async with aiofiles.open(audio_path, "wb") as out_file:
        content = await audio_file.read()
        await out_file.write(content)

    # Save rubric PDF
    rubric_path = os.path.join(session_dir, "rubric.pdf")
    async with aiofiles.open(rubric_path, "wb") as out_file:
        content = await rubric_pdf.read()
        await out_file.write(content)

    # Persist session metadata
    session = Session(
        id=session_id,
        call_title=call_title,
        agent_name=agent_name,
        client_name=client_name,
        call_date=call_date,
        audio_path=audio_path,
        rubric_path=rubric_path,
        status="pending",
    )
    db.add(session)
    await db.commit()

    # Kick off background pipeline
    background_tasks.add_task(run_analysis_pipeline, session_id)
    logger.info(f"Session {session_id} created; analysis pipeline queued")

    return {"session_id": session_id}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return session status and metadata."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session: Session | None = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    progress = get_progress(session.id) if session.status == "processing" else None

    return {
        "session_id": session.id,
        "call_title": session.call_title,
        "agent_name": session.agent_name,
        "client_name": session.client_name,
        "call_date": session.call_date,
        "status": session.status,
        "error_message": session.error_message,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "progress_pct": progress["pct"] if progress else None,
        "progress_stage": progress["stage"] if progress else None,
    }


@router.get("/{session_id}/transcript")
async def get_transcript(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the transcript utterances for a session."""
    # Ensure session exists
    session_result = await db.execute(select(Session).where(Session.id == session_id))
    session: Session | None = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.status == "pending":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="Session is still pending — analysis has not started yet",
        )
    if session.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="Session is still being processed — transcript not yet available",
        )
    if session.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Session analysis failed: {session.error_message}",
        )

    transcript_result = await db.execute(select(Transcript).where(Transcript.session_id == session_id))
    transcript: Transcript | None = transcript_result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found for this session")

    return {
        "session_id": session_id,
        "utterances": transcript.utterances,
        "total": len(transcript.utterances),
    }


@router.get("/{session_id}/analysis")
async def get_analysis(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the call analysis results for a session."""
    session_result = await db.execute(select(Session).where(Session.id == session_id))
    session: Session | None = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.status in ("pending", "processing"):
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Session is {session.status} — analysis not yet available",
        )
    if session.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Session analysis failed: {session.error_message}",
        )

    analysis_result = await db.execute(select(Analysis).where(Analysis.session_id == session_id))
    analysis: Analysis | None = analysis_result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found for this session")

    return {
        "session_id": session_id,
        "scores": analysis.scores,
        "strengths": analysis.strengths,
        "improvements": analysis.improvements,
        "key_moments": analysis.key_moments,
    }
