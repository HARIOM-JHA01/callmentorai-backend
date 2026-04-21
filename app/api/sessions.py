import os
import uuid
import logging
import aiofiles
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.config import get_settings
from app.database.connection import get_db
from app.models.session import Session, Transcript, Rubric, Analysis
from app.models.report import Report
from app.models.user import User
from app.services.progress import get_progress
from app.services.auth import get_optional_user
from app.services.processing_queue import processing_queue

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


async def _save_audio(audio_file: UploadFile, session_dir: str) -> str:
    audio_ext = _validate_extension(
        audio_file.filename or "file.unknown", ALLOWED_AUDIO_EXTENSIONS
    )
    audio_path = os.path.join(session_dir, f"audio{audio_ext}")
    async with aiofiles.open(audio_path, "wb") as f:
        await f.write(await audio_file.read())
    return audio_path


async def _save_rubric(rubric_pdf: UploadFile, session_dir: str) -> str:
    _validate_extension(rubric_pdf.filename or "file.unknown", ALLOWED_RUBRIC_EXTENSIONS)
    rubric_path = os.path.join(session_dir, "rubric.pdf")
    async with aiofiles.open(rubric_path, "wb") as f:
        await f.write(await rubric_pdf.read())
    return rubric_path


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_session(
    audio_file: UploadFile = File(..., description="Call audio file (.mp3, .wav, .m4a, .ogg)"),
    rubric_pdf: Optional[UploadFile] = File(None, description="Evaluation rubric PDF (optional)"),
    agent_name: Optional[str] = Form(None),
    client_name: Optional[str] = Form(None),
    call_title: Optional[str] = Form(None),
    call_date: Optional[str] = Form(None),
    team: Optional[str] = Form(None),
    supervisor: Optional[str] = Form(None),
    campaign: Optional[str] = Form(None),
    queue: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """Upload a call recording (and optional rubric) to create a new analysis session."""
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(settings.UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    audio_path = await _save_audio(audio_file, session_dir)
    rubric_path = await _save_rubric(rubric_pdf, session_dir) if rubric_pdf else None

    session = Session(
        id=session_id,
        call_title=call_title,
        agent_name=agent_name,
        client_name=client_name,
        call_date=call_date,
        team=team,
        supervisor=supervisor,
        campaign=campaign,
        queue=queue,
        audio_path=audio_path,
        rubric_path=rubric_path,
        status="pending",
        user_id=current_user.id if current_user else None,
    )
    db.add(session)
    await db.commit()

    processing_queue.enqueue(session_id)
    logger.info(f"Session {session_id} created and queued for analysis")

    return {"session_id": session_id}


@router.post("/batch-upload", status_code=status.HTTP_201_CREATED)
async def batch_upload_sessions(
    audio_files: List[UploadFile] = File(..., description="Multiple call audio files"),
    rubric_pdf: Optional[UploadFile] = File(None, description="Shared evaluation rubric PDF (optional)"),
    agent_name: Optional[str] = Form(None),
    team: Optional[str] = Form(None),
    supervisor: Optional[str] = Form(None),
    campaign: Optional[str] = Form(None),
    queue: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Upload multiple call recordings in a single request.
    An optional rubric PDF is shared across all files.
    Returns all created session IDs immediately; processing is queued.
    """
    if not audio_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one audio file is required.",
        )

    # Read shared rubric bytes once so we don't re-read for every session
    rubric_bytes: Optional[bytes] = None
    rubric_filename: Optional[str] = None
    if rubric_pdf:
        _validate_extension(rubric_pdf.filename or "file.unknown", ALLOWED_RUBRIC_EXTENSIONS)
        rubric_bytes = await rubric_pdf.read()
        rubric_filename = rubric_pdf.filename

    session_ids: list[str] = []

    for audio_file in audio_files:
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(settings.UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        audio_path = await _save_audio(audio_file, session_dir)

        rubric_path: Optional[str] = None
        if rubric_bytes is not None:
            rubric_path = os.path.join(session_dir, "rubric.pdf")
            async with aiofiles.open(rubric_path, "wb") as f:
                await f.write(rubric_bytes)

        # Derive a call title from the original filename (strip extension)
        derived_title = os.path.splitext(audio_file.filename or "")[0] or None

        session = Session(
            id=session_id,
            call_title=derived_title,
            agent_name=agent_name,
            team=team,
            supervisor=supervisor,
            campaign=campaign,
            queue=queue,
            audio_path=audio_path,
            rubric_path=rubric_path,
            status="pending",
            user_id=current_user.id if current_user else None,
        )
        db.add(session)
        session_ids.append(session_id)

    await db.commit()

    for sid in session_ids:
        processing_queue.enqueue(sid)

    logger.info(f"Batch upload: {len(session_ids)} sessions created and queued")
    return {"session_ids": session_ids, "queued": len(session_ids)}


@router.get("/queue")
async def get_queue_stats():
    """Return the current processing queue depth."""
    return processing_queue.queue_stats()


@router.get("/all")
async def get_all_sessions(
    db: AsyncSession = Depends(get_db),
):
    """Return all sessions with complete information (transcript, rubric, analysis, report)."""
    result = await db.execute(select(Session).order_by(desc(Session.created_at)))
    sessions = result.scalars().all()

    all_sessions = []
    for session in sessions:
        meta_es = (session.metadata_es or {}) if hasattr(session, "metadata_es") else {}

        session_data = {
            "session_id": session.id,
            "call_title": session.call_title,
            "call_title_es": meta_es.get("call_title"),
            "agent_name": session.agent_name,
            "agent_name_es": meta_es.get("agent_name"),
            "client_name": session.client_name,
            "client_name_es": meta_es.get("client_name"),
            "call_date": session.call_date,
            "status": session.status,
            "error_message": session.error_message,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        }

        transcript_data = None
        if session.status == "completed":
            transcript_result = await db.execute(
                select(Transcript).where(Transcript.session_id == session.id)
            )
            transcript = transcript_result.scalar_one_or_none()
            if transcript:
                transcript_data = {
                    "utterances": transcript.utterances,
                    "total": len(transcript.utterances),
                }

        rubric_data = None
        if session.status == "completed":
            rubric_result = await db.execute(
                select(Rubric).where(Rubric.session_id == session.id)
            )
            rubric = rubric_result.scalar_one_or_none()
            if rubric:
                rubric_data = {"criteria": rubric.criteria}

        analysis_data = None
        if session.status == "completed":
            analysis_result = await db.execute(
                select(Analysis).where(Analysis.session_id == session.id)
            )
            analysis = analysis_result.scalar_one_or_none()
            if analysis:
                analysis_data = {
                    "scores": analysis.scores,
                    "strengths": analysis.strengths,
                    "improvements": analysis.improvements,
                    "key_moments": analysis.key_moments,
                }

        report_data = None
        if session.status == "completed":
            report_result = await db.execute(
                select(Report).where(Report.session_id == session.id)
            )
            report = report_result.scalar_one_or_none()
            if report:
                report_data = report.report_data

        all_sessions.append(
            {
                "session": session_data,
                "transcript": transcript_data,
                "rubric": rubric_data,
                "analysis": analysis_data,
                "report": report_data,
            }
        )

    return all_sessions


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return session status and metadata."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session: Session | None = result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    progress = get_progress(session.id) if session.status == "processing" else None

    meta_es = (session.metadata_es or {}) if hasattr(session, "metadata_es") else {}
    return {
        "session_id": session.id,
        "call_title": session.call_title,
        "call_title_es": meta_es.get("call_title"),
        "agent_name": session.agent_name,
        "agent_name_es": meta_es.get("agent_name"),
        "client_name": session.client_name,
        "client_name_es": meta_es.get("client_name"),
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
    session_result = await db.execute(select(Session).where(Session.id == session_id))
    session: Session | None = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

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

    transcript_result = await db.execute(
        select(Transcript).where(Transcript.session_id == session_id)
    )
    transcript: Transcript | None = transcript_result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found for this session",
        )

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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

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

    analysis_result = await db.execute(
        select(Analysis).where(Analysis.session_id == session_id)
    )
    analysis: Analysis | None = analysis_result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found for this session",
        )

    return {
        "session_id": session_id,
        "scores": analysis.scores,
        "strengths": analysis.strengths,
        "improvements": analysis.improvements,
        "key_moments": analysis.key_moments,
    }
