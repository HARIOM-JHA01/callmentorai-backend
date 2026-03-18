import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import AsyncSessionLocal
from app.models.session import Session, Transcript, Rubric, Analysis
from app.services.transcription import transcribe_audio
from app.services.rubric_parser import parse_rubric
from app.services.call_analyzer import analyze_call
from app.services.embeddings import build_embeddings
from app.services.report_generator import generate_report

logger = logging.getLogger(__name__)


async def _update_session_status(db: AsyncSession, session_id: str, status: str, error: str | None = None) -> None:
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if session:
        session.status = status
        session.error_message = error
        session.updated_at = datetime.now(timezone.utc)
        db.add(session)
        await db.commit()


async def run_analysis_pipeline(session_id: str) -> None:
    """
    Full analysis pipeline for a session. Runs as a background task.

    Steps:
      1. Transcribe audio
      2. Parse rubric
      3. Analyze call
      4. Build transcript embeddings
      5. Generate report
    """
    async with AsyncSessionLocal() as db:
        try:
            # ----------------------------------------------------------------
            # Mark session as processing
            # ----------------------------------------------------------------
            await _update_session_status(db, session_id, "processing")

            # Load session to get file paths
            session_result = await db.execute(select(Session).where(Session.id == session_id))
            session: Session | None = session_result.scalar_one_or_none()
            if not session:
                raise ValueError(f"Session {session_id} not found in database")

            audio_path = session.audio_path
            rubric_path = session.rubric_path

            # ----------------------------------------------------------------
            # Step 1: Transcription
            # ----------------------------------------------------------------
            logger.info(f"[Pipeline] Step 1: Transcribing audio for session {session_id}")
            utterances = await transcribe_audio(audio_path)

            transcript_row = Transcript(
                session_id=session_id,
                utterances=utterances,
            )
            db.add(transcript_row)
            await db.commit()
            logger.info(f"[Pipeline] Transcript saved: {len(utterances)} utterances")

            # ----------------------------------------------------------------
            # Step 2: Rubric parsing
            # ----------------------------------------------------------------
            logger.info(f"[Pipeline] Step 2: Parsing rubric for session {session_id}")
            rubric_data = await parse_rubric(rubric_path)

            rubric_row = Rubric(
                session_id=session_id,
                criteria=rubric_data,
            )
            db.add(rubric_row)
            await db.commit()
            logger.info(f"[Pipeline] Rubric saved: {len(rubric_data.get('criteria', []))} criteria")

            # ----------------------------------------------------------------
            # Step 3: Call analysis
            # ----------------------------------------------------------------
            logger.info(f"[Pipeline] Step 3: Analyzing call for session {session_id}")
            analysis_data = await analyze_call(utterances, rubric_data)

            analysis_row = Analysis(
                session_id=session_id,
                scores=analysis_data["scores"],
                strengths=analysis_data["strengths"],
                improvements=analysis_data["improvements"],
                key_moments=analysis_data["key_moments"],
            )
            db.add(analysis_row)
            await db.commit()
            logger.info("[Pipeline] Analysis saved")

            # ----------------------------------------------------------------
            # Step 4: Build embeddings
            # ----------------------------------------------------------------
            logger.info(f"[Pipeline] Step 4: Building embeddings for session {session_id}")
            await build_embeddings(session_id, utterances)
            logger.info("[Pipeline] Embeddings built")

            # ----------------------------------------------------------------
            # Step 5: Generate report
            # ----------------------------------------------------------------
            logger.info(f"[Pipeline] Step 5: Generating report for session {session_id}")
            await generate_report(session_id, db)
            logger.info("[Pipeline] Report generated")

            # ----------------------------------------------------------------
            # Mark session as completed
            # ----------------------------------------------------------------
            await _update_session_status(db, session_id, "completed")
            logger.info(f"[Pipeline] Session {session_id} completed successfully")

        except Exception as exc:
            logger.exception(f"[Pipeline] Error processing session {session_id}: {exc}")
            try:
                async with AsyncSessionLocal() as error_db:
                    await _update_session_status(error_db, session_id, "failed", str(exc))
            except Exception as inner_exc:
                logger.exception(f"[Pipeline] Failed to save error status for session {session_id}: {inner_exc}")
