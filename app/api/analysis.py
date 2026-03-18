import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import get_db
from app.models.session import Session
from app.models.report import Report
from app.services.report_generator import generate_pdf_bytes, generate_docx_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/session", tags=["reports"])


@router.get("/{session_id}/report")
async def get_report(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the full coaching report for a session."""
    session_result = await db.execute(select(Session).where(Session.id == session_id))
    session: Session | None = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.status in ("pending", "processing"):
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Session is {session.status} — report not yet available",
        )
    if session.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Session analysis failed: {session.error_message}",
        )

    report_result = await db.execute(select(Report).where(Report.session_id == session_id))
    report: Report | None = report_result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found for this session")

    return {"session_id": session_id, "report": report.report_data}


@router.get("/{session_id}/report/pdf", response_class=Response)
async def download_report_pdf(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Generate and return the coaching report as a PDF file."""
    report_data = await _load_report_data(session_id, db)
    pdf_bytes = generate_pdf_bytes(report_data)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{session_id}.pdf"'},
    )


@router.get("/{session_id}/report/docx", response_class=Response)
async def download_report_docx(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Generate and return the coaching report as a DOCX file."""
    report_data = await _load_report_data(session_id, db)
    docx_bytes = generate_docx_bytes(report_data)
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="report_{session_id}.docx"'},
    )


async def _load_report_data(session_id: str, db: AsyncSession) -> dict:
    """Shared helper to load and validate report data from DB."""
    session_result = await db.execute(select(Session).where(Session.id == session_id))
    session: Session | None = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session status is '{session.status}' — only 'completed' sessions have downloadable reports",
        )

    report_result = await db.execute(select(Report).where(Report.session_id == session_id))
    report: Report | None = report_result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found for this session")

    return report.report_data
