import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.models.session import Session, Transcript, Rubric, Analysis
from app.models.report import Report

logger = logging.getLogger(__name__)
settings = get_settings()

_openai_client: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


async def _get_recommendations(analysis: dict, rubric: dict) -> list[str]:
    """Use GPT-4o to generate actionable coaching recommendations."""
    client = _get_openai()

    system_prompt = (
        "You are an expert call center coach. "
        "Based on the analysis results, generate 3-5 specific, actionable coaching recommendations. "
        "Return JSON only."
    )
    user_prompt = f"""
Analysis:
{json.dumps(analysis, indent=2)}

Rubric criteria:
{json.dumps(rubric.get("criteria", []), indent=2)}

Return a JSON object:
{{
  "recommendations": [
    "Actionable recommendation 1",
    "Actionable recommendation 2"
  ]
}}
"""
    response = await client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    parsed = json.loads(response.choices[0].message.content)
    return [str(r) for r in parsed.get("recommendations", [])]


async def generate_report(session_id: str, db: AsyncSession) -> dict:
    """
    Compile a full coaching report from all stored data for the session.

    Returns a report dict and upserts it into the DB.
    """
    # Load session
    session_result = await db.execute(select(Session).where(Session.id == session_id))
    session: Session = session_result.scalar_one_or_none()
    if not session:
        raise ValueError(f"Session {session_id} not found")

    # Load transcript
    transcript_result = await db.execute(select(Transcript).where(Transcript.session_id == session_id))
    transcript_row: Transcript | None = transcript_result.scalar_one_or_none()
    utterances = transcript_row.utterances if transcript_row else []

    # Load rubric
    rubric_result = await db.execute(select(Rubric).where(Rubric.session_id == session_id))
    rubric_row: Rubric | None = rubric_result.scalar_one_or_none()
    rubric = rubric_row.criteria if rubric_row else {"criteria": []}

    # Load analysis
    analysis_result = await db.execute(select(Analysis).where(Analysis.session_id == session_id))
    analysis_row: Analysis | None = analysis_result.scalar_one_or_none()
    if not analysis_row:
        raise ValueError(f"Analysis for session {session_id} not found")

    analysis_data = {
        "scores": analysis_row.scores,
        "strengths": analysis_row.strengths,
        "improvements": analysis_row.improvements,
        "key_moments": analysis_row.key_moments,
    }

    # Compute overall score
    scores = analysis_row.scores or []
    total_score = sum(s.get("score", 0) for s in scores)
    total_max = sum(s.get("max_score", 10) for s in scores)

    # Generate recommendations
    recommendations = await _get_recommendations(analysis_data, rubric)

    report_data: dict[str, Any] = {
        "summary": {
            "agent_name": session.agent_name or "Unknown Agent",
            "client_name": session.client_name or "Unknown Client",
            "call_title": session.call_title or "Untitled Call",
            "call_date": session.call_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "overall_score": round(total_score, 2),
            "max_score": total_max,
            "percentage": round((total_score / total_max * 100) if total_max > 0 else 0, 1),
        },
        "rubric_scores": scores,
        "strengths": analysis_row.strengths,
        "areas_for_improvement": analysis_row.improvements,
        "key_moments": analysis_row.key_moments,
        "recommendations": recommendations,
        "transcript_summary": {
            "total_utterances": len(utterances),
            "agent_turns": sum(1 for u in utterances if u.get("speaker") == "Agent"),
            "customer_turns": sum(1 for u in utterances if u.get("speaker") == "Customer"),
        },
    }

    # Upsert report in DB
    report_result = await db.execute(select(Report).where(Report.session_id == session_id))
    existing_report: Report | None = report_result.scalar_one_or_none()
    if existing_report:
        existing_report.report_data = report_data
        db.add(existing_report)
    else:
        new_report = Report(session_id=session_id, report_data=report_data)
        db.add(new_report)

    await db.commit()
    logger.info(f"Report generated and saved for session {session_id}")
    return report_data


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def generate_pdf_bytes(report_data: dict) -> bytes:
    """Generate a PDF report from report_data and return as bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    heading1 = ParagraphStyle("Heading1Custom", parent=styles["Heading1"], fontSize=18, spaceAfter=6)
    heading2 = ParagraphStyle("Heading2Custom", parent=styles["Heading2"], fontSize=14, spaceAfter=4)
    normal = styles["Normal"]

    story = []

    # Title
    story.append(Paragraph("CallMentor AI — Coaching Report", heading1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.4 * cm))

    # Summary
    summary = report_data.get("summary", {})
    story.append(Paragraph("Call Summary", heading2))
    summary_rows = [
        ["Agent", summary.get("agent_name", "")],
        ["Client", summary.get("client_name", "")],
        ["Call Title", summary.get("call_title", "")],
        ["Call Date", summary.get("call_date", "")],
        ["Overall Score", f"{summary.get('overall_score', 0)} / {summary.get('max_score', 0)} "
                          f"({summary.get('percentage', 0)}%)"],
    ]
    tbl = Table(summary_rows, colWidths=[4 * cm, 12 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.6 * cm))

    # Rubric Scores
    scores = report_data.get("rubric_scores", [])
    if scores:
        story.append(Paragraph("Evaluation Scores", heading2))
        score_rows = [["Category", "Score", "Max", "Reason"]]
        for s in scores:
            score_rows.append([
                s.get("category", ""),
                str(s.get("score", 0)),
                str(s.get("max_score", 0)),
                Paragraph(s.get("reason", ""), normal),
            ])
        score_tbl = Table(score_rows, colWidths=[4 * cm, 1.5 * cm, 1.5 * cm, 10 * cm])
        score_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(score_tbl)
        story.append(Spacer(1, 0.6 * cm))

    # Strengths
    strengths = report_data.get("strengths", [])
    if strengths:
        story.append(Paragraph("Strengths", heading2))
        for item in strengths:
            story.append(Paragraph(f"• {item}", normal))
        story.append(Spacer(1, 0.4 * cm))

    # Areas for Improvement
    improvements = report_data.get("areas_for_improvement", [])
    if improvements:
        story.append(Paragraph("Areas for Improvement", heading2))
        for item in improvements:
            story.append(Paragraph(f"• {item}", normal))
        story.append(Spacer(1, 0.4 * cm))

    # Key Moments
    key_moments = report_data.get("key_moments", [])
    if key_moments:
        story.append(Paragraph("Key Moments", heading2))
        for km in key_moments:
            story.append(Paragraph(f"<b>[{km.get('timestamp', '')}]</b> {km.get('description', '')}", normal))
        story.append(Spacer(1, 0.4 * cm))

    # Recommendations
    recommendations = report_data.get("recommendations", [])
    if recommendations:
        story.append(Paragraph("Coaching Recommendations", heading2))
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", normal))
        story.append(Spacer(1, 0.4 * cm))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_docx_bytes(report_data: dict) -> bytes:
    """Generate a DOCX report from report_data and return as bytes."""
    from docx import Document
    from docx.shared import Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Title
    title = doc.add_heading("CallMentor AI — Coaching Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Summary
    summary = report_data.get("summary", {})
    doc.add_heading("Call Summary", level=1)
    tbl = doc.add_table(rows=5, cols=2)
    tbl.style = "Table Grid"
    rows_data = [
        ("Agent", summary.get("agent_name", "")),
        ("Client", summary.get("client_name", "")),
        ("Call Title", summary.get("call_title", "")),
        ("Call Date", summary.get("call_date", "")),
        (
            "Overall Score",
            f"{summary.get('overall_score', 0)} / {summary.get('max_score', 0)} "
            f"({summary.get('percentage', 0)}%)",
        ),
    ]
    for i, (label, value) in enumerate(rows_data):
        tbl.rows[i].cells[0].text = label
        tbl.rows[i].cells[1].text = str(value)

    doc.add_paragraph()

    # Scores
    scores = report_data.get("rubric_scores", [])
    if scores:
        doc.add_heading("Evaluation Scores", level=1)
        score_tbl = doc.add_table(rows=1 + len(scores), cols=4)
        score_tbl.style = "Table Grid"
        headers = ["Category", "Score", "Max", "Reason"]
        for j, h in enumerate(headers):
            cell = score_tbl.rows[0].cells[j]
            cell.text = h
            run = cell.paragraphs[0].runs[0]
            run.bold = True
        for i, s in enumerate(scores, 1):
            row = score_tbl.rows[i]
            row.cells[0].text = s.get("category", "")
            row.cells[1].text = str(s.get("score", 0))
            row.cells[2].text = str(s.get("max_score", 0))
            row.cells[3].text = s.get("reason", "")
        doc.add_paragraph()

    # Strengths
    strengths = report_data.get("strengths", [])
    if strengths:
        doc.add_heading("Strengths", level=1)
        for item in strengths:
            doc.add_paragraph(item, style="List Bullet")
        doc.add_paragraph()

    # Improvements
    improvements = report_data.get("areas_for_improvement", [])
    if improvements:
        doc.add_heading("Areas for Improvement", level=1)
        for item in improvements:
            doc.add_paragraph(item, style="List Bullet")
        doc.add_paragraph()

    # Key Moments
    key_moments = report_data.get("key_moments", [])
    if key_moments:
        doc.add_heading("Key Moments", level=1)
        for km in key_moments:
            doc.add_paragraph(f"[{km.get('timestamp', '')}] {km.get('description', '')}", style="List Bullet")
        doc.add_paragraph()

    # Recommendations
    recommendations = report_data.get("recommendations", [])
    if recommendations:
        doc.add_heading("Coaching Recommendations", level=1)
        for i, rec in enumerate(recommendations, 1):
            doc.add_paragraph(f"{i}. {rec}")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()
