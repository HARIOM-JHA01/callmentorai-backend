import logging
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from openai import AsyncOpenAI

from app.config import get_settings
from app.database.connection import get_db
from app.models.session import Session, Rubric, Analysis
from app.services.embeddings import search_transcript

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/coach", tags=["coach"])

_openai_client: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="UUID of the analysis session")
    question: str = Field(..., min_length=1, max_length=2000, description="Coaching question to ask")


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
async def coach_chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask the AI coach a question about a specific call session.
    Uses FAISS-based transcript retrieval for grounded, specific answers.
    """
    # Validate session exists and is complete
    session_result = await db.execute(select(Session).where(Session.id == request.session_id))
    session: Session | None = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.status in ("pending", "processing"):
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Session is still {session.status} — coaching is not yet available",
        )
    if session.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Session analysis failed: {session.error_message}",
        )

    # Retrieve relevant transcript chunks via FAISS semantic search
    relevant_chunks = await search_transcript(request.session_id, request.question, top_k=5)

    # Load rubric and analysis from DB
    rubric_result = await db.execute(select(Rubric).where(Rubric.session_id == request.session_id))
    rubric_row: Rubric | None = rubric_result.scalar_one_or_none()
    rubric_data = rubric_row.criteria if rubric_row else {"criteria": []}

    analysis_result = await db.execute(select(Analysis).where(Analysis.session_id == request.session_id))
    analysis_row: Analysis | None = analysis_result.scalar_one_or_none()
    analysis_data = (
        {
            "scores": analysis_row.scores,
            "strengths": analysis_row.strengths,
            "improvements": analysis_row.improvements,
            "key_moments": analysis_row.key_moments,
        }
        if analysis_row
        else {}
    )

    # Build the context block
    transcript_context = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else "No relevant transcript found."
    rubric_summary = _format_rubric_for_prompt(rubric_data)
    analysis_summary = _format_analysis_for_prompt(analysis_data)

    system_prompt = (
        "You are an AI call coach. Your role is to help agents improve their customer service and sales skills. "
        "Use the provided transcript excerpts and rubric evaluation to answer coaching questions. "
        "Be specific — cite examples from the transcript using speaker labels and timestamps where available. "
        "Provide actionable, constructive advice."
    )

    user_prompt = f"""Session context:
Agent: {session.agent_name or 'Unknown'}
Client: {session.client_name or 'Unknown'}
Call Title: {session.call_title or 'Untitled'}

--- RELEVANT TRANSCRIPT EXCERPTS ---
{transcript_context}

--- RUBRIC CRITERIA ---
{rubric_summary}

--- ANALYSIS RESULTS ---
{analysis_summary}

--- COACHING QUESTION ---
{request.question}

Please answer the coaching question based on the above context. Be specific and cite evidence from the transcript."""

    client = _get_openai()
    logger.info(f"Coaching question for session {request.session_id}: {request.question[:80]}...")

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    return ChatResponse(response=answer)


def _format_rubric_for_prompt(rubric: dict) -> str:
    criteria = rubric.get("criteria", [])
    if not criteria:
        return "No rubric criteria available."
    lines = []
    for c in criteria:
        lines.append(f"- {c['name']} (max {c.get('max_score', 10)}): {c.get('description', '')}")
    return "\n".join(lines)


def _format_analysis_for_prompt(analysis: dict) -> str:
    if not analysis:
        return "No analysis data available."
    lines = []

    scores = analysis.get("scores", [])
    if scores:
        lines.append("Scores:")
        for s in scores:
            lines.append(f"  {s['category']}: {s['score']}/{s['max_score']} — {s.get('reason', '')}")

    strengths = analysis.get("strengths", [])
    if strengths:
        lines.append("Strengths:")
        for item in strengths:
            lines.append(f"  • {item}")

    improvements = analysis.get("improvements", [])
    if improvements:
        lines.append("Areas for Improvement:")
        for item in improvements:
            lines.append(f"  • {item}")

    key_moments = analysis.get("key_moments", [])
    if key_moments:
        lines.append("Key Moments:")
        for km in key_moments:
            lines.append(f"  [{km.get('timestamp', '')}] {km.get('description', '')}")

    return "\n".join(lines) if lines else "No analysis data available."
