from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.models.session import Session
from app.models.report import Report
from app.models.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/sessions")
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session)
        .where(Session.user_id == current_user.id)
        .order_by(Session.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    sessions = result.scalars().all()

    # Batch-load reports for overall_score
    session_ids = [s.id for s in sessions]
    reports_by_session: dict[str, Report] = {}
    if session_ids:
        rr = await db.execute(select(Report).where(Report.session_id.in_(session_ids)))
        for rep in rr.scalars().all():
            reports_by_session[rep.session_id] = rep

    items = []
    for s in sessions:
        rep = reports_by_session.get(s.id)
        overall_score = None
        max_score = None
        if rep:
            summary = (rep.report_data or {}).get("summary", {})
            overall_score = summary.get("overall_score")
            max_score = summary.get("max_score")
        items.append({
            "session_id": s.id,
            "call_title": s.call_title,
            "agent_name": s.agent_name,
            "client_name": s.client_name,
            "call_date": s.call_date,
            "status": s.status,
            "created_at": s.created_at.isoformat(),
            "overall_score": overall_score,
            "max_score": max_score,
        })

    return {"sessions": items, "total": len(items)}


@router.get("/stats")
async def get_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # All sessions for this user
    result = await db.execute(
        select(Session).where(Session.user_id == current_user.id).order_by(Session.created_at.desc())
    )
    sessions = result.scalars().all()

    total = len(sessions)
    completed = [s for s in sessions if s.status == "completed"]
    completion_rate = round(len(completed) / total * 100, 1) if total > 0 else 0.0

    # Load reports for completed sessions
    completed_ids = [s.id for s in completed]
    reports_by_session: dict[str, Report] = {}
    if completed_ids:
        rr = await db.execute(select(Report).where(Report.session_id.in_(completed_ids)))
        for rep in rr.scalars().all():
            reports_by_session[rep.session_id] = rep

    # Compute avg score and trend
    scores_with_meta = []
    for s in completed:
        rep = reports_by_session.get(s.id)
        if rep:
            summary = (rep.report_data or {}).get("summary", {})
            sc = summary.get("overall_score")
            ms = summary.get("max_score")
            if sc is not None:
                scores_with_meta.append({
                    "session_id": s.id,
                    "call_title": s.call_title or "Untitled Call",
                    "call_date": s.call_date,
                    "created_at": s.created_at.isoformat(),
                    "overall_score": sc,
                    "max_score": ms,
                })

    avg_score = round(sum(x["overall_score"] for x in scores_with_meta) / len(scores_with_meta), 2) if scores_with_meta else None
    avg_max = round(sum(x["max_score"] for x in scores_with_meta if x["max_score"]) / len(scores_with_meta), 2) if scores_with_meta else None

    # Trend: last 10 scored sessions (oldest → newest for chart left-to-right)
    trend = list(reversed(scores_with_meta[:10]))

    return {
        "total_calls": total,
        "completed_calls": len(completed),
        "completion_rate": completion_rate,
        "avg_score": avg_score,
        "avg_max_score": avg_max,
        "score_trend": trend,
    }
