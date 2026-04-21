from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.models.session import Analysis, Session
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
            "team": s.team,
            "supervisor": s.supervisor,
            "campaign": s.campaign,
            "queue": s.queue,
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


def _parse_any_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        if "T" in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            parsed = datetime.fromisoformat(f"{text}T00:00:00")
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _in_date_range(session: Session, date_range: str) -> bool:
    now = datetime.now(timezone.utc)
    if date_range == "today":
        min_dt = now - timedelta(days=1)
    elif date_range == "7d":
        min_dt = now - timedelta(days=7)
    elif date_range == "30d":
        min_dt = now - timedelta(days=30)
    else:
        min_dt = now - timedelta(days=90)

    session_dt = _parse_any_datetime(session.call_date) or session.created_at
    return session_dt >= min_dt


@router.get("/enterprise/compliance")
async def get_enterprise_compliance(
    date_range: str = Query("30d", pattern="^(today|7d|30d|90d)$"),
    team: str | None = None,
    supervisor: str | None = None,
    campaign: str | None = None,
    queue: str | None = None,
    status: str | None = Query(None, pattern="^(completed|pending|processing|failed)$"),
    agent_search: str | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Compliance is computed only from completed sessions with analyses.
    if status and status != "completed":
        return {"categories": []}

    stmt = (
        select(Session, Analysis)
        .join(Analysis, Analysis.session_id == Session.id)
        .where(Session.user_id == current_user.id)
        .where(Session.status == "completed")
    )
    if team:
        stmt = stmt.where(Session.team == team)
    if supervisor:
        stmt = stmt.where(Session.supervisor == supervisor)
    if campaign:
        stmt = stmt.where(Session.campaign == campaign)
    if queue:
        stmt = stmt.where(Session.queue == queue)
    if agent_search:
        stmt = stmt.where(Session.agent_name.ilike(f"%{agent_search}%"))

    result = await db.execute(stmt)
    rows = result.all()

    buckets: dict[str, list[float]] = {}
    for session, analysis in rows:
        if not _in_date_range(session, date_range):
            continue
        for item in analysis.scores or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("category") or "").strip()
            score = item.get("score")
            max_score = item.get("max_score")
            if not name or score is None or not max_score:
                continue
            try:
                pct = (float(score) / float(max_score)) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            buckets.setdefault(name, []).append(pct)

    categories = [
        {"name": name, "score": round(sum(values) / len(values))}
        for name, values in buckets.items()
        if values
    ]
    categories.sort(key=lambda x: x["score"], reverse=True)
    return {"categories": categories}


@router.get("/enterprise/filters")
async def get_enterprise_filters(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    teams_result = await db.execute(
        select(func.distinct(Session.team))
        .where(Session.user_id == current_user.id)
        .where(Session.team.is_not(None))
    )
    supervisors_result = await db.execute(
        select(func.distinct(Session.supervisor))
        .where(Session.user_id == current_user.id)
        .where(Session.supervisor.is_not(None))
    )
    campaigns_result = await db.execute(
        select(func.distinct(Session.campaign))
        .where(Session.user_id == current_user.id)
        .where(Session.campaign.is_not(None))
    )
    queues_result = await db.execute(
        select(func.distinct(Session.queue))
        .where(Session.user_id == current_user.id)
        .where(Session.queue.is_not(None))
    )

    teams = sorted(x.strip() for x in teams_result.scalars().all() if isinstance(x, str) and x.strip())
    supervisors = sorted(x.strip() for x in supervisors_result.scalars().all() if isinstance(x, str) and x.strip())
    campaigns = sorted(x.strip() for x in campaigns_result.scalars().all() if isinstance(x, str) and x.strip())
    queues = sorted(x.strip() for x in queues_result.scalars().all() if isinstance(x, str) and x.strip())

    return {
        "teams": teams,
        "supervisors": supervisors,
        "campaigns": campaigns,
        "queues": queues,
    }
