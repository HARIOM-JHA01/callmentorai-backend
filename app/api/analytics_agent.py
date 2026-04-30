import json
import logging

from fastapi import APIRouter, Depends
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.user import User
from app.services.auth import get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/analytics", tags=["analytics"])

_openai_client: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


class AnalyticsContext(BaseModel):
    date_range: str
    active_filters: dict
    kpis: dict
    team_stats: list[dict]
    agent_top: list[dict]
    agent_bottom: list[dict]
    score_trend: list[dict]
    compliance_by_category: list[dict]


class AnalyticsChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    context: AnalyticsContext
    lang: str = "en"


class AnalyticsChatResponse(BaseModel):
    response: str


def _build_user_prompt(request: AnalyticsChatRequest) -> str:
    ctx = request.context
    return (
        "Dashboard analytics context (JSON):\n"
        f"date_range: {ctx.date_range}\n"
        f"active_filters: {json.dumps(ctx.active_filters, ensure_ascii=True)}\n\n"
        f"kpis:\n{json.dumps(ctx.kpis, ensure_ascii=True, indent=2)}\n\n"
        f"team_stats:\n{json.dumps(ctx.team_stats, ensure_ascii=True, indent=2)}\n\n"
        f"agent_top:\n{json.dumps(ctx.agent_top, ensure_ascii=True, indent=2)}\n\n"
        f"agent_bottom:\n{json.dumps(ctx.agent_bottom, ensure_ascii=True, indent=2)}\n\n"
        f"score_trend:\n{json.dumps(ctx.score_trend, ensure_ascii=True, indent=2)}\n\n"
        f"compliance_by_category:\n{json.dumps(ctx.compliance_by_category, ensure_ascii=True, indent=2)}\n\n"
        f"Question: {request.question.strip()}"
    )


@router.post("/chat", response_model=AnalyticsChatResponse)
async def analytics_chat(
    request: AnalyticsChatRequest,
    current_user: User = Depends(get_current_user),
):
    lang = request.lang
    system_language = (
        "Respond entirely in Spanish." if lang == "es" else "Respond in English."
    )

    system_prompt = (
        "You are an AI analytics advisor for a call center operations dashboard. "
        f"{system_language} "
        "Answer questions concisely — keep total responses under 200 words. "
        "Structure: one short opening sentence, then bullet points (use '- ' prefix) for specific findings. "
        "Use **bold** for metric names, team names, or agent names. "
        "Always cite specific numbers from the data provided (scores, counts, percentages). "
        "Be direct and actionable — no filler phrases. "
        "When recommending action, be specific: name the team or agent."
    )

    user_prompt = _build_user_prompt(request)

    client = _get_openai()
    logger.info(
        "Analytics question from user %s: %s...",
        current_user.id,
        request.question[:80],
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    answer = (response.choices[0].message.content or "").strip()
    return AnalyticsChatResponse(response=answer)
