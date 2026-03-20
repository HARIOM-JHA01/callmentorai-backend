import json
import logging
from openai import AsyncOpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_openai_client: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _format_transcript(transcript: list[dict]) -> str:
    """Convert structured utterances into a labeled dialogue string."""
    lines = []
    for utt in transcript:
        speaker = utt.get("speaker", "Unknown")
        text = utt.get("text", "")
        start = utt.get("start", 0.0)
        minutes = int(start // 60)
        seconds = int(start % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        lines.append(f"{timestamp} {speaker}: {text}")
    return "\n".join(lines)


def _format_rubric(rubric: dict) -> str:
    """Format rubric criteria as a readable list for the prompt."""
    lines = []
    for c in rubric.get("criteria", []):
        lines.append(
            f"- {c['name']} (max score: {c['max_score']}): {c.get('description', '')}"
        )
    return "\n".join(lines)


async def analyze_call(transcript: list[dict], rubric: dict) -> dict:
    """
    Evaluate a call transcript against rubric criteria using GPT-4o.
    Always returns results in BOTH English and Spanish.

    Returns:
        {
            "scores": [{"category": str, "category_es": str, "score": int, "max_score": int,
                        "reason": str, "reason_es": str}],
            "strengths": {"en": [str], "es": [str]},
            "improvements": {"en": [str], "es": [str]},
            "key_moments": [{"timestamp": str, "description": str, "description_es": str}]
        }
    """
    transcript_text = _format_transcript(transcript)
    rubric_text = _format_rubric(rubric)

    system_prompt = (
        "You are an expert call center quality assurance analyst and coach. "
        "Analyze the provided call transcript against the given evaluation rubric. "
        "Be objective, specific, and cite exact moments from the transcript where relevant. "
        "Always return valid JSON only — no markdown, no extra commentary."
    )

    user_prompt = f"""Analyze the following call transcript against the provided rubric criteria.

RUBRIC CRITERIA:
{rubric_text}

CALL TRANSCRIPT:
{transcript_text}

Return ONLY a JSON object with the analysis written in BOTH English ("en") and Spanish ("es"):
{{
  "en": {{
    "scores": [
      {{
        "category": "Category Name in English",
        "score": 8,
        "max_score": 10,
        "reason": "Specific reason based on the transcript in English"
      }}
    ],
    "strengths": ["Specific strength in English"],
    "improvements": ["Specific improvement area in English"],
    "key_moments": [
      {{
        "timestamp": "MM:SS",
        "description": "Description of the key moment in English"
      }}
    ]
  }},
  "es": {{
    "scores": [
      {{
        "category": "Nombre de la categoría en español",
        "score": 8,
        "max_score": 10,
        "reason": "Razón específica basada en la transcripción en español"
      }}
    ],
    "strengths": ["Fortaleza específica en español"],
    "improvements": ["Área de mejora específica en español"],
    "key_moments": [
      {{
        "timestamp": "MM:SS",
        "description": "Descripción del momento clave en español"
      }}
    ]
  }}
}}

Ensure:
- Every rubric criterion has a corresponding score entry in BOTH languages
- Scores are identical between "en" and "es" (same integer values, only text differs)
- "en" and "es" arrays must be the same length and in the same order
- Key moments cite the most important 3-5 moments from the transcript
"""

    client = _get_openai()
    logger.info("Sending transcript and rubric to GPT-4o for bilingual call analysis")

    response = await client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        seed=42,
        max_tokens=8192,
    )

    raw_json = response.choices[0].message.content
    parsed = json.loads(raw_json)

    en = parsed.get("en", {})
    es = parsed.get("es", {})

    en_scores = en.get("scores", [])
    es_scores = es.get("scores", [])
    scores = []
    for i, s in enumerate(en_scores):
        es_s = es_scores[i] if i < len(es_scores) else {}
        scores.append(
            {
                "category": str(s.get("category", "")),
                "category_es": str(es_s.get("category", s.get("category", ""))),
                "score": int(s.get("score", 0)),
                "max_score": int(s.get("max_score", 10)),
                "reason": str(s.get("reason", "")),
                "reason_es": str(es_s.get("reason", s.get("reason", ""))),
            }
        )

    en_km = en.get("key_moments", [])
    es_km = es.get("key_moments", [])
    key_moments = []
    for i, km in enumerate(en_km):
        es_km_item = es_km[i] if i < len(es_km) else {}
        key_moments.append(
            {
                "timestamp": str(km.get("timestamp", "")),
                "description": str(km.get("description", "")),
                "description_es": str(es_km_item.get("description", km.get("description", ""))),
            }
        )

    result = {
        "scores": scores,
        "strengths": {
            "en": [str(x) for x in en.get("strengths", [])],
            "es": [str(x) for x in es.get("strengths", [])],
        },
        "improvements": {
            "en": [str(x) for x in en.get("improvements", [])],
            "es": [str(x) for x in es.get("improvements", [])],
        },
        "key_moments": key_moments,
    }
    logger.info(
        f"Bilingual call analysis complete: {len(scores)} scored categories, "
        f"{len(result['strengths']['en'])} strengths, {len(result['improvements']['en'])} improvements"
    )
    return result
