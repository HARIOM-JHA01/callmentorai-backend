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

    Returns:
        {
            "scores": [{"category": str, "score": int, "max_score": int, "reason": str}],
            "strengths": [str],
            "improvements": [str],
            "key_moments": [{"timestamp": str, "description": str}]
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

Return ONLY a JSON object in this exact format:
{{
  "scores": [
    {{
      "category": "Category Name",
      "score": 8,
      "max_score": 10,
      "reason": "Specific reason based on the transcript"
    }}
  ],
  "strengths": [
    "Specific strength observed in the call"
  ],
  "improvements": [
    "Specific area for improvement with actionable advice"
  ],
  "key_moments": [
    {{
      "timestamp": "MM:SS",
      "description": "Description of the key moment"
    }}
  ]
}}

Ensure:
- Every rubric criterion has a corresponding score entry
- Scores are integers within [0, max_score]
- Strengths and improvements are specific to this call
- Key moments cite the most important 3-5 moments from the transcript
"""

    client = _get_openai()
    logger.info("Sending transcript and rubric to GPT-4o for call analysis")

    response = await client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )

    raw_json = response.choices[0].message.content
    parsed = json.loads(raw_json)

    # Validate and normalise
    scores = []
    for s in parsed.get("scores", []):
        scores.append(
            {
                "category": str(s.get("category", "")),
                "score": int(s.get("score", 0)),
                "max_score": int(s.get("max_score", 10)),
                "reason": str(s.get("reason", "")),
            }
        )

    strengths = [str(x) for x in parsed.get("strengths", [])]
    improvements = [str(x) for x in parsed.get("improvements", [])]
    key_moments = []
    for km in parsed.get("key_moments", []):
        key_moments.append(
            {
                "timestamp": str(km.get("timestamp", "")),
                "description": str(km.get("description", "")),
            }
        )

    result = {
        "scores": scores,
        "strengths": strengths,
        "improvements": improvements,
        "key_moments": key_moments,
    }
    logger.info(
        f"Call analysis complete: {len(scores)} scored categories, "
        f"{len(strengths)} strengths, {len(improvements)} improvements"
    )
    return result
