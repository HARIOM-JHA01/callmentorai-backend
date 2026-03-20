import json
import logging
from datetime import date
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


async def extract_call_metadata(
    utterances: list[dict],
    existing_title: str | None,
    existing_speaker1_name: str | None,
    existing_speaker2_name: str | None,
    existing_date: str | None,
) -> dict:
    """
    Use an LLM to infer call title, speaker identities, and date from the transcript.
    Always returns both English and Spanish versions of text fields.

    Returns:
        {
            "call_title": str,          # English
            "call_title_es": str,       # Spanish
            "speaker1_name": str | None,
            "speaker1_name_es": str | None,
            "speaker2_name": str | None,
            "speaker2_name_es": str | None,
            "call_date": str,           # YYYY-MM-DD
        }
    """
    today = date.today().isoformat()

    # Build the full transcript snippet
    snippet_lines = []
    for utt in utterances:
        speaker = utt.get("speaker", "?")
        text = utt.get("text", "")
        snippet_lines.append(f"{speaker}: {text}")
    transcript_snippet = "\n".join(snippet_lines)

    prompt = f"""You are analysing a phone/video call transcript. Extract the following metadata
and return it in BOTH English and Spanish. Respond ONLY with a valid JSON object and nothing else.

Fields to extract:
- "call_title": Short descriptive title in English (e.g. "Sales Discovery Call with Acme Corp"). Max 80 chars.
- "call_title_es": Same title translated to Spanish (e.g. "Llamada de Descubrimiento de Ventas con Acme Corp"). Max 80 chars.
- "speaker1_name": Real name or role of Speaker 1 in English (e.g. "John (Sales Rep)"). Null if not determinable.
- "speaker1_name_es": Same in Spanish (e.g. "Juan (Representante de Ventas)"). Null if not determinable.
- "speaker2_name": Real name or role of Speaker 2 in English. Null if not determinable.
- "speaker2_name_es": Same in Spanish. Null if not determinable.
- "call_date": Date of the call in YYYY-MM-DD format if mentioned, otherwise null.

Already known (do NOT override with null):
- call_title: {json.dumps(existing_title)}
- speaker1_name: {json.dumps(existing_speaker1_name)}
- speaker2_name: {json.dumps(existing_speaker2_name)}
- call_date: {json.dumps(existing_date)}

Transcript:
{transcript_snippet}

JSON response:"""

    try:
        client = _get_openai()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception as exc:
        logger.warning(f"[MetadataExtractor] LLM extraction failed: {exc}. Using defaults.")
        data = {}

    en_title = existing_title or data.get("call_title") or "Untitled Call"
    return {
        "call_title": en_title,
        "call_title_es": data.get("call_title_es") or en_title,
        "speaker1_name": existing_speaker1_name or data.get("speaker1_name"),
        "speaker1_name_es": data.get("speaker1_name_es") or existing_speaker1_name,
        "speaker2_name": existing_speaker2_name or data.get("speaker2_name"),
        "speaker2_name_es": data.get("speaker2_name_es") or existing_speaker2_name,
        "call_date": existing_date or data.get("call_date") or today,
    }
