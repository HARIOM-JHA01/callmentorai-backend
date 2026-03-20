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
    Use an LLM to infer call title, speaker identities, and date from the transcript
    when any of those fields are missing. Fields already provided by the user are
    returned unchanged.

    Returns:
        {
            "call_title": str,
            "speaker1_name": str | None,   # real name/role of Speaker 1
            "speaker2_name": str | None,   # real name/role of Speaker 2
            "call_date": str,              # YYYY-MM-DD, defaults to today
        }
    """
    # If everything is already provided, nothing to do
    today = date.today().isoformat()
    if existing_title and existing_speaker1_name and existing_speaker2_name and existing_date:
        return {
            "call_title": existing_title,
            "speaker1_name": existing_speaker1_name,
            "speaker2_name": existing_speaker2_name,
            "call_date": existing_date,
        }

    # Build the full transcript
    snippet_lines = []
    for utt in utterances:
        speaker = utt.get("speaker", "?")
        text = utt.get("text", "")
        snippet_lines.append(f"{speaker}: {text}")
    transcript_snippet = "\n".join(snippet_lines)

    prompt = f"""You are analysing a phone/video call transcript. Based on the dialogue below,
extract the following metadata. Respond ONLY with a valid JSON object and nothing else.

Fields to extract:
- "call_title": A short, descriptive title for this call (e.g. "Sales Discovery Call with Acme Corp", "Support Call – Billing Issue"). Max 80 chars.
- "speaker1_name": The real name or role of the person labelled "Speaker 1" if it can be inferred (e.g. "John (Sales Rep)", "Customer Support Agent"). Null if not determinable.
- "speaker2_name": The real name or role of the person labelled "Speaker 2" if it can be inferred. Null if not determinable.
- "call_date": The date of the call in YYYY-MM-DD format if mentioned in the transcript, otherwise null.

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
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception as exc:
        logger.warning(f"[MetadataExtractor] LLM extraction failed: {exc}. Using defaults.")
        data = {}

    return {
        "call_title": existing_title or data.get("call_title") or "Untitled Call",
        "speaker1_name": existing_speaker1_name or data.get("speaker1_name"),
        "speaker2_name": existing_speaker2_name or data.get("speaker2_name"),
        "call_date": existing_date or data.get("call_date") or today,
    }
