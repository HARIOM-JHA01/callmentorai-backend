import json
import logging
from pypdf import PdfReader
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


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file using pypdf."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


async def parse_rubric(pdf_path: str) -> dict:
    """
    Parse a rubric PDF and return a structured criteria dict.

    Returns:
        {
            "criteria": [
                {"name": str, "max_score": int, "description": str},
                ...
            ]
        }
    """
    logger.info(f"Extracting text from rubric PDF: {pdf_path}")
    raw_text = _extract_pdf_text(pdf_path)

    if not raw_text.strip():
        logger.warning("PDF text extraction returned empty content. Using minimal rubric.")
        return {
            "criteria": [
                {"name": "Overall Performance", "max_score": 10, "description": "General agent performance"}
            ]
        }

    system_prompt = (
        "You are an expert in call center quality assurance. "
        "Your task is to read the provided evaluation rubric and extract all evaluation criteria "
        "in a structured JSON format. Each criterion must have a name, a maximum score (integer), "
        "and a brief description. If no max_score is specified for a criterion, default to 10. "
        "Always return valid JSON only — no markdown, no extra commentary."
    )

    user_prompt = f"""Extract the evaluation criteria from the following rubric text.

Return ONLY a JSON object in this exact format:
{{
  "criteria": [
    {{
      "name": "Criterion Name",
      "max_score": 10,
      "description": "Brief description of what is being evaluated"
    }}
  ]
}}

Rubric text:
---
{raw_text}
---"""

    client = _get_openai()
    logger.info("Sending rubric text to GPT-4o for structured parsing")

    response = await client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw_json = response.choices[0].message.content
    parsed = json.loads(raw_json)

    # Validate structure
    if "criteria" not in parsed or not isinstance(parsed["criteria"], list):
        logger.error(f"Unexpected rubric parse result: {raw_json}")
        raise ValueError("LLM did not return a valid rubric structure with 'criteria' list.")

    # Ensure each criterion has required fields and correct types
    clean_criteria = []
    for c in parsed["criteria"]:
        clean_criteria.append(
            {
                "name": str(c.get("name", "Unnamed")),
                "max_score": int(c.get("max_score", 10)),
                "description": str(c.get("description", "")),
            }
        )

    result = {"criteria": clean_criteria}
    logger.info(f"Rubric parsed: {len(clean_criteria)} criteria found")
    return result
