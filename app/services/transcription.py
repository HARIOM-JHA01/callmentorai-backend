import asyncio
import logging
import os
from functools import partial

from elevenlabs.client import ElevenLabs
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between retries


async def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Transcribe an audio file using ElevenLabs Scribe with speaker diarization.

    Returns a list of utterance dicts:
        [{"speaker": "Speaker 1"|"Speaker 2", "text": str, "start": float, "end": float}]
    """
    client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    file_mb = os.path.getsize(audio_path) / 1_048_576
    logger.info(f"[Transcription] Reading {audio_path} ({file_mb:.1f} MB) …")

    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    logger.info(f"[Transcription] Sending {file_mb:.1f} MB to ElevenLabs Scribe …")

    last_exc: Exception | None = None
    response = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if attempt > 1:
                logger.info(f"[Transcription] Retry {attempt}/{_MAX_RETRIES} …")
                await asyncio.sleep(_RETRY_DELAY)

            # ElevenLabs SDK is sync — run in a thread to avoid blocking the event loop
            fn = partial(
                client.speech_to_text.convert,
                file=(os.path.basename(audio_path), audio_bytes),
                model_id="scribe_v1",
                diarize=True,
            )
            response = await asyncio.get_event_loop().run_in_executor(None, fn)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"[Transcription] Attempt {attempt} failed: {type(exc).__name__}: {exc}"
            )
            if attempt == _MAX_RETRIES:
                raise

    if response is None:
        raise last_exc  # type: ignore[misc]

    logger.info("[Transcription] ElevenLabs processing complete.")

    result = _build_utterances(response)

    logger.info(f"[Transcription] Complete — {len(result)} utterances\n")
    transcript_lines = "\n".join(
        f"  [{u['start']:.1f}s] {u['speaker']}: {u['text']}" for u in result
    )
    logger.info(f"[Transcription] Full transcript:\n{transcript_lines}\n")
    return result


def _build_utterances(response) -> list[dict]:
    """
    Group ElevenLabs word-level results by consecutive speaker into utterances.
    Dynamically remaps whatever speaker IDs ElevenLabs returns (which may be
    non-consecutive, e.g. speaker_0, speaker_2, speaker_4) to sequential labels
    (Speaker 1, Speaker 2, …) in order of first appearance.
    """
    words = getattr(response, "words", None) or []
    if not words:
        return []

    # Dynamic remap: first seen speaker_id → "Speaker 1", second → "Speaker 2", …
    speaker_remap: dict[str, str] = {}

    def get_label(sp: str | None) -> str:
        key = sp or "unknown"
        if key not in speaker_remap:
            speaker_remap[key] = f"Speaker {len(speaker_remap) + 1}"
        return speaker_remap[key]

    utterances: list[dict] = []
    current_speaker: str | None = None
    current_words: list[str] = []
    current_start = 0.0
    current_end = 0.0

    for w in words:
        word_type = getattr(w, "type", "word")
        if word_type != "word":
            continue

        sp = getattr(w, "speaker_id", None)
        text = getattr(w, "text", "").strip()
        start = float(getattr(w, "start", 0) or 0)
        end = float(getattr(w, "end", 0) or 0)

        if not text:
            continue

        if sp != current_speaker:
            if current_words and current_speaker is not None:
                utterances.append(
                    {
                        "speaker": get_label(current_speaker),
                        "text": " ".join(current_words),
                        "start": round(current_start, 3),
                        "end": round(current_end, 3),
                    }
                )
            current_speaker = sp
            current_words = [text]
            current_start = start
            current_end = end
        else:
            current_words.append(text)
            current_end = end

    if current_words and current_speaker is not None:
        utterances.append(
            {
                "speaker": get_label(current_speaker),
                "text": " ".join(current_words),
                "start": round(current_start, 3),
                "end": round(current_end, 3),
            }
        )

    return utterances
