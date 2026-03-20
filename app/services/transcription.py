import asyncio
import logging
import os

from deepgram import AsyncDeepgramClient
from deepgram.core.request_options import RequestOptions
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SPEAKER_MAP = {0: "Speaker 1", 1: "Speaker 2"}

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between retries


async def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Transcribe an audio file using Deepgram with speaker diarization.

    Returns a list of utterance dicts:
        [{"speaker": "Speaker 1"|"Speaker 2", "text": str, "start": float, "end": float}]
    """
    deepgram = AsyncDeepgramClient(api_key=settings.DEEPGRAM_API_KEY)

    file_mb = os.path.getsize(audio_path) / 1_048_576
    logger.info(f"[Transcription] Reading {audio_path} ({file_mb:.1f} MB) into memory …")

    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    logger.info(f"[Transcription] Sending {file_mb:.1f} MB to Deepgram …")

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if attempt > 1:
                logger.info(f"[Transcription] Retry {attempt}/{_MAX_RETRIES} …")
                await asyncio.sleep(_RETRY_DELAY)
            response = await deepgram.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model="nova-3",
                diarize=True,
                punctuate=True,
                utterances=True,
                filler_words=True,
                detect_language=True,
                request_options=RequestOptions(timeout_in_seconds=600),
            )
            break
        except Exception as exc:
            last_exc = exc
            logger.warning(f"[Transcription] Attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt == _MAX_RETRIES:
                raise
    else:
        raise last_exc  # type: ignore[misc]

    logger.info("[Transcription] Deepgram processing complete.")

    utterances_raw = response.results.utterances
    if not utterances_raw:
        # Fallback: try to build utterances from words if diarization data exists
        logger.warning(
            "No utterances returned from Deepgram; attempting word-level fallback."
        )
        return _build_utterances_from_words(response)

    result = []
    for utt in utterances_raw:
        speaker_int = utt.speaker if utt.speaker is not None else 0
        speaker_label = SPEAKER_MAP.get(speaker_int, f"Speaker {speaker_int}")
        result.append(
            {
                "speaker": speaker_label,
                "text": utt.transcript.strip(),
                "start": round(float(utt.start), 3),
                "end": round(float(utt.end), 3),
            }
        )

    logger.info(f"[Transcription] Complete — {len(result)} utterances\n")
    transcript_lines = "\n".join(
        f"  [{u['start']:.1f}s] {u['speaker']}: {u['text']}" for u in result
    )
    logger.info(f"[Transcription] Full transcript:\n{transcript_lines}\n")
    return result


def _build_utterances_from_words(response) -> list[dict]:
    """
    Fallback: group consecutive words by the same speaker into utterances.
    """
    try:
        words = response.results.channels[0].alternatives[0].words
    except (AttributeError, IndexError, TypeError):
        return []

    if not words:
        return []

    utterances = []
    current_speaker = None
    current_words = []
    current_start = 0.0
    current_end = 0.0

    for w in words:
        sp = w.speaker if hasattr(w, "speaker") and w.speaker is not None else 0
        if sp != current_speaker:
            if current_words and current_speaker is not None:
                utterances.append(
                    {
                        "speaker": SPEAKER_MAP.get(
                            current_speaker, f"Speaker {current_speaker}"
                        ),
                        "text": " ".join(current_words),
                        "start": round(current_start, 3),
                        "end": round(current_end, 3),
                    }
                )
            current_speaker = sp
            current_words = [
                w.punctuated_word if hasattr(w, "punctuated_word") else w.word
            ]
            current_start = float(w.start)
            current_end = float(w.end)
        else:
            current_words.append(
                w.punctuated_word if hasattr(w, "punctuated_word") else w.word
            )
            current_end = float(w.end)

    if current_words and current_speaker is not None:
        utterances.append(
            {
                "speaker": SPEAKER_MAP.get(
                    current_speaker, f"Speaker {current_speaker}"
                ),
                "text": " ".join(current_words),
                "start": round(current_start, 3),
                "end": round(current_end, 3),
            }
        )

    return utterances
