import asyncio
import logging
import os
from typing import AsyncIterator

from deepgram import AsyncDeepgramClient
from deepgram.core.request_options import RequestOptions
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SPEAKER_MAP = {0: "Agent", 1: "Customer"}
_CHUNK_SIZE = 256 * 1024  # 256 KB per chunk


async def _stream_with_progress(audio_path: str) -> AsyncIterator[bytes]:
    """Yield audio file in chunks, logging upload progress to the terminal."""
    total = os.path.getsize(audio_path)
    sent = 0
    last_pct = -1

    with open(audio_path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            sent += len(chunk)
            pct = int(sent / total * 100)
            if pct // 10 != last_pct // 10:
                logger.info(
                    f"[Transcription] Uploading audio … {pct}%  "
                    f"({sent / 1_048_576:.1f} / {total / 1_048_576:.1f} MB)"
                )
                last_pct = pct
            yield chunk
            await asyncio.sleep(0)  # yield control back to event loop

    logger.info("[Transcription] Upload complete — waiting for Deepgram to process …")


async def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Transcribe an audio file using Deepgram with speaker diarization.

    Returns a list of utterance dicts:
        [{"speaker": "Agent"|"Customer", "text": str, "start": float, "end": float}]
    """
    deepgram = AsyncDeepgramClient(api_key=settings.DEEPGRAM_API_KEY)

    file_mb = os.path.getsize(audio_path) / 1_048_576
    logger.info(f"[Transcription] Starting upload: {audio_path} ({file_mb:.1f} MB)")

    response = await deepgram.listen.v1.media.transcribe_file(
        request=_stream_with_progress(audio_path),
        model="nova-2",
        diarize=True,
        punctuate=True,
        utterances=True,
        request_options=RequestOptions(timeout_in_seconds=300),
    )

    utterances_raw = response.results.utterances
    if not utterances_raw:
        # Fallback: try to build utterances from words if diarization data exists
        logger.warning("No utterances returned from Deepgram; attempting word-level fallback.")
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
                        "speaker": SPEAKER_MAP.get(current_speaker, f"Speaker {current_speaker}"),
                        "text": " ".join(current_words),
                        "start": round(current_start, 3),
                        "end": round(current_end, 3),
                    }
                )
            current_speaker = sp
            current_words = [w.punctuated_word if hasattr(w, "punctuated_word") else w.word]
            current_start = float(w.start)
            current_end = float(w.end)
        else:
            current_words.append(w.punctuated_word if hasattr(w, "punctuated_word") else w.word)
            current_end = float(w.end)

    if current_words and current_speaker is not None:
        utterances.append(
            {
                "speaker": SPEAKER_MAP.get(current_speaker, f"Speaker {current_speaker}"),
                "text": " ".join(current_words),
                "start": round(current_start, 3),
                "end": round(current_end, 3),
            }
        )

    return utterances
