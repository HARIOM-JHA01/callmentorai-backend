import json
import logging
import os
import numpy as np
import faiss
from openai import AsyncOpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_openai_client: AsyncOpenAI | None = None
CHUNK_SIZE = 5  # utterances per chunk
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _chunk_transcript(transcript: list[dict], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Chunk transcript utterances into text segments of ~chunk_size utterances."""
    chunks = []
    for i in range(0, len(transcript), chunk_size):
        segment = transcript[i: i + chunk_size]
        lines = []
        for utt in segment:
            speaker = utt.get("speaker", "Unknown")
            text = utt.get("text", "")
            start = utt.get("start", 0.0)
            minutes = int(start // 60)
            seconds = int(start % 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {speaker}: {text}")
        chunks.append("\n".join(lines))
    return chunks


def _index_path(session_id: str) -> str:
    return os.path.join(settings.UPLOAD_DIR, session_id, "faiss.index")


def _chunks_path(session_id: str) -> str:
    return os.path.join(settings.UPLOAD_DIR, session_id, "chunks.json")


async def _embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts via OpenAI."""
    client = _get_openai()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


async def build_embeddings(session_id: str, transcript: list[dict]) -> None:
    """
    Chunk the transcript, generate embeddings, build a FAISS index,
    and persist the index and chunks to disk.
    """
    chunks = _chunk_transcript(transcript)
    if not chunks:
        logger.warning(f"No chunks to embed for session {session_id}")
        return

    logger.info(f"Generating embeddings for {len(chunks)} chunks (session {session_id})")
    vectors = await _embed_texts(chunks)

    # Build FAISS flat L2 index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vectors)

    idx_path = _index_path(session_id)
    chunks_path = _chunks_path(session_id)

    faiss.write_index(index, idx_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"FAISS index saved to {idx_path} ({index.ntotal} vectors)")


async def search_transcript(session_id: str, query: str, top_k: int = 5) -> list[str]:
    """
    Embed the query and retrieve the top_k most relevant transcript chunks.
    Returns a list of chunk strings.
    """
    idx_path = _index_path(session_id)
    chunks_path = _chunks_path(session_id)

    if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
        logger.warning(f"No FAISS index found for session {session_id}")
        return []

    index = faiss.read_index(idx_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: list[str] = json.load(f)

    query_vec = await _embed_texts([query])
    distances, indices = index.search(query_vec, min(top_k, index.ntotal))

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])

    logger.info(f"Transcript search for session {session_id}: {len(results)} results returned")
    return results
