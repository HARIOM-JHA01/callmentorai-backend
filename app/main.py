import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.config import get_settings
from app.database.connection import create_tables, get_db
from app.api import sessions, analysis, coach, auth, dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----------------------------------------------------------------
    # Startup
    # ----------------------------------------------------------------
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory ready: {settings.UPLOAD_DIR}")

    logger.info("Running database migrations …")
    await create_tables()
    logger.info("Database tables ready")

    # Re-enqueue any sessions that were left pending from a previous run
    from app.models.session import Session as SessionModel
    from app.services.processing_queue import processing_queue

    async for db in get_db():
        result = await db.execute(
            select(SessionModel).where(SessionModel.status == "pending")
        )
        pending = result.scalars().all()
        if pending:
            logger.info(f"Re-enqueueing {len(pending)} pending sessions from previous run")
            for s in pending:
                processing_queue.enqueue(s.id)
        else:
            processing_queue.start()
        break

    yield

    # ----------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------
    logger.info("Application shutting down")


app = FastAPI(
    title="CallMentor AI",
    description=(
        "AI-powered platform for analyzing customer service and sales calls. "
        "Upload a call recording and an evaluation rubric to receive automated coaching feedback, "
        "scoring, and a downloadable report. Interact with the AI coaching assistant to deep-dive "
        "into any aspect of the call."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ----------------------------------------------------------------
# CORS — allow all origins during development; tighten in production
# ----------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------
# Routers
# ----------------------------------------------------------------
app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(analysis.router)
app.include_router(coach.router)
app.include_router(dashboard.router)


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "CallMentor AI"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
