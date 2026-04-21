from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables() -> None:
    # Import all models so Base has them registered
    from app.models import user as user_models  # noqa: F401
    from app.models import session as session_models  # noqa: F401
    from app.models import report as report_models  # noqa: F401

    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Add bilingual column to existing sessions table if it doesn't exist
        await conn.execute(text(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS metadata_es JSONB"
        ))
        await conn.execute(text(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS team VARCHAR(100)"
        ))
        await conn.execute(text(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS supervisor VARCHAR(100)"
        ))
        await conn.execute(text(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS campaign VARCHAR(100)"
        ))
        await conn.execute(text(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS queue VARCHAR(100)"
        ))
