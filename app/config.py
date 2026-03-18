from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/callmentorai"
    OPENAI_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""
    UPLOAD_DIR: str = "uploads/"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
