import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    def __init__(self) -> None:
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.APP_TIMEZONE = os.getenv("APP_TIMEZONE", "UTC")

    @property
    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    def get_openai_kwargs(self, *, allow_unconfigured: bool = False) -> dict[str, Any]:
        api_key = self.OPENAI_API_KEY or ("test-key" if allow_unconfigured else "")
        return {
            "model": self.OPENAI_MODEL,
            "api_key": api_key,
        }


settings = Settings()
