"""AICC PoC 환경 설정"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Database
DB_PATH = Path(__file__).resolve().parent / "aicc.db"

# ChromaDB
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_data"

# Defaults
DEFAULT_LANGUAGE = "ko-KR"
DEFAULT_LLM_MODEL = "claude-sonnet-4-20250514"
DEFAULT_STT_PROVIDER = "google"
DEFAULT_TTS_PROVIDER = "google"
