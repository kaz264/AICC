"""테스트 공통 설정 및 fixture"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# 테스트용 환경변수 (실제 API 호출 방지)
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["DAILY_API_KEY"] = "test-key"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
os.environ["ELEVENLABS_API_KEY"] = "test-key"

# DB를 임시 파일로 교체
_temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["TEST_DB_PATH"] = _temp_db.name

import backend.config as config
config.DB_PATH = Path(_temp_db.name)
config.CHROMA_PATH = Path(tempfile.mkdtemp()) / "chroma_test"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """각 테스트 전에 DB 초기화"""
    from backend.db.database import init_db
    await init_db()
    yield


@pytest.fixture
def client():
    """FastAPI 동기 테스트 클라이언트"""
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client():
    """FastAPI 비동기 테스트 클라이언트"""
    from backend.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
