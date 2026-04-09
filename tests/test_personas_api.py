"""페르소나 REST API 테스트"""

import pytest


# ── 생성 ──

async def test_create_persona(async_client):
    resp = await async_client.post("/api/personas/", json={
        "name": "API 테스트 상담사",
        "system_prompt": "당신은 API 테스트 상담사입니다.",
        "greeting_message": "반갑습니다!",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "API 테스트 상담사"
    assert data["id"] is not None
    assert data["language"] == "ko-KR"


async def test_create_persona_missing_required(async_client):
    resp = await async_client.post("/api/personas/", json={
        "name": "이름만",
        # system_prompt 누락
    })
    assert resp.status_code == 422  # Validation error


# ── 목록 ──

async def test_list_personas(async_client):
    # 하나 생성
    await async_client.post("/api/personas/", json={
        "name": "목록 테스트",
        "system_prompt": "테스트",
    })

    resp = await async_client.get("/api/personas/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


# ── 조회 ──

async def test_get_persona(async_client):
    create_resp = await async_client.post("/api/personas/", json={
        "name": "조회 대상",
        "system_prompt": "테스트",
    })
    persona_id = create_resp.json()["id"]

    resp = await async_client.get(f"/api/personas/{persona_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "조회 대상"


async def test_get_persona_not_found(async_client):
    resp = await async_client.get("/api/personas/nonexistent-id")
    assert resp.status_code == 404


# ── 수정 ──

async def test_update_persona(async_client):
    create_resp = await async_client.post("/api/personas/", json={
        "name": "수정 전",
        "system_prompt": "원본",
    })
    persona_id = create_resp.json()["id"]

    resp = await async_client.put(f"/api/personas/{persona_id}", json={
        "name": "수정 후",
        "tts_provider": "elevenlabs",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "수정 후"
    assert data["tts_provider"] == "elevenlabs"
    assert data["system_prompt"] == "원본"  # 변경 안 한 필드 유지


async def test_update_persona_not_found(async_client):
    resp = await async_client.put("/api/personas/nonexistent-id", json={
        "name": "없는거",
    })
    assert resp.status_code == 404


# ── 삭제 ──

async def test_delete_persona(async_client):
    create_resp = await async_client.post("/api/personas/", json={
        "name": "삭제 대상",
        "system_prompt": "테스트",
    })
    persona_id = create_resp.json()["id"]

    resp = await async_client.delete(f"/api/personas/{persona_id}")
    assert resp.status_code == 200

    # 삭제 후 조회 불가
    get_resp = await async_client.get(f"/api/personas/{persona_id}")
    assert get_resp.status_code == 404


async def test_delete_persona_not_found(async_client):
    resp = await async_client.delete("/api/personas/nonexistent-id")
    assert resp.status_code == 404


# ── 건강 체크 ──

async def test_health(async_client):
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


async def test_root(async_client):
    resp = await async_client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "AICC PoC"
