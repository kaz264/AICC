"""DB CRUD 테스트"""

import pytest
from backend.db.database import (
    create_persona,
    get_persona,
    list_personas,
    update_persona,
    delete_persona,
    create_call_log,
    end_call_log,
    list_call_logs,
)
from backend.models.persona import PersonaCreate, PersonaUpdate


# ── Persona CRUD ──

async def test_create_persona():
    data = PersonaCreate(
        name="테스트 상담사",
        system_prompt="당신은 테스트 상담사입니다.",
        greeting_message="안녕하세요, 테스트입니다.",
    )
    persona = await create_persona(data)

    assert persona.id is not None
    assert persona.name == "테스트 상담사"
    assert persona.system_prompt == "당신은 테스트 상담사입니다."
    assert persona.language == "ko-KR"
    assert persona.created_at is not None


async def test_get_persona():
    data = PersonaCreate(name="조회 테스트", system_prompt="테스트 프롬프트")
    created = await create_persona(data)

    fetched = await get_persona(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "조회 테스트"


async def test_get_persona_not_found():
    result = await get_persona("nonexistent-id")
    assert result is None


async def test_list_personas():
    # 기존 데이터 + 새로 2개 추가
    await create_persona(PersonaCreate(name="목록A", system_prompt="A"))
    await create_persona(PersonaCreate(name="목록B", system_prompt="B"))

    personas = await list_personas()
    assert len(personas) >= 2
    names = [p.name for p in personas]
    assert "목록A" in names
    assert "목록B" in names


async def test_update_persona():
    data = PersonaCreate(name="수정 전", system_prompt="원본 프롬프트")
    created = await create_persona(data)

    updated = await update_persona(
        created.id,
        PersonaUpdate(name="수정 후", vad_sensitivity=0.8),
    )
    assert updated is not None
    assert updated.name == "수정 후"
    assert updated.vad_sensitivity == 0.8
    assert updated.system_prompt == "원본 프롬프트"  # 변경 안 한 필드 유지


async def test_update_persona_not_found():
    result = await update_persona("nonexistent-id", PersonaUpdate(name="없는거"))
    assert result is None


async def test_delete_persona():
    data = PersonaCreate(name="삭제 대상", system_prompt="삭제될 프롬프트")
    created = await create_persona(data)

    deleted = await delete_persona(created.id)
    assert deleted is True

    # 삭제 후 조회 불가
    fetched = await get_persona(created.id)
    assert fetched is None


async def test_delete_persona_not_found():
    deleted = await delete_persona("nonexistent-id")
    assert deleted is False


# ── Call Log ──

async def test_create_and_end_call_log():
    persona = await create_persona(
        PersonaCreate(name="통화 테스트", system_prompt="테스트")
    )

    call = await create_call_log(persona.id, "https://test.daily.co/room")
    assert call.status == "active"
    assert call.persona_id == persona.id

    await end_call_log(call.id, duration_seconds=60, latency_avg_ms=450.5)

    logs = await list_call_logs()
    ended_call = next((l for l in logs if l.id == call.id), None)
    assert ended_call is not None
    assert ended_call.status == "ended"
    assert ended_call.duration_seconds == 60
    assert ended_call.latency_avg_ms == 450.5
