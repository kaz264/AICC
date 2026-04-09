"""페르소나 로더 테스트 — 외부 API 호출 없이 설정 로직만 검증"""

import pytest
from backend.models.persona import Persona
from backend.pipeline.persona_loader import build_system_prompt


def _make_persona(**overrides) -> Persona:
    """테스트용 페르소나 생성 헬퍼"""
    defaults = dict(
        id="test-id",
        name="테스트",
        system_prompt="당신은 테스트 상담사입니다.",
        greeting_message="안녕하세요",
        language="ko-KR",
        stt_provider="google",
        tts_provider="google",
        tts_voice_id="ko-KR-Neural2-C",
        llm_model="claude-sonnet-4-20250514",
        vad_sensitivity=0.5,
        interrupt_threshold_ms=800,
        filler_enabled=True,
        knowledge_base_id=None,
    )
    defaults.update(overrides)
    return Persona(**defaults)


def test_build_system_prompt_with_filler():
    """추임새 활성화 시 프롬프트에 추임새 지시가 포함"""
    persona = _make_persona(filler_enabled=True)
    prompt = build_system_prompt(persona)

    assert "당신은 테스트 상담사입니다." in prompt
    assert "추임새" in prompt or "아, 네~" in prompt


def test_build_system_prompt_without_filler():
    """추임새 비활성화 시 추임새 지시가 없어야 함"""
    persona = _make_persona(filler_enabled=False)
    prompt = build_system_prompt(persona)

    assert "당신은 테스트 상담사입니다." in prompt
    assert "아, 네~" not in prompt


def test_build_system_prompt_preserves_original():
    """원본 시스템 프롬프트가 손상되지 않아야 함"""
    original = "당신은 전문적인 보험 상담사입니다. 고객에게 친절하게 상담하세요."
    persona = _make_persona(system_prompt=original, filler_enabled=False)
    prompt = build_system_prompt(persona)

    assert prompt == original


def test_persona_default_values():
    """페르소나 기본값이 올바르게 설정되는지 확인"""
    persona = _make_persona()

    assert persona.language == "ko-KR"
    assert persona.stt_provider == "google"
    assert persona.tts_provider == "google"
    assert persona.vad_sensitivity == 0.5
    assert persona.interrupt_threshold_ms == 800
    assert persona.filler_enabled is True


def test_persona_custom_values():
    """커스텀 값이 올바르게 반영되는지 확인"""
    persona = _make_persona(
        tts_provider="elevenlabs",
        tts_voice_id="custom-voice-123",
        vad_sensitivity=0.8,
        interrupt_threshold_ms=1200,
        knowledge_base_id="my_kb",
    )

    assert persona.tts_provider == "elevenlabs"
    assert persona.tts_voice_id == "custom-voice-123"
    assert persona.vad_sensitivity == 0.8
    assert persona.interrupt_threshold_ms == 1200
    assert persona.knowledge_base_id == "my_kb"
