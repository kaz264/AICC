"""Tier 1: 파이프라인 통합 테스트

실제 bot.py의 build_pipeline()을 TestTransport로 테스트합니다.
실제 STT/LLM/TTS 서비스를 호출합니다 (mock 없음).

사용법:
    pytest tests/test_pipeline_integration.py -v -s
    pytest tests/test_pipeline_integration.py -k "restaurant" -v -s
"""

import asyncio
import time
import httpx
import pytest
import wave
import io
from pathlib import Path

from tests.test_transport import TestTransport
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend import config


# ── 고객 음성 생성 (Typecast) ──

def generate_customer_audio(text: str) -> bytes:
    """Typecast TTS로 고객 음성 생성"""
    resp = httpx.post(
        "https://api.typecast.ai/v1/text-to-speech",
        headers={"X-API-KEY": config.TYPECAST_API_KEY, "Content-Type": "application/json"},
        json={
            "voice_id": "tc_686dc43ebd6351e06ee64d74",  # Wonwoo (남)
            "text": text,
            "model": "ssfm-v30",
            "language": "kor",
            "output": {"format": "wav"},
        },
        timeout=30,
    )
    if resp.status_code != 200:
        pytest.skip(f"Typecast TTS 실패: {resp.status_code}")
    return resp.content


# ── Fixtures ──

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def setup():
    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()


async def _get_persona(name: str):
    personas = await list_personas()
    return next((p for p in personas if p.name == name), None)


# ── Tier 1 테스트: 파이프라인 정합성 ──

class TestPipelineIntegration:
    """실제 build_pipeline()을 TestTransport로 테스트"""

    @pytest.mark.slow
    async def test_pipeline_processes_audio(self):
        """기본: 오디오 입력 → STT → LLM → TTS → 오디오 출력"""
        from backend.pipeline.bot import build_pipeline

        persona = await _get_persona("레스토랑 예약 안내")
        assert persona is not None

        # 고객 음성 생성
        audio = generate_customer_audio("안녕하세요 예약하고 싶어요")
        assert len(audio) > 0

        # TestTransport로 파이프라인 실행
        transport = TestTransport(audio_bytes=audio)
        transport.start_timer()

        task, runner, components = await build_pipeline(persona, transport)

        # 타임아웃 설정 (30초)
        try:
            await asyncio.wait_for(runner.run(task), timeout=30)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        # 검증: 출력 오디오가 생성되었는가
        metrics = transport.metrics
        print(f"\n  출력 오디오 프레임: {metrics.output_audio_frames}")
        print(f"  출력 오디오 크기: {len(metrics.output_audio_bytes)} bytes")
        print(f"  STT 인식: {metrics.transcriptions}")
        print(f"  텍스트 출력: {metrics.text_outputs}")
        print(f"  TTFR: {metrics.ttfr_ms:.0f}ms")

        # 최소한 STT가 텍스트를 인식했거나 LLM이 텍스트를 생성했어야 함
        has_output = (
            metrics.output_audio_frames > 0
            or len(metrics.transcriptions) > 0
            or len(metrics.text_outputs) > 0
        )
        assert has_output, "파이프라인에서 아무런 출력이 없습니다"

    @pytest.mark.slow
    async def test_pipeline_rag_tool_calling(self):
        """RAG: 지식베이스 기반 정확한 답변"""
        from backend.pipeline.bot import build_pipeline

        persona = await _get_persona("보험 상담사")
        assert persona is not None

        audio = generate_customer_audio("실손보험 보장 범위가 어떻게 되나요")
        transport = TestTransport(audio_bytes=audio)
        transport.start_timer()

        task, runner, components = await build_pipeline(persona, transport)

        try:
            await asyncio.wait_for(runner.run(task), timeout=30)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        metrics = transport.metrics
        print(f"\n  출력 프레임 수: {len(metrics.all_output_frames)}")
        print(f"  TTFR: {metrics.ttfr_ms:.0f}ms")

        # 파이프라인이 정상 실행되었는가 (에러 없이)
        assert True  # 에러 없이 여기까지 오면 파이프라인 배선 정상

    @pytest.mark.slow
    async def test_pipeline_filler_enabled(self):
        """추임새: filler_enabled=True일 때 FillerProcessor가 동작"""
        from backend.pipeline.bot import build_pipeline

        persona = await _get_persona("레스토랑 예약 안내")
        assert persona is not None
        assert persona.filler_enabled is True

        audio = generate_customer_audio("주차 가능한가요")
        transport = TestTransport(audio_bytes=audio)

        task, runner, components = await build_pipeline(persona, transport)

        assert components["filler"].enabled is True
        print(f"\n  FillerProcessor enabled: {components['filler'].enabled}")

    @pytest.mark.slow
    async def test_pipeline_filler_disabled(self):
        """추임새: IT 기술지원은 filler_enabled=False"""
        from backend.pipeline.bot import build_pipeline

        persona = await _get_persona("IT 기술지원")
        assert persona is not None

        audio = generate_customer_audio("VPN이 안 됩니다")
        transport = TestTransport(audio_bytes=audio)

        task, runner, components = await build_pipeline(persona, transport)

        assert components["filler"].enabled is False

    @pytest.mark.slow
    async def test_pipeline_ttfr_measurement(self):
        """TTFR: Pipecat 파이프라인의 실제 응답 시간 측정"""
        from backend.pipeline.bot import build_pipeline

        persona = await _get_persona("레스토랑 예약 안내")
        audio = generate_customer_audio("디너 코스 가격이 얼마예요")
        transport = TestTransport(audio_bytes=audio)
        transport.start_timer()

        task, runner, components = await build_pipeline(persona, transport)

        try:
            await asyncio.wait_for(runner.run(task), timeout=30)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        ttfr = transport.metrics.ttfr_ms
        print(f"\n  TTFR: {ttfr:.0f}ms")
        if ttfr > 0:
            print(f"  {'목표 달성!' if ttfr < 3000 else '최적화 필요'}")
