"""AICC 음성 대화 데모 생성기

고객 Agent와 AICC 페르소나가 텍스트로 대화한 뒤,
양쪽 음성을 Google TTS로 합성하여 하나의 MP3 파일로 출력합니다.

사용법:
    python -m tests.voice_demo
    python -m tests.voice_demo --scenario restaurant_reservation
    python -m tests.voice_demo --scenario insurance_claim --output demo_insurance.mp3
"""

import asyncio
import json
import argparse
import os
import io
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

from google.cloud import texttospeech
from google.oauth2 import service_account
from anthropic import Anthropic

from backend import config
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge


# ── Google TTS 음성 설정 ──

# 고객 음성 (페르소나와 구별되는 다른 음성)
CUSTOMER_VOICE = texttospeech.VoiceSelectionParams(
    language_code="ko-KR",
    name="ko-KR-Neural2-B",  # 남성 음성 (고객 역할)
)

# 페르소나별 음성 매핑
PERSONA_VOICES = {
    "보험 상담사": texttospeech.VoiceSelectionParams(
        language_code="ko-KR", name="ko-KR-Neural2-A"  # 여성
    ),
    "레스토랑 예약 안내": texttospeech.VoiceSelectionParams(
        language_code="ko-KR", name="ko-KR-Neural2-C"  # 여성
    ),
    "IT 기술지원": texttospeech.VoiceSelectionParams(
        language_code="ko-KR", name="ko-KR-Neural2-C"  # 남성 대체
    ),
}

AUDIO_CONFIG = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=1.05,  # 약간 빠르게 (자연스러움)
    pitch=0.0,
)


def get_tts_client() -> texttospeech.TextToSpeechClient:
    """GCP 인증 + TTS 클라이언트 생성"""
    creds_path = Path(config.GOOGLE_APPLICATION_CREDENTIALS)
    if not creds_path.is_absolute():
        creds_path = Path(__file__).resolve().parent.parent / creds_path
    credentials = service_account.Credentials.from_service_account_file(str(creds_path))
    return texttospeech.TextToSpeechClient(credentials=credentials)


def synthesize_speech(
    tts_client: texttospeech.TextToSpeechClient,
    text: str,
    voice: texttospeech.VoiceSelectionParams,
) -> bytes:
    """텍스트 → MP3 바이트 변환"""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=AUDIO_CONFIG
    )
    return response.audio_content


def synthesize_silence(
    tts_client: texttospeech.TextToSpeechClient,
    duration_ms: int,
) -> bytes:
    """SSML break 태그로 무음 MP3 생성"""
    ssml = f'<speak><break time="{duration_ms}ms"/></speak>'
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=CUSTOMER_VOICE,
        audio_config=AUDIO_CONFIG,
    )
    return response.audio_content


def concat_mp3_bytes(parts: list[bytes], output_path: str):
    """여러 MP3 바이트를 단순 이어붙이기 (MP3는 프레임 단위라 직접 concat 가능)"""
    with open(output_path, "wb") as f:
        for part in parts:
            f.write(part)


def run_text_conversation(
    client: Anthropic,
    system_prompt: str,
    persona_config: dict,
    scenario: dict,
) -> list[dict]:
    """텍스트 레벨 대화 실행"""
    conversation = []
    follow_ups = scenario.get("follow_ups", [])
    follow_up_idx = 0

    # 고객 첫 메시지
    customer_msg = scenario["opening_message"]
    conversation.append({"role": "customer", "content": customer_msg})

    max_turns = len(follow_ups) + 2

    for _ in range(max_turns):
        # AICC 응답 (RAG 포함)
        rag_context = ""
        kb_id = persona_config.get("knowledge_base_id")
        if kb_id:
            msgs = [m["content"] for m in conversation if m["role"] == "customer"]
            query = " ".join(msgs[-2:])
            results = search_knowledge(kb_id, query, n_results=3)
            if results:
                rag_context = "\n\n## 참고 지식\n" + "\n\n".join(results)

        llm_messages = [
            {"role": "user" if m["role"] == "customer" else "assistant", "content": m["content"]}
            for m in conversation
        ]

        resp = client.messages.create(
            model=persona_config.get("llm_model", "claude-sonnet-4-20250514"),
            max_tokens=200,
            system=system_prompt + rag_context,
            messages=llm_messages,
        )
        aicc_response = resp.content[0].text
        conversation.append({"role": "aicc", "content": aicc_response})

        # 고객 후속 질문
        if follow_up_idx < len(follow_ups):
            customer_msg = follow_ups[follow_up_idx]
            follow_up_idx += 1
            conversation.append({"role": "customer", "content": customer_msg})
        else:
            # 자연스러운 마무리
            conversation.append({"role": "customer", "content": "네, 감사합니다. 도움이 됐어요."})
            # AICC 마무리 인사
            llm_messages_final = [
                {"role": "user" if m["role"] == "customer" else "assistant", "content": m["content"]}
                for m in conversation
            ]
            resp = client.messages.create(
                model=persona_config.get("llm_model", "claude-sonnet-4-20250514"),
                max_tokens=100,
                system=system_prompt,
                messages=llm_messages_final,
            )
            conversation.append({"role": "aicc", "content": resp.content[0].text})
            break

    return conversation


async def generate_voice_demo(
    scenario_id: str = "restaurant_reservation",
    output_path: str | None = None,
):
    """음성 대화 데모 MP3 생성"""

    # 시나리오 로드
    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scenario:
        print(f"시나리오 '{scenario_id}'를 찾을 수 없습니다.")
        return

    print(f"시나리오: {scenario['id']} ({scenario['persona_name']})")

    # 초기화
    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()

    personas = await list_personas()
    persona = next((p for p in personas if p.name == scenario["persona_name"]), None)
    if not persona:
        print(f"페르소나 '{scenario['persona_name']}'을 찾을 수 없습니다.")
        return

    # 1. 텍스트 대화 생성
    print("\n1단계: 텍스트 대화 생성 중...")
    anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system_prompt = build_system_prompt(persona)

    conversation = run_text_conversation(
        anthropic_client,
        system_prompt,
        {"llm_model": persona.llm_model, "knowledge_base_id": persona.knowledge_base_id},
        scenario,
    )

    # 대화 내용 출력
    for msg in conversation:
        role = "고객" if msg["role"] == "customer" else "상담사"
        print(f"  {role}: {msg['content']}")

    # 2. TTS 음성 합성
    print(f"\n2단계: 음성 합성 중... ({len(conversation)}턴)")
    tts_client = get_tts_client()
    persona_voice = PERSONA_VOICES.get(persona.name, CUSTOMER_VOICE)

    mp3_parts: list[bytes] = []

    # 무음 미리 생성
    silence_600 = synthesize_silence(tts_client, 600)
    silence_1000 = synthesize_silence(tts_client, 1000)

    # 인트로
    intro_text = f"AICC 데모입니다. {persona.name} 페르소나와 고객의 대화를 들려드리겠습니다."
    mp3_parts.append(synthesize_speech(tts_client, intro_text, CUSTOMER_VOICE))
    mp3_parts.append(silence_1000)

    # 각 턴 합성
    for i, msg in enumerate(conversation):
        print(f"  [{i+1}/{len(conversation)}] {msg['role']}: {msg['content'][:40]}...")
        voice = CUSTOMER_VOICE if msg["role"] == "customer" else persona_voice
        mp3_parts.append(synthesize_speech(tts_client, msg["content"], voice))
        mp3_parts.append(silence_600)

    # 아웃트로
    mp3_parts.append(synthesize_speech(tts_client, "데모가 끝났습니다. 감사합니다.", CUSTOMER_VOICE))

    # 3. MP3 저장
    if not output_path:
        output_path = f"demo_{scenario_id}.mp3"

    output_full = Path(__file__).resolve().parent.parent / output_path
    print(f"\n3단계: MP3 저장 중...")
    concat_mp3_bytes(mp3_parts, str(output_full))

    file_size_kb = output_full.stat().st_size / 1024
    print(f"\n완료!")
    print(f"  파일: {output_full}")
    print(f"  크기: {file_size_kb:.0f}KB")
    print(f"  턴 수: {len(conversation)}")


def main():
    parser = argparse.ArgumentParser(description="AICC 음성 대화 데모 생성")
    parser.add_argument(
        "--scenario", type=str, default="restaurant_reservation",
        help="시나리오 ID (기본: restaurant_reservation)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="출력 파일 경로 (기본: demo_{scenario}.mp3)",
    )
    args = parser.parse_args()
    asyncio.run(generate_voice_demo(args.scenario, args.output))


if __name__ == "__main__":
    main()
