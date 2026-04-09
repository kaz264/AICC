"""AICC 음성 대화 데모 생성기

고객 Agent와 AICC 페르소나가 텍스트로 대화한 뒤,
양쪽 음성을 ElevenLabs TTS로 합성하여 하나의 MP3 파일로 출력합니다.

사용법:
    python -m tests.voice_demo
    python -m tests.voice_demo --scenario restaurant_reservation
    python -m tests.voice_demo --scenario insurance_claim --output demo_insurance.mp3
"""

import asyncio
import json
import argparse
import re
import time
import httpx
from pathlib import Path
from anthropic import Anthropic

from backend import config
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge


# ── 텍스트 정제 ──

def clean_text_for_tts(text: str) -> str:
    """TTS가 읽으면 안 되는 특수문자 제거"""
    # 물결표, 이모지 등 제거
    text = text.replace("~", "")
    text = text.replace("♪", "")
    text = text.replace("♥", "")
    text = text.replace("★", "")
    # 연속 느낌표/물음표를 하나로
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\?+", "?", text)
    # 마크다운 기호 제거
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"-\s+", "", text)
    # 괄호 안 부연설명은 유지하되 괄호 제거
    text = text.replace("(", ", ").replace(")", ",")
    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    # 연속 쉼표 정리
    text = re.sub(r",\s*,", ",", text)
    return text


# ── ElevenLabs TTS ──

# 음성 ID (ElevenLabs 한국어 네이티브 음성)
VOICES = {
    # 고객 - 남성 (Hyunbin: 외교적, 서울 억양)
    "customer": "s07IwTCOrCDCaETjUVjx",
    # 페르소나별
    "보험 상담사": "uyVNoMrnUku1dZyVEXwD",      # Anna Kim (여, 부드럽고 차분)
    "레스토랑 예약 안내": "uyVNoMrnUku1dZyVEXwD", # Anna Kim (여)
    "IT 기술지원": "ZJCNdZEjYwkOElxugmW2",      # Hyuk (남, 명확)
}

ELEVENLABS_MODEL = "eleven_multilingual_v2"


def synthesize_elevenlabs(text: str, voice_id: str) -> bytes:
    """ElevenLabs TTS로 음성 합성"""
    text = clean_text_for_tts(text)
    resp = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": 0.4,           # 낮을수록 감정 변화 많음
                "similarity_boost": 0.7,
                "style": 0.3,               # 스타일 강화
                "use_speaker_boost": True,
            },
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  [TTS Error] {resp.status_code}: {resp.text[:100]}")
        return b""
    return resp.content


def generate_silence_mp3(duration_ms: int) -> bytes:
    """최소 크기의 무음 MP3 생성 (Google TTS SSML 사용)"""
    from google.cloud import texttospeech
    from google.oauth2 import service_account

    creds_path = Path(config.GOOGLE_APPLICATION_CREDENTIALS)
    if not creds_path.is_absolute():
        creds_path = Path(__file__).resolve().parent.parent / creds_path
    credentials = service_account.Credentials.from_service_account_file(str(creds_path))
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

    ssml = f'<speak><break time="{duration_ms}ms"/></speak>'
    response = tts_client.synthesize_speech(
        input=texttospeech.SynthesisInput(ssml=ssml),
        voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Neural2-A"),
        audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
    )
    return response.audio_content


def concat_mp3_bytes(parts: list[bytes], output_path: str):
    """여러 MP3 바이트를 이어붙이기"""
    with open(output_path, "wb") as f:
        for part in parts:
            if part:
                f.write(part)


# ── 텍스트 대화 엔진 ──

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

    # 시스템 프롬프트에 추가 지시
    enhanced_prompt = system_prompt + """

## 음성 출력용 추가 규칙
- 물결표(~)를 절대 사용하지 마세요.
- 이모지나 특수문자를 사용하지 마세요.
- 불릿 포인트나 번호 목록 대신 자연스러운 대화체로 말하세요.
"""

    customer_msg = scenario["opening_message"]
    conversation.append({"role": "customer", "content": customer_msg})

    max_turns = len(follow_ups) + 2

    for _ in range(max_turns):
        # RAG
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
            system=enhanced_prompt + rag_context,
            messages=llm_messages,
        )
        aicc_response = resp.content[0].text
        conversation.append({"role": "aicc", "content": aicc_response})

        if follow_up_idx < len(follow_ups):
            customer_msg = follow_ups[follow_up_idx]
            follow_up_idx += 1
            conversation.append({"role": "customer", "content": customer_msg})
        else:
            conversation.append({"role": "customer", "content": "네, 감사합니다. 도움이 됐어요."})
            llm_messages_final = [
                {"role": "user" if m["role"] == "customer" else "assistant", "content": m["content"]}
                for m in conversation
            ]
            resp = client.messages.create(
                model=persona_config.get("llm_model", "claude-sonnet-4-20250514"),
                max_tokens=100,
                system=enhanced_prompt,
                messages=llm_messages_final,
            )
            conversation.append({"role": "aicc", "content": resp.content[0].text})
            break

    return conversation


# ── 메인 ──

async def generate_voice_demo(
    scenario_id: str = "restaurant_reservation",
    output_path: str | None = None,
):
    """음성 대화 데모 MP3 생성"""

    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scenario:
        print(f"시나리오 '{scenario_id}'를 찾을 수 없습니다.")
        return

    print(f"시나리오: {scenario['id']} ({scenario['persona_name']})")
    print(f"TTS 엔진: ElevenLabs (eleven_multilingual_v2)")

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

    for msg in conversation:
        role = "고객" if msg["role"] == "customer" else "상담사"
        print(f"  {role}: {msg['content']}")

    # 2. ElevenLabs TTS 합성
    print(f"\n2단계: ElevenLabs 음성 합성 중... ({len(conversation)}턴)")

    customer_voice = VOICES["customer"]
    persona_voice = VOICES.get(persona.name, customer_voice)

    mp3_parts: list[bytes] = []

    # 무음 (Google TTS로 생성)
    print("  무음 생성 중...")
    silence_300 = generate_silence_mp3(300)    # STT 처리 시간
    silence_700 = generate_silence_mp3(700)    # 턴 사이 간격
    silence_1000 = generate_silence_mp3(1000)  # LLM 처리 대기
    silence_1200 = generate_silence_mp3(1200)

    # 추임새 목록 (페르소나 음성으로 미리 합성)
    filler_texts = ["네,", "네, 알겠습니다.", "아, 그렇군요."]
    fillers_audio: list[bytes] = []
    if persona.filler_enabled:
        print("  추임새 합성 중...")
        for ft in filler_texts:
            audio = synthesize_elevenlabs(ft, persona_voice)
            fillers_audio.append(audio)
            time.sleep(0.3)

    filler_idx = 0

    # 인트로
    intro_text = f"AICC 데모. {persona.name} 페르소나와 고객의 대화입니다."
    mp3_parts.append(synthesize_elevenlabs(intro_text, VOICES["customer"]))
    mp3_parts.append(silence_1200)

    # 각 턴 합성 (추임새 + 실제 지연 시뮬레이션)
    for i, msg in enumerate(conversation):
        clean = clean_text_for_tts(msg["content"])
        print(f"  [{i+1}/{len(conversation)}] {msg['role']}: {clean[:50]}...")

        if msg["role"] == "customer":
            # 고객 발화
            audio = synthesize_elevenlabs(msg["content"], customer_voice)
            mp3_parts.append(audio)
            # 고객 발화 후 짧은 간격 (STT 처리 시뮬레이션)
            mp3_parts.append(silence_300)
        else:
            # 상담사 응답: 추임새 → 대기 → 본 답변
            if fillers_audio and persona.filler_enabled:
                mp3_parts.append(fillers_audio[filler_idx % len(fillers_audio)])
                filler_idx += 1
                mp3_parts.append(silence_1000)  # LLM 처리 대기

            audio = synthesize_elevenlabs(msg["content"], persona_voice)
            mp3_parts.append(audio)
            mp3_parts.append(silence_700)  # 답변 후 간격

        # ElevenLabs rate limit 방지
        time.sleep(0.3)

    # 아웃트로
    mp3_parts.append(synthesize_elevenlabs("데모가 끝났습니다.", VOICES["customer"]))

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
