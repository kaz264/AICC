"""AICC 음성 대화 데모 — Typecast TTS 버전

사용법:
    python -m tests.voice_demo_typecast
    python -m tests.voice_demo_typecast --scenario insurance_claim
"""

import asyncio
import json
import argparse
import re
import time
import subprocess
import tempfile
import shutil
import httpx
from pathlib import Path
from anthropic import Anthropic

FFMPEG = r"C:\Users\minsu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

from backend import config
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge


# ── Typecast 음성 설정 ──

TYPECAST_API_KEY = config.OPENROUTER_API_KEY  # placeholder
TYPECAST_MODEL = "ssfm-v30"

# Conversational 한국어 음성
VOICES = {
    "customer": "tc_686dc43ebd6351e06ee64d74",       # Wonwoo (남, 대화형)
    "보험 상담사": "tc_68f9c6a72f0f04a417bb136f",      # Moonjung (여, 대화형, 감정 7개)
    "레스토랑 예약 안내": "tc_68537c9420b646f2176890ba", # Seojin (여, 대화형)
    "IT 기술지원": "tc_68662745779b66ba84fc4d84",       # Seheon (남, 대화형)
}


def clean_text_for_tts(text: str) -> str:
    text = text.replace("~", "").replace("♪", "").replace("♥", "").replace("★", "")
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"-\s+", "", text)
    text = text.replace("(", ", ").replace(")", ",")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",\s*,", ",", text)
    return text


def typecast_tts(voice_id: str, text: str) -> bytes:
    """Typecast TTS API 호출"""
    text = clean_text_for_tts(text)
    api_key = getattr(config, 'TYPECAST_API_KEY', None)
    if not api_key:
        # .env에서 직접 읽기
        import os
        api_key = os.getenv('TYPECAST_API_KEY', '')

    resp = httpx.post(
        "https://api.typecast.ai/v1/text-to-speech",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={
            "voice_id": voice_id,
            "text": text,
            "model": TYPECAST_MODEL,
            "language": "kor",
            "output": {"format": "mp3"},
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  [Typecast Error] {resp.status_code}: {resp.text[:100]}")
        return b""
    return resp.content


def generate_silence(duration_ms: int) -> bytes:
    """Google TTS SSML로 무음 생성"""
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


def run_text_conversation(client, system_prompt, persona_config, scenario):
    conversation = []
    follow_ups = scenario.get("follow_ups", [])
    follow_up_idx = 0

    enhanced_prompt = system_prompt + """

## 음성 출력용 추가 규칙
- 물결표를 절대 사용하지 마세요.
- 이모지나 특수문자를 사용하지 마세요.
- 불릿 포인트나 번호 목록 대신 자연스러운 대화체로 말하세요.
"""

    customer_msg = scenario["opening_message"]
    conversation.append({"role": "customer", "content": customer_msg})

    for _ in range(len(follow_ups) + 2):
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
        conversation.append({"role": "aicc", "content": resp.content[0].text})

        if follow_up_idx < len(follow_ups):
            conversation.append({"role": "customer", "content": follow_ups[follow_up_idx]})
            follow_up_idx += 1
        else:
            conversation.append({"role": "customer", "content": "네, 감사합니다. 도움이 됐어요."})
            llm_final = [
                {"role": "user" if m["role"] == "customer" else "assistant", "content": m["content"]}
                for m in conversation
            ]
            resp = client.messages.create(
                model=persona_config.get("llm_model", "claude-sonnet-4-20250514"),
                max_tokens=100, system=enhanced_prompt, messages=llm_final,
            )
            conversation.append({"role": "aicc", "content": resp.content[0].text})
            break

    return conversation


async def generate_demo(scenario_id="restaurant_reservation", output_path=None):
    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scenario:
        print(f"시나리오 '{scenario_id}'를 찾을 수 없습니다.")
        return

    print(f"시나리오: {scenario['id']} ({scenario['persona_name']})")
    print(f"TTS: Typecast (ssfm-v30, 한국어 네이티브)")

    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()

    personas = await list_personas()
    persona = next((p for p in personas if p.name == scenario["persona_name"]), None)
    if not persona:
        print(f"페르소나를 찾을 수 없습니다.")
        return

    # 1. 텍스트 대화
    print("\n1단계: 텍스트 대화 생성 중...")
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system_prompt = build_system_prompt(persona)
    conversation = run_text_conversation(
        client, system_prompt,
        {"llm_model": persona.llm_model, "knowledge_base_id": persona.knowledge_base_id},
        scenario,
    )
    for msg in conversation:
        role = "고객" if msg["role"] == "customer" else "상담사"
        print(f"  {role}: {msg['content']}")

    # 2. Typecast TTS
    print(f"\n2단계: Typecast 음성 합성 중... ({len(conversation)}턴)")

    customer_voice = VOICES["customer"]
    persona_voice = VOICES.get(persona.name, customer_voice)

    mp3_parts: list[bytes] = []

    print("  무음 생성 중...")
    silence_500 = generate_silence(500)
    silence_800 = generate_silence(800)
    silence_1200 = generate_silence(1200)

    # 인트로
    mp3_parts.append(typecast_tts(customer_voice, f"AICC 데모. {persona.name} 페르소나와 고객의 대화입니다."))
    mp3_parts.append(silence_1200)
    time.sleep(0.5)

    # 대화 (추임새는 LLM이 본 답변에 자연스럽게 포함)
    for i, msg in enumerate(conversation):
        clean = clean_text_for_tts(msg["content"])
        print(f"  [{i+1}/{len(conversation)}] {msg['role']}: {clean[:50]}...")

        if msg["role"] == "customer":
            mp3_parts.append(typecast_tts(customer_voice, msg["content"]))
            mp3_parts.append(silence_500)
        else:
            mp3_parts.append(typecast_tts(persona_voice, msg["content"]))
            mp3_parts.append(silence_800)

        time.sleep(0.5)  # rate limit

    mp3_parts.append(typecast_tts(customer_voice, "데모가 끝났습니다."))

    # 3. 개별 파일 저장 → ffmpeg로 합치기
    if not output_path:
        output_path = f"demo_typecast_{scenario_id}.mp3"
    output_full = Path(__file__).resolve().parent.parent / output_path

    tmp_dir = Path(tempfile.mkdtemp())
    part_files = []

    print(f"\n3단계: 오디오 파일 합치기...")
    for i, part in enumerate(mp3_parts):
        if not part:
            continue
        # Typecast는 WAV, Google은 MP3 → 확장자 자동 판별
        is_wav = part[:4] == b'RIFF'
        ext = ".wav" if is_wav else ".mp3"
        part_path = tmp_dir / f"part_{i:04d}{ext}"
        part_path.write_bytes(part)
        part_files.append(part_path)

    # ffmpeg concat: 모든 파일을 MP3로 통일하여 합치기
    filter_inputs = ""
    filter_concat = ""
    for i, pf in enumerate(part_files):
        filter_inputs += f" -i \"{pf}\""
    filter_concat = f"concat=n={len(part_files)}:v=0:a=1[out]"

    # ffmpeg 명령 구성
    cmd_parts = [f'"{FFMPEG}"', '-y']
    for pf in part_files:
        cmd_parts.append(f'-i "{pf}"')
    cmd_parts.append(f'-filter_complex "{" ".join(f"[{i}:a]" for i in range(len(part_files)))}concat=n={len(part_files)}:v=0:a=1[out]"')
    cmd_parts.append('-map "[out]"')
    cmd_parts.append(f'-c:a libmp3lame -q:a 2 "{output_full}"')

    cmd = " ".join(cmd_parts)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[:300]}")

    # 임시 파일 정리
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n완료!")
    print(f"  파일: {output_full}")
    print(f"  크기: {output_full.stat().st_size // 1024}KB")


def main():
    parser = argparse.ArgumentParser(description="AICC Typecast 데모")
    parser.add_argument("--scenario", default="restaurant_reservation")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    asyncio.run(generate_demo(args.scenario, args.output))


if __name__ == "__main__":
    main()
