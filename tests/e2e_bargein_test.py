"""AICC Barge-in 테스트

고객이 상담사 응답 도중에 끼어드는 시나리오를 테스트합니다.
- 고객: 여성 음성 (Typecast Seojin)
- 상담사: 남성 음성 (Typecast Seheon / IT 기술지원)

시나리오:
  턴 1: 고객 질문 → 상담사 응답
  턴 2: 고객 질문 → 상담사가 길게 답하는 중 → 고객이 끼어듦 ("아, 그건 아니고요")
  턴 3: 상담사가 끊기고 새로 응답
  턴 4~6: 정상 대화 후 마무리

사용법:
    python -m tests.e2e_bargein_test
"""

import asyncio
import json
import re
import time
import subprocess
import tempfile
import shutil
import httpx
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic
from google.cloud import texttospeech, speech
from google.oauth2 import service_account

from backend import config
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge

FFMPEG = "ffmpeg"

# ── 음성 설정: 여성 고객 + 남성 상담사 ──
CUSTOMER_VOICE = "tc_68537c9420b646f2176890ba"   # Seojin (여성, Conversational)
PERSONA_VOICE = "tc_68662745779b66ba84fc4d84"     # Seheon (남성, Conversational)


def clean_text(text):
    text = text.replace("~", "").replace("♪", "").replace("♥", "")
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"-\s+", "", text)
    text = text.replace("(", ", ").replace(")", ",")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_gcp_credentials():
    creds_path = Path(config.GOOGLE_APPLICATION_CREDENTIALS)
    if not creds_path.is_absolute():
        creds_path = Path(__file__).resolve().parent.parent / creds_path
    return service_account.Credentials.from_service_account_file(str(creds_path))


def typecast_tts(voice_id, text):
    text = clean_text(text)
    resp = httpx.post(
        "https://api.typecast.ai/v1/text-to-speech",
        headers={"X-API-KEY": config.TYPECAST_API_KEY, "Content-Type": "application/json"},
        json={"voice_id": voice_id, "text": text, "model": "ssfm-v30", "language": "kor", "output": {"format": "wav"}},
        timeout=30,
    )
    return resp.content if resp.status_code == 200 else b""


def stt_recognize(audio_bytes):
    credentials = get_gcp_credentials()
    client = speech.SpeechClient(credentials=credentials)
    is_wav = audio_bytes[:4] == b'RIFF'

    import struct
    sample_rate = struct.unpack('<I', audio_bytes[24:28])[0] if is_wav else 24000

    stt_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16 if is_wav
        else speech.RecognitionConfig.AudioEncoding.MP3,
        language_code="ko-KR", model="latest_long",
        sample_rate_hertz=sample_rate if is_wav else None,
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=stt_config, audio=speech.RecognitionAudio(content=audio_bytes))
    return "".join(r.alternatives[0].transcript for r in response.results)


def generate_silence(duration_ms):
    tts_client = texttospeech.TextToSpeechClient(credentials=get_gcp_credentials())
    ssml = f'<speak><break time="{duration_ms}ms"/></speak>'
    response = tts_client.synthesize_speech(
        input=texttospeech.SynthesisInput(ssml=ssml),
        voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Neural2-A"),
        audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
    )
    return response.audio_content


def truncate_audio(audio_bytes, keep_ratio=0.4):
    """오디오를 앞부분만 남기고 자르기 (Barge-in 시뮬레이션)"""
    if audio_bytes[:4] == b'RIFF':
        # WAV: 헤더(44) + 데이터
        header = audio_bytes[:44]
        data = audio_bytes[44:]
        truncated_data = data[:int(len(data) * keep_ratio)]
        # WAV 헤더의 데이터 크기 업데이트
        import struct
        new_size = len(truncated_data)
        header = header[:4] + struct.pack('<I', new_size + 36) + header[8:40] + struct.pack('<I', new_size) + header[44:]
        return header + truncated_data
    else:
        return audio_bytes[:int(len(audio_bytes) * keep_ratio)]


def concat_audio(parts, output_path):
    tmp_dir = Path(tempfile.mkdtemp())
    part_files = []
    for i, part in enumerate(parts):
        if not part:
            continue
        ext = ".wav" if part[:4] == b'RIFF' else ".mp3"
        p = tmp_dir / f"p_{i:04d}{ext}"
        p.write_bytes(part)
        part_files.append(p)
    if not part_files:
        return
    cmd_parts = [f'"{FFMPEG}"', '-y']
    for pf in part_files:
        cmd_parts.append(f'-i "{pf}"')
    inputs = " ".join(f"[{i}:a]" for i in range(len(part_files)))
    cmd_parts.append(f'-filter_complex "{inputs}concat=n={len(part_files)}:v=0:a=1[out]"')
    cmd_parts.append('-map "[out]"')
    cmd_parts.append(f'-c:a libmp3lame -q:a 2 "{output_path}"')
    subprocess.run(" ".join(cmd_parts), shell=True, capture_output=True, timeout=120)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def llm_respond(client, system_prompt, conversation, kb_id, model):
    rag_context = ""
    if kb_id:
        user_msgs = [m["content"] for m in conversation if m["role"] == "user"]
        query = " ".join(user_msgs[-2:]) if len(user_msgs) >= 2 else user_msgs[-1] if user_msgs else ""
        if query:
            results = search_knowledge(kb_id, query, n_results=3)
            if results:
                rag_context = "\n\n## 참고 지식\n" + "\n\n".join(results)

    enhanced = system_prompt + rag_context + "\n\n## 음성 출력용 규칙\n- 물결표 사용 금지\n- 이모지/특수문자 금지\n- 자연스러운 대화체"

    resp = client.messages.create(
        model=model, max_tokens=200, system=enhanced, messages=conversation,
    )
    return resp.content[0].text


# ── Barge-in 시나리오 ──

BARGEIN_SCENARIO = {
    "persona_name": "IT 기술지원",
    "customer_profile": "30대 여성 회사원, 이메일이 안 되는데 상담사가 엉뚱한 답변을 해서 끊고 다시 설명하는 상황",
    "turns": [
        {
            "type": "normal",
            "customer": "안녕하세요, 이메일이 갑자기 안 돼요. 아웃룩에서 메일이 안 보내져요.",
        },
        {
            "type": "bargein",
            "customer_interrupt": "아, 그건 아니고요. 보내기가 안 되는 거예요. 받는 건 돼요.",
            "description": "상담사가 수신 문제로 오해하고 답변하는 도중 고객이 끼어듦",
        },
        {
            "type": "normal",
            "customer": "보내기 버튼을 누르면 오류가 나요. 뭐라고 나오냐면... 서버 연결 실패라고요.",
        },
        {
            "type": "normal",
            "customer": "네, 해볼게요. 잠깐만요.",
        },
        {
            "type": "normal",
            "customer": "아, 됐어요! 감사합니다.",
        },
    ],
}


async def run_bargein_test():
    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()

    personas = await list_personas()
    persona = next((p for p in personas if p.name == "IT 기술지원"), None)
    if not persona:
        print("IT 기술지원 페르소나를 찾을 수 없습니다.")
        return

    print(f"\n{'='*70}")
    print(f"Barge-in 테스트: IT 기술지원")
    print(f"고객: 여성 (Seojin) | 상담사: 남성 (Seheon)")
    print(f"{'='*70}")

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system_prompt = build_system_prompt(persona)
    llm_conversation = []
    audio_parts = []

    silence_500 = generate_silence(500)
    silence_300 = generate_silence(300)
    silence_150 = generate_silence(150)  # 끼어들기 직전 짧은 간격

    # 인트로
    intro = typecast_tts(CUSTOMER_VOICE, "Barge-in 테스트 데모. 고객이 상담사 응답 도중에 끼어드는 시나리오입니다.")
    audio_parts.append(intro)
    audio_parts.append(silence_500)
    time.sleep(0.5)

    for i, turn in enumerate(BARGEIN_SCENARIO["turns"]):
        print(f"\n  -- 턴 {i+1} ({turn['type']}) --")

        if turn["type"] == "normal":
            # 일반 턴: 고객 → 상담사
            customer_text = turn["customer"]
            print(f"  고객: {customer_text}")

            customer_audio = typecast_tts(CUSTOMER_VOICE, customer_text)
            audio_parts.append(customer_audio)
            audio_parts.append(silence_300)
            time.sleep(0.5)

            # STT
            stt_text = stt_recognize(customer_audio) if customer_audio else customer_text
            print(f"  STT: {stt_text}")

            # LLM
            llm_conversation.append({"role": "user", "content": stt_text or customer_text})
            response = llm_respond(client, system_prompt, llm_conversation, persona.knowledge_base_id, persona.llm_model)
            llm_conversation.append({"role": "assistant", "content": response})
            print(f"  상담사: {response}")

            response_audio = typecast_tts(PERSONA_VOICE, response)
            audio_parts.append(response_audio)
            audio_parts.append(silence_500)
            time.sleep(0.5)

        elif turn["type"] == "bargein":
            # Barge-in: 상담사 응답 중간에 고객이 끊고 새 질문
            print(f"  [Barge-in] {turn['description']}")

            # 상담사가 이전 맥락으로 길게 답변 시작 (일부러 긴 답변 유도)
            llm_conversation.append({"role": "user", "content": "이메일이 안 돼요"})
            long_prompt = system_prompt + "\n\n이 질문에 대해 가능한 여러 원인을 상세하게 설명해주세요. 4-5문장으로 답변하세요."
            long_response = client.messages.create(
                model=persona.llm_model, max_tokens=300,
                system=long_prompt, messages=llm_conversation,
            ).content[0].text
            llm_conversation.append({"role": "assistant", "content": long_response})
            print(f"  상담사 (길게): {long_response}")

            # 상담사 음성 생성 → 40%만 재생 (나머지 끊김)
            full_response_audio = typecast_tts(PERSONA_VOICE, long_response)
            truncated_audio = truncate_audio(full_response_audio, keep_ratio=0.35)
            audio_parts.append(truncated_audio)
            audio_parts.append(silence_150)  # 끊기고 바로
            time.sleep(0.5)

            # 고객 끼어들기
            interrupt_text = turn["customer_interrupt"]
            print(f"  고객 (끼어들기): {interrupt_text}")

            interrupt_audio = typecast_tts(CUSTOMER_VOICE, interrupt_text)
            audio_parts.append(interrupt_audio)
            audio_parts.append(silence_300)
            time.sleep(0.5)

            # STT
            stt_text = stt_recognize(interrupt_audio) if interrupt_audio else interrupt_text
            print(f"  STT: {stt_text}")

            # 상담사가 끼어든 내용에 맞게 새로 응답
            llm_conversation.append({"role": "user", "content": f"[고객이 끊고 말함] {stt_text or interrupt_text}"})
            new_response = llm_respond(client, system_prompt, llm_conversation, persona.knowledge_base_id, persona.llm_model)
            llm_conversation.append({"role": "assistant", "content": new_response})
            print(f"  상담사 (새 응답): {new_response}")

            new_response_audio = typecast_tts(PERSONA_VOICE, new_response)
            audio_parts.append(new_response_audio)
            audio_parts.append(silence_500)
            time.sleep(0.5)

    # 아웃트로
    outro = typecast_tts(CUSTOMER_VOICE, "Barge-in 테스트가 끝났습니다.")
    audio_parts.append(outro)

    # MP3 합치기
    output_path = Path(__file__).resolve().parent.parent / "e2e_bargein_test.mp3"
    print(f"\n  녹음 합치기...")
    concat_audio(audio_parts, str(output_path))
    print(f"  파일: {output_path} ({output_path.stat().st_size // 1024}KB)")

    # 평가
    print(f"\n  평가 Bot 분석 중...")
    conv_text = "\n".join(
        f"{'고객' if m['role']=='user' else '상담사'}: {m['content']}"
        for m in llm_conversation
    )

    eval_resp = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        messages=[{"role": "user", "content": f"""AICC Barge-in 테스트를 평가하세요.

## 시나리오
고객(여성)이 이메일 문제로 IT 상담. 상담사(남성)가 수신 문제로 오해하고 답변하던 중 고객이 끼어들어 정정.

## 대화
{conv_text}

## 평가 기준
1. Barge-in 후 상담사가 고객의 정정을 정확히 반영했는가
2. 끊긴 후 자연스럽게 대화를 이어갔는가
3. 최종적으로 고객 문제가 해결되었는가

JSON만 출력:
{{"bargein_handling": {{"score": 1-10, "reason": ""}}, "recovery": {{"score": 1-10, "reason": ""}}, "goal_achieved": true/false, "summary": ""}}"""}],
    )

    try:
        text = eval_resp.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        evaluation = json.loads(text.strip())
    except:
        evaluation = {"raw": eval_resp.content[0].text[:300]}

    print(f"  평가: {json.dumps(evaluation, ensure_ascii=False, indent=2)}")
    print(f"\n{'='*70}")


def main():
    asyncio.run(run_bargein_test())


if __name__ == "__main__":
    main()
