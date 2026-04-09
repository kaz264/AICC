"""AICC E2E 스트리밍 파이프라인 테스트

배치 방식이 아닌 실제 Pipecat과 유사한 스트리밍 방식으로 테스트합니다:
  - STT: Google Streaming Recognition (중간결과 즉시 활용)
  - LLM: Claude Streaming (첫 토큰 시간 측정)
  - TTS: 첫 문장만 먼저 합성 (나머지는 백그라운드)

측정 지표:
  - TTFR (Time to First Response): 고객 발화 종료 → 상담사 음성 시작
  - 이게 실제 고객이 느끼는 지연시간

사용법:
    python -m tests.e2e_streaming_test
    python -m tests.e2e_streaming_test --scenario insurance_claim
    python -m tests.e2e_streaming_test --all
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
from datetime import datetime
from dataclasses import dataclass, field
from anthropic import Anthropic
from google.cloud import texttospeech, speech
from google.oauth2 import service_account

from backend import config
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge

FFMPEG = r"C:\Users\minsu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"


@dataclass
class StreamingTurnMetrics:
    turn: int
    role: str
    text_sent: str = ""
    text_stt: str = ""
    text_response: str = ""
    first_sentence: str = ""
    # 스트리밍 지연 측정
    stt_streaming_ms: float = 0       # STT 스트리밍 최종 결과까지
    stt_interim_ms: float = 0         # STT 첫 중간결과까지
    rag_ms: float = 0
    llm_ttft_ms: float = 0            # LLM 첫 토큰 시간
    llm_first_sentence_ms: float = 0  # LLM 첫 문장 완성까지
    llm_total_ms: float = 0
    tts_first_chunk_ms: float = 0     # TTS 첫 문장 합성 시간
    tts_total_ms: float = 0
    # 핵심 지표
    ttfr_ms: float = 0                # Time to First Response (체감 지연)
    total_pipeline_ms: float = 0


# ── GCP 클라이언트 ──

def get_gcp_credentials():
    creds_path = Path(config.GOOGLE_APPLICATION_CREDENTIALS)
    if not creds_path.is_absolute():
        creds_path = Path(__file__).resolve().parent.parent / creds_path
    return service_account.Credentials.from_service_account_file(str(creds_path))


# ── 텍스트 정제 ──

def clean_text(text: str) -> str:
    text = text.replace("~", "").replace("♪", "").replace("♥", "").replace("★", "")
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"-\s+", "", text)
    text = text.replace("(", ", ").replace(")", ",")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_first_sentence(text: str) -> str:
    """첫 문장 추출 (TTS 즉시 재생용)"""
    for sep in [".", "!", "?"]:
        idx = text.find(sep)
        if idx > 0:
            return text[:idx + 1]
    # 구분자 없으면 쉼표 기준
    idx = text.find(",")
    if idx > 5:
        return text[:idx + 1]
    return text


# ── STT 스트리밍 ──

def stt_streaming(audio_bytes: bytes) -> tuple[str, float, float]:
    """Google STT 스트리밍 → (text, interim_ms, final_ms)

    오디오를 청크 단위로 보내 스트리밍 인식을 시뮬레이션합니다.
    """
    credentials = get_gcp_credentials()
    client = speech.SpeechClient(credentials=credentials)

    is_wav = audio_bytes[:4] == b'RIFF'

    stt_config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16 if is_wav
            else speech.RecognitionConfig.AudioEncoding.MP3,
            language_code="ko-KR",
            model="latest_long",
            enable_automatic_punctuation=True,
            sample_rate_hertz=24000 if is_wav else None,
        ),
        interim_results=True,
    )

    # 오디오를 청크로 분할하여 스트리밍
    if is_wav:
        audio_data = audio_bytes[44:]  # WAV 헤더 스킵
    else:
        audio_data = audio_bytes

    chunk_size = 4096
    chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]

    def request_generator():
        yield speech.StreamingRecognizeRequest(streaming_config=stt_config)
        for chunk in chunks:
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    start = time.perf_counter()
    interim_time = None
    final_text = ""

    try:
        responses = client.streaming_recognize(requests=request_generator())
        for response in responses:
            for result in response.results:
                if not interim_time and result.alternatives:
                    interim_time = (time.perf_counter() - start) * 1000
                if result.is_final:
                    final_text = result.alternatives[0].transcript
    except Exception as e:
        print(f"    [STT Streaming Error] {e}")
        # 폴백: 배치 STT
        return stt_batch_fallback(audio_bytes, client, start)

    final_ms = (time.perf_counter() - start) * 1000
    interim_ms = interim_time or final_ms

    return final_text, interim_ms, final_ms


def stt_batch_fallback(audio_bytes: bytes, client, start_time) -> tuple[str, float, float]:
    """스트리밍 실패 시 배치 폴백"""
    is_wav = audio_bytes[:4] == b'RIFF'
    import struct
    sample_rate = struct.unpack('<I', audio_bytes[24:28])[0] if is_wav else 24000

    stt_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16 if is_wav
        else speech.RecognitionConfig.AudioEncoding.MP3,
        language_code="ko-KR",
        model="latest_long",
        sample_rate_hertz=sample_rate if is_wav else None,
        enable_automatic_punctuation=True,
    )
    audio = speech.RecognitionAudio(content=audio_bytes)
    response = client.recognize(config=stt_config, audio=audio)
    elapsed = (time.perf_counter() - start_time) * 1000

    text = ""
    for result in response.results:
        text += result.alternatives[0].transcript
    return text, elapsed, elapsed


# ── LLM 스트리밍 ──

def llm_streaming(
    anthropic_client: Anthropic,
    system_prompt: str,
    conversation: list[dict],
    kb_id: str | None,
    model: str,
) -> tuple[str, str, float, float, float, float]:
    """Claude 스트리밍 → (full_text, first_sentence, rag_ms, ttft_ms, first_sentence_ms, total_ms)"""

    # RAG
    rag_start = time.perf_counter()
    rag_context = ""
    if kb_id:
        user_msgs = [m["content"] for m in conversation if m["role"] == "user"]
        query = " ".join(user_msgs[-2:]) if len(user_msgs) >= 2 else user_msgs[-1] if user_msgs else ""
        if query:
            results = search_knowledge(kb_id, query, n_results=3)
            if results:
                rag_context = "\n\n## 참고 지식\n" + "\n\n".join(results)
    rag_ms = (time.perf_counter() - rag_start) * 1000

    enhanced_prompt = system_prompt + rag_context + """

## 음성 출력용 규칙
- 물결표를 사용하지 마세요.
- 이모지나 특수문자를 사용하지 마세요.
- 자연스러운 대화체로 말하세요.
"""

    # LLM 스트리밍
    llm_start = time.perf_counter()
    ttft = None
    first_sentence_time = None
    full_text = ""
    first_sentence = ""
    found_first_sentence = False

    with anthropic_client.messages.stream(
        model=model,
        max_tokens=200,
        system=enhanced_prompt,
        messages=conversation,
    ) as stream:
        for text in stream.text_stream:
            if ttft is None:
                ttft = (time.perf_counter() - llm_start) * 1000
            full_text += text

            # 첫 문장 완성 감지
            if not found_first_sentence:
                for sep in [".", "!", "?", ","]:
                    if sep in full_text and len(full_text) > 5:
                        idx = full_text.index(sep)
                        first_sentence = full_text[:idx + 1]
                        first_sentence_time = (time.perf_counter() - llm_start) * 1000
                        found_first_sentence = True
                        break

    total_ms = (time.perf_counter() - llm_start) * 1000
    ttft_ms = ttft or total_ms
    first_sentence_ms = first_sentence_time or total_ms
    if not first_sentence:
        first_sentence = full_text

    return full_text, first_sentence, rag_ms, ttft_ms, first_sentence_ms, total_ms


# ── TTS ──

TYPECAST_VOICES = {
    "customer": "tc_686dc43ebd6351e06ee64d74",
    "보험 상담사": "tc_68f9c6a72f0f04a417bb136f",
    "레스토랑 예약 안내": "tc_68537c9420b646f2176890ba",
    "IT 기술지원": "tc_68662745779b66ba84fc4d84",
}


def tts_typecast(voice_id: str, text: str) -> tuple[bytes, float]:
    text = clean_text(text)
    start = time.perf_counter()
    resp = httpx.post(
        "https://api.typecast.ai/v1/text-to-speech",
        headers={"X-API-KEY": config.TYPECAST_API_KEY, "Content-Type": "application/json"},
        json={"voice_id": voice_id, "text": text, "model": "ssfm-v30", "language": "kor", "output": {"format": "wav"}},
        timeout=30,
    )
    elapsed = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        return b"", elapsed
    return resp.content, elapsed


# ── 무음 + 오디오 합치기 ──

def generate_silence(duration_ms: int) -> bytes:
    google_tts = texttospeech.TextToSpeechClient(credentials=get_gcp_credentials())
    ssml = f'<speak><break time="{duration_ms}ms"/></speak>'
    response = google_tts.synthesize_speech(
        input=texttospeech.SynthesisInput(ssml=ssml),
        voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Neural2-A"),
        audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
    )
    return response.audio_content


def concat_audio(parts: list[bytes], output_path: str):
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


# ── 평가 Bot ──

def evaluate(anthropic_client, scenario, turns):
    conv_text = ""
    for t in turns:
        if t.role == "customer":
            conv_text += f"고객: {t.text_sent}\n"
            if t.text_stt:
                conv_text += f"  [STT 인식: \"{t.text_stt}\" | 중간결과: {t.stt_interim_ms:.0f}ms, 최종: {t.stt_streaming_ms:.0f}ms]\n"
        else:
            conv_text += f"상담사: {t.text_response}\n"
            conv_text += f"  [TTFR={t.ttfr_ms:.0f}ms | STT={t.stt_streaming_ms:.0f} + RAG={t.rag_ms:.0f} + LLM_TTFT={t.llm_ttft_ms:.0f} + TTS_1st={t.tts_first_chunk_ms:.0f} = 스트리밍 합계]\n"
            conv_text += f"  [LLM 첫문장: \"{t.first_sentence}\" ({t.llm_first_sentence_ms:.0f}ms)]\n"

    aicc_turns = [t for t in turns if t.role == "aicc"]
    avg_ttfr = sum(t.ttfr_ms for t in aicc_turns) / len(aicc_turns) if aicc_turns else 0

    eval_prompt = f"""AICC 스트리밍 파이프라인 E2E 테스트를 평가하세요.

## 시나리오: {scenario['persona_name']}
- 고객: {scenario['customer_profile']}
- 목표: {scenario['customer_goal']}

## 대화 + 성능 데이터
{conv_text}

## 핵심 지표
- 평균 TTFR (체감 응답 지연): {avg_ttfr:.0f}ms
- 목표: TTFR 1500ms 이내

## 평가 (JSON만 출력)
{{
  "scores": {{
    "accuracy": {{"score": 1-10, "reason": ""}},
    "tone": {{"score": 1-10, "reason": ""}},
    "empathy": {{"score": 1-10, "reason": ""}},
    "conciseness": {{"score": 1-10, "reason": ""}},
    "stt_accuracy": {{"score": 1-10, "reason": ""}},
    "latency_ttfr": {{"score": 1-10, "reason": "TTFR {avg_ttfr:.0f}ms 기준"}},
    "overall_experience": {{"score": 1-10, "reason": ""}}
  }},
  "goal_achieved": true/false,
  "avg_ttfr_ms": {avg_ttfr:.0f},
  "strengths": [],
  "improvements": [],
  "summary": ""
}}"""

    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1500,
        messages=[{"role": "user", "content": eval_prompt}],
    )
    try:
        text = resp.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except:
        return {"error": "파싱 실패", "raw": resp.content[0].text[:500]}


# ── 메인 테스트 ──

async def run_streaming_test(scenario_id: str) -> dict:
    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scenario:
        print(f"시나리오 '{scenario_id}'를 찾을 수 없습니다.")
        return {}

    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()

    personas = await list_personas()
    persona = next((p for p in personas if p.name == scenario["persona_name"]), None)
    if not persona:
        return {}

    print(f"\n{'='*70}")
    print(f"E2E 스트리밍 테스트: {scenario_id} ({persona.name})")
    print(f"{'='*70}")

    anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system_prompt = build_system_prompt(persona)
    customer_voice = TYPECAST_VOICES["customer"]
    persona_voice = TYPECAST_VOICES.get(persona.name, customer_voice)

    turns: list[StreamingTurnMetrics] = []
    llm_conversation: list[dict] = []
    audio_parts: list[bytes] = []
    silence = generate_silence(500)

    customer_texts = [scenario["opening_message"]] + scenario.get("follow_ups", []) + ["네, 감사합니다. 도움이 됐어요."]

    for turn_num, customer_text in enumerate(customer_texts):
        print(f"\n  ── 턴 {turn_num + 1}/{len(customer_texts)} ──")

        # ① 고객 TTS
        print(f"  고객: {customer_text}")
        customer_audio, customer_tts_ms = tts_typecast(customer_voice, customer_text)

        customer_turn = StreamingTurnMetrics(turn=turn_num, role="customer", text_sent=customer_text)
        turns.append(customer_turn)
        audio_parts.append(customer_audio)
        audio_parts.append(silence)

        # ② STT 스트리밍
        pipeline_start = time.perf_counter()

        if customer_audio:
            stt_text, stt_interim_ms, stt_final_ms = stt_streaming(customer_audio)
            print(f"    STT: \"{stt_text}\" (중간결과: {stt_interim_ms:.0f}ms, 최종: {stt_final_ms:.0f}ms)")
        else:
            stt_text, stt_interim_ms, stt_final_ms = customer_text, 0, 0

        customer_turn.text_stt = stt_text
        customer_turn.stt_interim_ms = stt_interim_ms
        customer_turn.stt_streaming_ms = stt_final_ms

        # ③ LLM 스트리밍 (STT 최종결과 사용)
        llm_conversation.append({"role": "user", "content": stt_text or customer_text})

        full_text, first_sentence, rag_ms, ttft_ms, first_sent_ms, llm_total_ms = llm_streaming(
            anthropic_client, system_prompt, llm_conversation,
            persona.knowledge_base_id, persona.llm_model,
        )
        print(f"  상담사: {full_text}")
        print(f"    RAG: {rag_ms:.0f}ms | LLM TTFT: {ttft_ms:.0f}ms | 첫문장: {first_sent_ms:.0f}ms | 전체: {llm_total_ms:.0f}ms")
        print(f"    첫 문장: \"{first_sentence}\"")

        # ④ TTS — 첫 문장 먼저 합성 (스트리밍 시뮬레이션)
        first_audio, first_tts_ms = tts_typecast(persona_voice, first_sentence)
        full_audio, full_tts_ms = tts_typecast(persona_voice, full_text)
        print(f"    TTS 첫문장: {first_tts_ms:.0f}ms | TTS 전체: {full_tts_ms:.0f}ms")

        # ⑤ TTFR 계산 (스트리밍 기준: STT중간결과 + RAG + LLM첫문장 + TTS첫문장)
        # 실제 Pipecat에서는 STT 중간결과로 LLM을 미리 시작할 수 있지만,
        # 보수적으로 STT 최종결과 기준으로 계산
        ttfr = stt_final_ms + rag_ms + first_sent_ms + first_tts_ms
        # 낙관적 TTFR (STT 중간결과 + LLM TTFT + TTS 첫문장)
        ttfr_optimistic = stt_interim_ms + rag_ms + ttft_ms + first_tts_ms

        total_pipeline = (time.perf_counter() - pipeline_start) * 1000

        print(f"    TTFR (보수적): {ttfr:.0f}ms | TTFR (낙관적): {ttfr_optimistic:.0f}ms")

        aicc_turn = StreamingTurnMetrics(
            turn=turn_num, role="aicc",
            text_response=full_text, first_sentence=first_sentence,
            text_stt=stt_text,
            stt_streaming_ms=stt_final_ms, stt_interim_ms=stt_interim_ms,
            rag_ms=rag_ms, llm_ttft_ms=ttft_ms,
            llm_first_sentence_ms=first_sent_ms, llm_total_ms=llm_total_ms,
            tts_first_chunk_ms=first_tts_ms, tts_total_ms=full_tts_ms,
            ttfr_ms=ttfr, total_pipeline_ms=total_pipeline,
        )
        turns.append(aicc_turn)
        llm_conversation.append({"role": "assistant", "content": full_text})

        audio_parts.append(full_audio or first_audio)
        audio_parts.append(silence)

        time.sleep(0.5)

    # 녹음 파일
    output_path = Path(__file__).resolve().parent.parent / f"e2e_streaming_{scenario_id}.mp3"
    concat_audio(audio_parts, str(output_path))
    print(f"\n  녹음: {output_path} ({output_path.stat().st_size // 1024}KB)")

    # 평가
    print(f"\n  평가 Bot 분석 중...")
    evaluation = evaluate(anthropic_client, scenario, turns)

    # 리포트
    aicc_turns = [t for t in turns if t.role == "aicc"]
    print(f"\n{'─'*70}")
    print(f"  스트리밍 지연 요약")
    print(f"{'─'*70}")
    print(f"  {'턴':>3} | {'STT':>7} | {'RAG':>6} | {'LLM_TTFT':>9} | {'LLM_1st':>8} | {'TTS_1st':>8} | {'TTFR':>7}")
    for t in aicc_turns:
        print(f"  {t.turn+1:>3} | {t.stt_streaming_ms:>6.0f}ms | {t.rag_ms:>5.0f}ms | {t.llm_ttft_ms:>8.0f}ms | {t.llm_first_sentence_ms:>7.0f}ms | {t.tts_first_chunk_ms:>7.0f}ms | {t.ttfr_ms:>6.0f}ms")

    avg_ttfr = sum(t.ttfr_ms for t in aicc_turns) / len(aicc_turns) if aicc_turns else 0
    avg_stt = sum(t.stt_streaming_ms for t in aicc_turns) / len(aicc_turns) if aicc_turns else 0
    avg_rag = sum(t.rag_ms for t in aicc_turns) / len(aicc_turns) if aicc_turns else 0
    avg_ttft = sum(t.llm_ttft_ms for t in aicc_turns) / len(aicc_turns) if aicc_turns else 0
    avg_1st_sent = sum(t.llm_first_sentence_ms for t in aicc_turns) / len(aicc_turns) if aicc_turns else 0
    avg_tts_1st = sum(t.tts_first_chunk_ms for t in aicc_turns) / len(aicc_turns) if aicc_turns else 0

    print(f"  {'AVG':>3} | {avg_stt:>6.0f}ms | {avg_rag:>5.0f}ms | {avg_ttft:>8.0f}ms | {avg_1st_sent:>7.0f}ms | {avg_tts_1st:>7.0f}ms | {avg_ttfr:>6.0f}ms")

    print(f"\n  배치 E2E 대비: {avg_ttfr:.0f}ms (스트리밍) vs ~7,856ms (배치)")
    if avg_ttfr < 1500:
        print(f"  → 목표 1.5초 이내 달성!")
    elif avg_ttfr < 3000:
        print(f"  → 1.5~3초. 추임새로 체감 개선 가능")
    else:
        print(f"  → 3초 초과. 추가 최적화 필요")

    # 평가 점수
    scores = evaluation.get("scores", {})
    if scores:
        print(f"\n  평가: ", end="")
        parts = [f"{k}:{v.get('score','?')}" for k, v in scores.items() if isinstance(v, dict)]
        print(" | ".join(parts))
    print(f"  요약: {evaluation.get('summary', '')}")

    # JSON 저장
    report = {
        "timestamp": datetime.now().isoformat(),
        "scenario_id": scenario_id,
        "persona": persona.name,
        "mode": "streaming",
        "avg_ttfr_ms": avg_ttfr,
        "avg_breakdown": {
            "stt_ms": avg_stt, "rag_ms": avg_rag,
            "llm_ttft_ms": avg_ttft, "llm_first_sentence_ms": avg_1st_sent,
            "tts_first_chunk_ms": avg_tts_1st,
        },
        "evaluation": evaluation,
    }
    report_path = Path(__file__).parent / "e2e_streaming_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  리포트: {report_path}")

    return report


async def main_async(scenario_id, run_all):
    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    ids = [s["id"] for s in scenarios] if run_all else [scenario_id or "restaurant_reservation"]

    for sid in ids:
        await run_streaming_test(sid)


def main():
    parser = argparse.ArgumentParser(description="AICC E2E 스트리밍 테스트")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args.scenario, args.all))


if __name__ == "__main__":
    main()
