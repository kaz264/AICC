"""AICC E2E 음성 파이프라인 테스트

전체 파이프라인을 자동으로 테스트합니다:
  고객 Bot (TTS) → AICC Bot (STT → RAG → LLM → TTS) → 고객 Bot (STT → 다음질문)
  + 평가 Bot (대화 품질 + 지연시간 + 종합 리포트)

사용법:
    python -m tests.e2e_voice_test
    python -m tests.e2e_voice_test --scenario restaurant_reservation
    python -m tests.e2e_voice_test --all
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
import io
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from anthropic import Anthropic
from google.cloud import texttospeech, speech
from google.oauth2 import service_account

from backend import config
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge

FFMPEG = r"C:\Users\minsu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"


# ── 데이터 모델 ──

@dataclass
class TurnMetrics:
    turn: int
    role: str  # "customer" | "aicc"
    text_sent: str = ""
    text_received_stt: str = ""
    text_response: str = ""
    tts_ms: float = 0
    stt_ms: float = 0
    rag_ms: float = 0
    llm_ms: float = 0
    response_tts_ms: float = 0
    total_pipeline_ms: float = 0
    audio_bytes: bytes = field(default=b"", repr=False)


@dataclass
class E2ETestResult:
    scenario_id: str
    persona_name: str
    turns: list[TurnMetrics] = field(default_factory=list)
    full_audio: bytes = field(default=b"", repr=False)
    evaluation: dict = field(default_factory=dict)
    total_duration_ms: float = 0


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


# ── GCP 클라이언트 ──

def get_gcp_credentials():
    creds_path = Path(config.GOOGLE_APPLICATION_CREDENTIALS)
    if not creds_path.is_absolute():
        creds_path = Path(__file__).resolve().parent.parent / creds_path
    return service_account.Credentials.from_service_account_file(str(creds_path))


def get_tts_client():
    return texttospeech.TextToSpeechClient(credentials=get_gcp_credentials())


def get_stt_client():
    return speech.SpeechClient(credentials=get_gcp_credentials())


# ── TTS (Typecast) ──

TYPECAST_VOICES = {
    "customer": "tc_686dc43ebd6351e06ee64d74",       # Wonwoo (남)
    "보험 상담사": "tc_68f9c6a72f0f04a417bb136f",      # Moonjung (여)
    "레스토랑 예약 안내": "tc_68537c9420b646f2176890ba", # Seojin (여)
    "IT 기술지원": "tc_68662745779b66ba84fc4d84",       # Seheon (남)
}


def tts_typecast(voice_id: str, text: str) -> tuple[bytes, float]:
    """Typecast TTS → (audio_bytes, latency_ms)"""
    text = clean_text(text)
    start = time.perf_counter()
    resp = httpx.post(
        f"https://api.typecast.ai/v1/text-to-speech/{voice_id}" if "/" in voice_id
        else "https://api.typecast.ai/v1/text-to-speech",
        headers={"X-API-KEY": config.TYPECAST_API_KEY, "Content-Type": "application/json"},
        json={
            "voice_id": voice_id,
            "text": text,
            "model": "ssfm-v30",
            "language": "kor",
            "output": {"format": "wav"},
        },
        timeout=30,
    )
    elapsed = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        print(f"    [TTS Error] {resp.status_code}: {resp.text[:100]}")
        return b"", elapsed
    return resp.content, elapsed


# ── STT (Google) ──

def stt_google(audio_bytes: bytes, stt_client) -> tuple[str, float]:
    """Google STT → (transcribed_text, latency_ms)"""
    start = time.perf_counter()

    # WAV인지 확인
    is_wav = audio_bytes[:4] == b'RIFF'

    audio = speech.RecognitionAudio(content=audio_bytes)
    stt_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16 if is_wav
        else speech.RecognitionConfig.AudioEncoding.MP3,
        language_code="ko-KR",
        model="latest_long",
        enable_automatic_punctuation=True,
    )
    if is_wav:
        # WAV에서 sample rate 추출
        import struct
        sample_rate = struct.unpack('<I', audio_bytes[24:28])[0]
        stt_config.sample_rate_hertz = sample_rate

    response = stt_client.recognize(config=stt_config, audio=audio)
    elapsed = (time.perf_counter() - start) * 1000

    text = ""
    for result in response.results:
        text += result.alternatives[0].transcript

    return text, elapsed


# ── LLM (Claude) ──

def llm_respond(
    anthropic_client: Anthropic,
    system_prompt: str,
    conversation: list[dict],
    kb_id: str | None,
    model: str,
) -> tuple[str, float, float]:
    """LLM 응답 → (response_text, rag_ms, llm_ms)"""
    # RAG
    rag_start = time.perf_counter()
    rag_context = ""
    if kb_id:
        user_msgs = [m["content"] for m in conversation if m["role"] == "user"]
        query = " ".join(user_msgs[-2:]) if len(user_msgs) >= 2 else user_msgs[-1] if user_msgs else ""
        if query:
            results = search_knowledge(kb_id, query, n_results=3)
            if results:
                rag_context = "\n\n## 참고 지식 (이 정보를 기반으로 구체적으로 답변하세요)\n" + "\n\n".join(results)
    rag_ms = (time.perf_counter() - rag_start) * 1000

    # LLM
    llm_start = time.perf_counter()
    enhanced_prompt = system_prompt + rag_context + """

## 음성 출력용 규칙
- 물결표를 사용하지 마세요.
- 이모지나 특수문자를 사용하지 마세요.
- 자연스러운 대화체로 말하세요.
"""
    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=200,
        system=enhanced_prompt,
        messages=conversation,
    )
    llm_ms = (time.perf_counter() - llm_start) * 1000

    return resp.content[0].text, rag_ms, llm_ms


# ── 무음 생성 ──

def generate_silence_wav(duration_ms: int) -> bytes:
    """무음 WAV 생성"""
    google_tts = get_tts_client()
    ssml = f'<speak><break time="{duration_ms}ms"/></speak>'
    response = google_tts.synthesize_speech(
        input=texttospeech.SynthesisInput(ssml=ssml),
        voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Neural2-A"),
        audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
    )
    return response.audio_content


# ── 오디오 합치기 ──

def concat_audio(parts: list[bytes], output_path: str):
    """ffmpeg로 WAV+MP3 혼합 오디오 합치기"""
    tmp_dir = Path(tempfile.mkdtemp())
    part_files = []
    for i, part in enumerate(parts):
        if not part:
            continue
        is_wav = part[:4] == b'RIFF'
        ext = ".wav" if is_wav else ".mp3"
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

def evaluate_conversation(
    anthropic_client: Anthropic,
    scenario: dict,
    turns: list[TurnMetrics],
) -> dict:
    """대화 품질 + 지연시간 종합 평가"""

    # 대화 내용 구성
    conv_text = ""
    for t in turns:
        if t.role == "customer":
            conv_text += f"고객 (원문): {t.text_sent}\n"
            if t.text_received_stt:
                conv_text += f"고객 (STT 인식): {t.text_received_stt}\n"
        else:
            conv_text += f"상담사: {t.text_response}\n"
            conv_text += f"  [지연: TTS={t.tts_ms:.0f}ms, STT={t.stt_ms:.0f}ms, RAG={t.rag_ms:.0f}ms, LLM={t.llm_ms:.0f}ms, 응답TTS={t.response_tts_ms:.0f}ms, 전체={t.total_pipeline_ms:.0f}ms]\n"

    # 지연 통계
    aicc_turns = [t for t in turns if t.role == "aicc"]
    avg_pipeline = sum(t.total_pipeline_ms for t in aicc_turns) / len(aicc_turns) if aicc_turns else 0
    max_pipeline = max((t.total_pipeline_ms for t in aicc_turns), default=0)

    # STT 정확도 (고객 발화)
    customer_turns = [t for t in turns if t.role == "customer" and t.text_received_stt]
    stt_accuracy_note = ""
    for ct in customer_turns:
        if ct.text_sent and ct.text_received_stt:
            stt_accuracy_note += f"  원문: {ct.text_sent}\n  인식: {ct.text_received_stt}\n\n"

    eval_prompt = f"""당신은 AICC(AI 컨택센터) 품질 평가 전문가입니다.
아래 대화와 성능 데이터를 분석하여 종합 평가하세요.

## 시나리오
- 페르소나: {scenario['persona_name']}
- 고객 프로필: {scenario['customer_profile']}
- 고객 목표: {scenario['customer_goal']}

## 대화 내용 (구간별 지연시간 포함)
{conv_text}

## STT 인식 정확도
{stt_accuracy_note if stt_accuracy_note else "STT 데이터 없음"}

## 지연시간 통계
- 평균 전체 파이프라인: {avg_pipeline:.0f}ms
- 최대 전체 파이프라인: {max_pipeline:.0f}ms
- 목표: 1500ms 이내

## 평가 기준
1. **대화 품질** (정확성, 톤, 공감, 간결성) — 각 10점
2. **STT 인식률** — 원문 대비 인식 정확도 10점
3. **응답 속도** — 목표 1.5초 기준 10점
4. **종합 서비스 품질** — 실제 고객이 받을 인상 10점

## 출력 형식 (JSON만)
{{
  "scores": {{
    "accuracy": {{"score": 1-10, "reason": "이유"}},
    "tone": {{"score": 1-10, "reason": "이유"}},
    "empathy": {{"score": 1-10, "reason": "이유"}},
    "conciseness": {{"score": 1-10, "reason": "이유"}},
    "stt_accuracy": {{"score": 1-10, "reason": "이유"}},
    "latency": {{"score": 1-10, "reason": "이유"}},
    "overall_experience": {{"score": 1-10, "reason": "이유"}}
  }},
  "goal_achieved": true/false,
  "avg_latency_ms": {avg_pipeline:.0f},
  "strengths": ["강점1", "강점2"],
  "improvements": ["개선점1", "개선점2"],
  "stt_issues": ["인식 오류 사례"],
  "summary": "한 줄 요약"
}}"""

    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": eval_prompt}],
    )

    try:
        text = resp.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        return {"error": "평가 파싱 실패", "raw": resp.content[0].text}


# ── E2E 테스트 실행 ──

async def run_e2e_test(scenario_id: str, verbose: bool = True) -> E2ETestResult:
    """단일 시나리오 E2E 테스트"""

    # 시나리오 로드
    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scenario:
        print(f"시나리오 '{scenario_id}'를 찾을 수 없습니다.")
        return E2ETestResult(scenario_id=scenario_id, persona_name="?")

    # 초기화
    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()

    personas = await list_personas()
    persona = next((p for p in personas if p.name == scenario["persona_name"]), None)
    if not persona:
        print(f"페르소나를 찾을 수 없습니다.")
        return E2ETestResult(scenario_id=scenario_id, persona_name=scenario["persona_name"])

    print(f"\n{'='*70}")
    print(f"E2E 테스트: {scenario_id} ({persona.name})")
    print(f"{'='*70}")

    # 클라이언트 초기화
    anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    stt_client = get_stt_client()
    system_prompt = build_system_prompt(persona)

    customer_voice = TYPECAST_VOICES["customer"]
    persona_voice = TYPECAST_VOICES.get(persona.name, customer_voice)

    result = E2ETestResult(scenario_id=scenario_id, persona_name=persona.name)
    llm_conversation: list[dict] = []  # LLM에 보낼 대화 이력
    audio_parts: list[bytes] = []

    # 무음
    silence_500 = generate_silence_wav(500)
    silence_800 = generate_silence_wav(800)

    follow_ups = scenario.get("follow_ups", [])
    follow_up_idx = 0
    customer_texts = [scenario["opening_message"]] + follow_ups + ["네, 감사합니다. 도움이 됐어요."]

    test_start = time.perf_counter()

    for turn_num, customer_text in enumerate(customer_texts):
        print(f"\n  ── 턴 {turn_num + 1} ──")

        # ① 고객 TTS: 텍스트 → 음성
        print(f"  고객: {customer_text}")
        customer_audio, customer_tts_ms = tts_typecast(customer_voice, customer_text)
        print(f"    TTS: {customer_tts_ms:.0f}ms ({len(customer_audio)//1024}KB)")

        customer_turn = TurnMetrics(
            turn=turn_num, role="customer",
            text_sent=customer_text, tts_ms=customer_tts_ms,
            audio_bytes=customer_audio,
        )

        # ② AICC STT: 음성 → 텍스트 (실제 파이프라인 시뮬레이션)
        pipeline_start = time.perf_counter()

        if customer_audio:
            stt_text, stt_ms = stt_google(customer_audio, stt_client)
            customer_turn.text_received_stt = stt_text
            customer_turn.stt_ms = stt_ms
            print(f"    STT 인식: \"{stt_text}\" ({stt_ms:.0f}ms)")
        else:
            stt_text = customer_text  # TTS 실패 시 원문 사용
            stt_ms = 0

        result.turns.append(customer_turn)
        audio_parts.append(customer_audio)
        audio_parts.append(silence_500)

        # ③ AICC LLM: 인식된 텍스트로 응답 생성
        llm_conversation.append({"role": "user", "content": stt_text or customer_text})

        response_text, rag_ms, llm_ms = llm_respond(
            anthropic_client, system_prompt, llm_conversation,
            persona.knowledge_base_id, persona.llm_model,
        )
        print(f"  상담사: {response_text}")
        print(f"    RAG: {rag_ms:.0f}ms | LLM: {llm_ms:.0f}ms")

        # ④ AICC TTS: 응답 텍스트 → 음성
        response_audio, response_tts_ms = tts_typecast(persona_voice, response_text)
        print(f"    응답 TTS: {response_tts_ms:.0f}ms ({len(response_audio)//1024}KB)")

        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

        aicc_turn = TurnMetrics(
            turn=turn_num, role="aicc",
            text_response=response_text,
            stt_ms=stt_ms, rag_ms=rag_ms, llm_ms=llm_ms,
            response_tts_ms=response_tts_ms,
            total_pipeline_ms=pipeline_ms,
            audio_bytes=response_audio,
        )
        result.turns.append(aicc_turn)
        llm_conversation.append({"role": "assistant", "content": response_text})

        audio_parts.append(response_audio)
        audio_parts.append(silence_800)

        print(f"    전체 파이프라인: {pipeline_ms:.0f}ms")

        time.sleep(0.5)  # rate limit

    result.total_duration_ms = (time.perf_counter() - test_start) * 1000

    # 녹음 파일 생성
    output_path = Path(__file__).resolve().parent.parent / f"e2e_{scenario_id}.mp3"
    print(f"\n  녹음 파일 생성 중...")
    concat_audio(audio_parts, str(output_path))
    print(f"  저장: {output_path} ({output_path.stat().st_size // 1024}KB)")

    # 평가
    print(f"\n  평가 Bot 분석 중...")
    result.evaluation = evaluate_conversation(anthropic_client, scenario, result.turns)

    return result


def print_report(results: list[E2ETestResult]):
    """종합 리포트 출력"""
    print(f"\n{'='*70}")
    print(f"AICC E2E 종합 리포트")
    print(f"{'='*70}")
    print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"시나리오: {len(results)}개\n")

    for r in results:
        ev = r.evaluation
        scores = ev.get("scores", {})

        print(f"  [{r.scenario_id}] {r.persona_name}")

        # 점수
        if scores:
            score_parts = []
            for k, v in scores.items():
                if isinstance(v, dict):
                    score_parts.append(f"{k}:{v.get('score', '?')}")
            print(f"    점수: {' | '.join(score_parts)}")

        # 지연
        aicc_turns = [t for t in r.turns if t.role == "aicc"]
        if aicc_turns:
            avg_lat = sum(t.total_pipeline_ms for t in aicc_turns) / len(aicc_turns)
            avg_stt = sum(t.stt_ms for t in aicc_turns) / len(aicc_turns)
            avg_rag = sum(t.rag_ms for t in aicc_turns) / len(aicc_turns)
            avg_llm = sum(t.llm_ms for t in aicc_turns) / len(aicc_turns)
            avg_tts = sum(t.response_tts_ms for t in aicc_turns) / len(aicc_turns)
            print(f"    지연: STT={avg_stt:.0f} + RAG={avg_rag:.0f} + LLM={avg_llm:.0f} + TTS={avg_tts:.0f} = {avg_lat:.0f}ms 평균")

        goal = ev.get("goal_achieved", "?")
        summary = ev.get("summary", "")
        print(f"    목표달성: {'O' if goal else 'X'} | {summary}")

        # STT 이슈
        stt_issues = ev.get("stt_issues", [])
        if stt_issues:
            for issue in stt_issues[:2]:
                print(f"    STT 이슈: {issue}")

        # 개선점
        improvements = ev.get("improvements", [])
        for imp in improvements[:2]:
            print(f"    개선: {imp}")
        print()

    # 전체 평균
    all_scores = []
    for r in results:
        overall = r.evaluation.get("scores", {}).get("overall_experience", {})
        if isinstance(overall, dict) and "score" in overall:
            all_scores.append(overall["score"])
    if all_scores:
        print(f"  종합 평균: {sum(all_scores)/len(all_scores):.1f}/10")
    print(f"{'='*70}")

    # JSON 리포트 저장
    report_path = Path(__file__).resolve().parent / "e2e_report.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "results": [],
    }
    for r in results:
        turns_data = []
        for t in r.turns:
            td = {
                "turn": t.turn, "role": t.role,
                "text_sent": t.text_sent, "text_received_stt": t.text_received_stt,
                "text_response": t.text_response,
                "tts_ms": t.tts_ms, "stt_ms": t.stt_ms,
                "rag_ms": t.rag_ms, "llm_ms": t.llm_ms,
                "response_tts_ms": t.response_tts_ms,
                "total_pipeline_ms": t.total_pipeline_ms,
            }
            turns_data.append(td)
        report_data["results"].append({
            "scenario_id": r.scenario_id,
            "persona_name": r.persona_name,
            "turns": turns_data,
            "evaluation": r.evaluation,
            "total_duration_ms": r.total_duration_ms,
        })

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n리포트 저장: {report_path}")


async def main_async(scenario_id: str | None, run_all: bool):
    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    if run_all:
        ids = [s["id"] for s in scenarios]
    elif scenario_id:
        ids = [scenario_id]
    else:
        ids = ["restaurant_reservation"]

    results = []
    for sid in ids:
        r = await run_e2e_test(sid, verbose=True)
        results.append(r)

    print_report(results)


def main():
    parser = argparse.ArgumentParser(description="AICC E2E 음성 파이프라인 테스트")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="전체 시나리오 실행")
    args = parser.parse_args()
    asyncio.run(main_async(args.scenario, args.all))


if __name__ == "__main__":
    main()
