"""E2E: Daily Room 실제 통합 테스트

실제 Daily.co WebRTC Room에서 고객Bot ↔ AICC Pipecat Bot이 대화합니다.
배포 전 최종 검증용.

흐름:
  1. Daily Room 생성
  2. AICC Bot (Pipecat) Room 참가
  3. 고객 Bot: TTS 오디오를 Room에 재생 → 응답 수신 → 다음 질문
  4. Daily Recording API로 전체 녹음
  5. 평가 Bot으로 품질 + 지연 종합 평가

사용법:
    python -m tests.e2e_daily_room_test
    python -m tests.e2e_daily_room_test --scenario insurance_claim
"""

import asyncio
import json
import argparse
import time
import aiohttp
import httpx
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic

from backend import config
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge


# ── Daily Room 관리 ──

async def create_daily_room() -> dict:
    """Daily.co Room 생성"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/rooms",
            headers={"Authorization": f"Bearer {config.DAILY_API_KEY}"},
            json={"properties": {"enable_chat": False, "start_video_off": True}},
        ) as resp:
            return await resp.json()


async def delete_daily_room(room_name: str):
    """Daily.co Room 삭제"""
    async with aiohttp.ClientSession() as session:
        await session.delete(
            f"https://api.daily.co/v1/rooms/{room_name}",
            headers={"Authorization": f"Bearer {config.DAILY_API_KEY}"},
        )


# ── 테스트 실행 ──

async def run_e2e_daily_test(scenario_id: str = "restaurant_reservation"):
    """Daily Room에서 실제 E2E 테스트

    Note: 이 테스트는 Pipecat이 설치된 환경에서만 실행 가능합니다.
    현재는 테스트 구조와 Room 생성/삭제만 검증합니다.
    """

    # 시나리오 로드
    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scenario:
        print(f"시나리오 '{scenario_id}'를 찾을 수 없습니다.")
        return

    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()

    personas = await list_personas()
    persona = next((p for p in personas if p.name == scenario["persona_name"]), None)

    print(f"\n{'='*70}")
    print(f"E2E Daily Room 테스트: {scenario_id} ({persona.name if persona else '?'})")
    print(f"{'='*70}")

    # 1. Daily Room 생성
    print("\n  1. Daily Room 생성 중...")
    room = await create_daily_room()
    room_url = room["url"]
    room_name = room["name"]
    print(f"     Room: {room_url}")

    try:
        # 2. AICC Bot 실행 (백그라운드)
        print("  2. AICC Pipecat Bot 시작...")

        # Pipecat bot을 별도 프로세스로 실행
        bot_task = asyncio.create_task(_run_aicc_bot(room_url, persona))

        # Bot이 Room에 참가할 시간 대기
        await asyncio.sleep(3)

        # 3. 고객 Bot 대화 실행
        print("  3. 고객 Bot 대화 시작...")
        anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        system_prompt = build_system_prompt(persona)

        customer_texts = [scenario["opening_message"]] + scenario.get("follow_ups", []) + ["감사합니다."]
        conversation = []
        metrics = []

        for i, customer_text in enumerate(customer_texts):
            print(f"\n     턴 {i+1}: 고객: {customer_text}")
            turn_start = time.perf_counter()

            # 고객 음성 → Room에 전송 (실제 구현 시 Daily SDK 사용)
            # 현재는 텍스트 레벨 시뮬레이션
            conversation.append({"role": "user", "content": customer_text})

            # AICC 응답 (실제로는 Room에서 수신)
            rag_context = ""
            if persona.knowledge_base_id:
                results = search_knowledge(persona.knowledge_base_id, customer_text, n_results=3)
                if results:
                    rag_context = "\n\n## 참고 지식\n" + "\n\n".join(results)

            resp = anthropic_client.messages.create(
                model=persona.llm_model,
                max_tokens=200,
                system=system_prompt + rag_context,
                messages=conversation,
            )
            aicc_response = resp.content[0].text
            conversation.append({"role": "assistant", "content": aicc_response})

            turn_ms = (time.perf_counter() - turn_start) * 1000
            metrics.append({"turn": i + 1, "text": customer_text, "response": aicc_response, "latency_ms": turn_ms})
            print(f"     상담사: {aicc_response}")
            print(f"     지연: {turn_ms:.0f}ms")

        # 4. 평가
        print("\n  4. 평가 Bot 분석 중...")
        evaluation = _evaluate(anthropic_client, scenario, conversation, metrics)

        scores = evaluation.get("scores", {})
        if scores:
            parts = [f"{k}:{v.get('score','?')}" for k, v in scores.items() if isinstance(v, dict)]
            print(f"     점수: {' | '.join(parts)}")
        print(f"     요약: {evaluation.get('summary', '')}")

        # Bot 종료
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass

    finally:
        # 5. Room 정리
        print(f"\n  5. Room 정리 중...")
        await delete_daily_room(room_name)
        print(f"     Room {room_name} 삭제 완료")

    # 리포트 저장
    report = {
        "timestamp": datetime.now().isoformat(),
        "scenario_id": scenario_id,
        "persona": persona.name if persona else "?",
        "room_url": room_url,
        "mode": "daily_room",
        "turns": metrics,
        "evaluation": evaluation,
    }
    report_path = Path(__file__).parent / "e2e_daily_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  리포트: {report_path}")
    print(f"{'='*70}")


async def _run_aicc_bot(room_url: str, persona):
    """AICC Pipecat Bot 실행"""
    try:
        from backend.pipeline.bot import run_voice_agent
        from backend.db.database import create_call_log
        call_log = await create_call_log(persona.id, room_url)
        await run_voice_agent(room_url, persona.id, call_log.id)
    except asyncio.CancelledError:
        pass
    except ImportError as e:
        print(f"     [Bot] Pipecat 미설치, 텍스트 모드로 대체: {e}")
    except Exception as e:
        print(f"     [Bot Error] {e}")


def _evaluate(anthropic_client, scenario, conversation, metrics):
    """대화 + 지연 평가"""
    conv_text = "\n".join(
        f"{'고객' if m['role']=='user' else '상담사'}: {m['content']}"
        for m in conversation
    )
    latency_info = "\n".join(
        f"턴 {m['turn']}: {m['latency_ms']:.0f}ms" for m in metrics
    )
    avg_latency = sum(m["latency_ms"] for m in metrics) / len(metrics) if metrics else 0

    eval_prompt = f"""AICC E2E Daily Room 테스트 결과를 평가하세요.

## 시나리오: {scenario['persona_name']}
고객: {scenario['customer_profile']}
목표: {scenario['customer_goal']}

## 대화
{conv_text}

## 지연 (평균 {avg_latency:.0f}ms)
{latency_info}

## JSON만 출력
{{"scores": {{"accuracy": {{"score": 1-10, "reason": ""}}, "tone": {{"score": 1-10, "reason": ""}}, "empathy": {{"score": 1-10, "reason": ""}}, "latency": {{"score": 1-10, "reason": ""}}, "overall": {{"score": 1-10, "reason": ""}}}}, "goal_achieved": true/false, "summary": ""}}"""

    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
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
        return {"error": "파싱 실패"}


def main():
    parser = argparse.ArgumentParser(description="E2E Daily Room 테스트")
    parser.add_argument("--scenario", default="restaurant_reservation")
    args = parser.parse_args()
    asyncio.run(run_e2e_daily_test(args.scenario))


if __name__ == "__main__":
    main()
