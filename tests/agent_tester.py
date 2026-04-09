"""AICC 자동 테스트 에이전트

고객 역할의 AI Agent가 AICC 페르소나와 텍스트 레벨에서 대화하고,
응답 품질을 자동 평가합니다.

사용법:
    python -m tests.agent_tester                    # 전체 시나리오 실행
    python -m tests.agent_tester --scenario insurance_claim  # 특정 시나리오
    python -m tests.agent_tester --verbose          # 대화 내용 출력
"""

import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic

from backend import config
from backend.db.database import init_db, list_personas, create_persona
from backend.models.persona import PersonaCreate
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge


# ── 테스트 에이전트 코어 ──

class CustomerAgent:
    """고객 역할 AI 에이전트"""

    def __init__(self, client: Anthropic, scenario: dict):
        self.client = client
        self.scenario = scenario
        self.system_prompt = f"""당신은 AICC 시스템을 테스트하는 가상 고객입니다.
아래 프로필대로 행동하세요.

## 고객 프로필
{scenario['customer_profile']}

## 목표
{scenario['customer_goal']}

## 행동 규칙
- 자연스러운 한국어 구어체를 사용하세요
- 한 번에 1-2문장만 말하세요
- 상담사의 답변에 따라 자연스럽게 반응하세요
- 미리 정해진 후속 질문이 있으면 적절한 타이밍에 사용하세요
- 목표가 달성되었거나 더 이상 질문이 없으면 "[대화종료]"를 포함해 마무리하세요
"""
        self.follow_up_index = 0

    def get_response(self, conversation_history: list[dict]) -> str:
        """대화 이력을 기반으로 고객 응답 생성"""
        # 후속 질문이 남아있으면 우선 사용
        follow_ups = self.scenario.get("follow_ups", [])
        if self.follow_up_index < len(follow_ups):
            response = follow_ups[self.follow_up_index]
            self.follow_up_index += 1
            return response

        # 후속 질문 소진 후 자연스럽게 마무리
        messages = conversation_history + [
            {"role": "user", "content": "후속 질문을 모두 했습니다. 자연스럽게 대화를 마무리하세요. 반드시 [대화종료]를 포함하세요."}
        ]
        resp = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=self.system_prompt,
            messages=messages,
        )
        return resp.content[0].text


class AIContactCenter:
    """AICC 페르소나 시뮬레이터 (텍스트 레벨)"""

    def __init__(self, client: Anthropic, persona_config: dict, system_prompt: str):
        self.client = client
        self.persona_config = persona_config
        self.system_prompt = system_prompt

    def get_response(self, conversation_history: list[dict]) -> str:
        """고객 메시지에 대한 AICC 응답 생성"""
        # RAG: 지식베이스에서 관련 정보 검색
        rag_context = ""
        kb_id = self.persona_config.get("knowledge_base_id")
        if kb_id and conversation_history:
            last_user_msg = conversation_history[-1]["content"]
            results = search_knowledge(kb_id, last_user_msg, n_results=3)
            if results:
                rag_context = "\n\n## 참고 지식\n" + "\n".join(f"- {r}" for r in results)

        full_prompt = self.system_prompt + rag_context

        resp = self.client.messages.create(
            model=self.persona_config.get("llm_model", "claude-sonnet-4-20250514"),
            max_tokens=300,
            system=full_prompt,
            messages=conversation_history,
        )
        return resp.content[0].text


class ConversationEvaluator:
    """대화 품질 자동 평가"""

    def __init__(self, client: Anthropic):
        self.client = client

    def evaluate(self, scenario: dict, conversation: list[dict]) -> dict:
        """대화 이력을 평가하고 점수 + 피드백 반환"""
        criteria = scenario["evaluation_criteria"]
        conv_text = "\n".join(
            f"{'고객' if m['role'] == 'user' else '상담사'}: {m['content']}"
            for m in conversation
        )

        eval_prompt = f"""당신은 AICC(AI 컨택센터) 품질 평가 전문가입니다.
아래 대화를 평가 기준에 따라 엄격하게 채점하세요.

## 시나리오
- 페르소나: {scenario['persona_name']}
- 고객 프로필: {scenario['customer_profile']}
- 고객 목표: {scenario['customer_goal']}

## 대화 내용
{conv_text}

## 평가 기준
{json.dumps(criteria, ensure_ascii=False, indent=2)}

## 출력 형식 (반드시 이 JSON 형식으로만 응답하세요)
{{
  "scores": {{
    "accuracy": {{"score": 1-10, "reason": "이유"}},
    "tone": {{"score": 1-10, "reason": "이유"}},
    "empathy": {{"score": 1-10, "reason": "이유"}},
    "conciseness": {{"score": 1-10, "reason": "이유"}}
  }},
  "overall_score": 1-10,
  "goal_achieved": true/false,
  "strengths": ["강점1", "강점2"],
  "improvements": ["개선점1", "개선점2"],
  "summary": "한 줄 요약"
}}"""

        resp = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": eval_prompt}],
        )

        try:
            text = resp.content[0].text
            # JSON 블록 추출
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            return {"error": "평가 파싱 실패", "raw": resp.content[0].text}


# ── 테스트 러너 ──

async def run_scenario(scenario: dict, verbose: bool = False) -> dict:
    """단일 시나리오 실행"""
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # DB 초기화 + 페르소나 로드
    await init_db()
    personas = await list_personas()
    persona = next(
        (p for p in personas if p.name == scenario["persona_name"]), None
    )

    if not persona:
        # 시드 데이터가 없으면 기본 페르소나 생성
        from backend.main import _seed_sample_personas
        await _seed_sample_personas()
        personas = await list_personas()
        persona = next(
            (p for p in personas if p.name == scenario["persona_name"]), None
        )

    if not persona:
        return {"error": f"페르소나 '{scenario['persona_name']}' 을 찾을 수 없습니다"}

    # 에이전트 초기화
    system_prompt = build_system_prompt(persona)
    aicc = AIContactCenter(
        client,
        persona_config={
            "llm_model": persona.llm_model,
            "knowledge_base_id": persona.knowledge_base_id,
        },
        system_prompt=system_prompt,
    )
    customer = CustomerAgent(client, scenario)
    evaluator = ConversationEvaluator(client)

    # 대화 실행
    conversation: list[dict] = []
    max_turns = len(scenario.get("follow_ups", [])) + 3  # 후속질문 + 여유 턴

    if verbose:
        print(f"\n{'='*60}")
        print(f"시나리오: {scenario['id']} ({scenario['persona_name']})")
        print(f"{'='*60}")

    # 첫 메시지: 고객이 먼저 말함
    customer_msg = scenario["opening_message"]
    conversation.append({"role": "user", "content": customer_msg})
    if verbose:
        print(f"\n고객: {customer_msg}")

    for turn in range(max_turns):
        # AICC 응답
        aicc_response = aicc.get_response(conversation)
        conversation.append({"role": "assistant", "content": aicc_response})
        if verbose:
            print(f"상담사: {aicc_response}")

        # 고객 응답
        customer_msg = customer.get_response(conversation)
        if verbose:
            print(f"고객: {customer_msg}")

        if "[대화종료]" in customer_msg:
            conversation.append({"role": "user", "content": customer_msg})
            break

        conversation.append({"role": "user", "content": customer_msg})

    # 평가
    if verbose:
        print(f"\n{'─'*60}")
        print("평가 중...")

    evaluation = evaluator.evaluate(scenario, conversation)

    return {
        "scenario_id": scenario["id"],
        "persona": scenario["persona_name"],
        "turns": len(conversation),
        "conversation": conversation,
        "evaluation": evaluation,
    }


async def run_all_scenarios(
    scenario_filter: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """전체 시나리오 실행 및 리포트 생성"""
    scenarios_path = Path(__file__).parent / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
        if not scenarios:
            print(f"시나리오 '{scenario_filter}'를 찾을 수 없습니다.")
            return []

    # 지식베이스 시드
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()

    results = []
    for scenario in scenarios:
        result = await run_scenario(scenario, verbose=verbose)
        results.append(result)

    # 리포트 출력
    print(f"\n{'='*60}")
    print("AICC 자동 테스트 리포트")
    print(f"{'='*60}")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"시나리오 수: {len(results)}")
    print()

    total_score = 0
    for r in results:
        eval_data = r.get("evaluation", {})
        score = eval_data.get("overall_score", "N/A")
        goal = eval_data.get("goal_achieved", "N/A")
        summary = eval_data.get("summary", "")

        print(f"  [{r['scenario_id']}] {r['persona']}")
        print(f"    종합 점수: {score}/10 | 목표 달성: {'O' if goal else 'X'} | 턴 수: {r['turns']}")
        if summary:
            print(f"    요약: {summary}")

        # 세부 점수
        scores = eval_data.get("scores", {})
        if scores:
            score_parts = []
            for k, v in scores.items():
                if isinstance(v, dict):
                    score_parts.append(f"{k}:{v.get('score', '?')}")
            print(f"    세부: {' | '.join(score_parts)}")

        # 개선점
        improvements = eval_data.get("improvements", [])
        if improvements:
            for imp in improvements:
                print(f"    - 개선: {imp}")

        print()

        if isinstance(score, (int, float)):
            total_score += score

    avg = total_score / len(results) if results else 0
    print(f"{'─'*60}")
    print(f"평균 점수: {avg:.1f}/10")
    print(f"{'='*60}")

    # 결과를 JSON 파일로 저장
    report_path = Path(__file__).parent / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        # conversation은 너무 길 수 있으므로 요약만 저장
        summary_results = []
        for r in results:
            summary_results.append({
                "scenario_id": r["scenario_id"],
                "persona": r["persona"],
                "turns": r["turns"],
                "evaluation": r.get("evaluation", {}),
            })
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": summary_results,
            "average_score": avg,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n리포트 저장: {report_path}")
    return results


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="AICC 자동 테스트 에이전트")
    parser.add_argument("--scenario", type=str, help="특정 시나리오 ID만 실행")
    parser.add_argument("--verbose", "-v", action="store_true", help="대화 내용 출력")
    args = parser.parse_args()

    asyncio.run(run_all_scenarios(
        scenario_filter=args.scenario,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
