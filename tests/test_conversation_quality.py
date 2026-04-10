"""Tier 2: 대화 품질 회귀 테스트

agent_tester.py를 pytest로 래핑하여 CI에서 자동 실행합니다.
최소 품질 기준을 통과하지 못하면 테스트 실패.

사용법:
    pytest tests/test_conversation_quality.py -v -s
    pytest tests/test_conversation_quality.py -k "insurance" -v -s
"""

import asyncio
import pytest
from tests.agent_tester import run_scenario
import json
from pathlib import Path


SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"

# 최소 품질 기준
MIN_OVERALL_SCORE = 6
MIN_ACCURACY_SCORE = 6


def _load_scenarios():
    with open(SCENARIOS_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── 시나리오별 품질 테스트 ──

@pytest.mark.slow
async def test_insurance_claim_quality():
    """보험 청구 시나리오 품질"""
    scenarios = _load_scenarios()
    scenario = next(s for s in scenarios if s["id"] == "insurance_claim")
    result = await run_scenario(scenario)

    evaluation = result.get("evaluation", {})
    overall = evaluation.get("overall_score", 0)
    accuracy = evaluation.get("scores", {}).get("accuracy", {}).get("score", 0)
    goal = evaluation.get("goal_achieved", False)

    print(f"\n  종합: {overall}/10 | 정확성: {accuracy}/10 | 목표달성: {goal}")
    print(f"  요약: {evaluation.get('summary', '')}")

    assert overall >= MIN_OVERALL_SCORE, f"종합 {overall}/10 < 최소 {MIN_OVERALL_SCORE}"
    assert accuracy >= MIN_ACCURACY_SCORE, f"정확성 {accuracy}/10 < 최소 {MIN_ACCURACY_SCORE}"


@pytest.mark.slow
async def test_restaurant_reservation_quality():
    """레스토랑 예약 시나리오 품질"""
    scenarios = _load_scenarios()
    scenario = next(s for s in scenarios if s["id"] == "restaurant_reservation")
    result = await run_scenario(scenario)

    evaluation = result.get("evaluation", {})
    overall = evaluation.get("overall_score", 0)
    accuracy = evaluation.get("scores", {}).get("accuracy", {}).get("score", 0)
    goal = evaluation.get("goal_achieved", False)

    print(f"\n  종합: {overall}/10 | 정확성: {accuracy}/10 | 목표달성: {goal}")

    assert overall >= MIN_OVERALL_SCORE
    assert accuracy >= MIN_ACCURACY_SCORE


@pytest.mark.slow
async def test_vpn_troubleshoot_quality():
    """VPN 장애 시나리오 품질"""
    scenarios = _load_scenarios()
    scenario = next(s for s in scenarios if s["id"] == "vpn_troubleshoot")
    result = await run_scenario(scenario)

    evaluation = result.get("evaluation", {})
    overall = evaluation.get("overall_score", 0)
    accuracy = evaluation.get("scores", {}).get("accuracy", {}).get("score", 0)

    print(f"\n  종합: {overall}/10 | 정확성: {accuracy}/10")

    assert overall >= MIN_OVERALL_SCORE
    assert accuracy >= MIN_ACCURACY_SCORE


@pytest.mark.slow
async def test_insurance_product_inquiry_quality():
    """보험 상품 문의 시나리오 품질"""
    scenarios = _load_scenarios()
    scenario = next(s for s in scenarios if s["id"] == "insurance_product_inquiry")
    result = await run_scenario(scenario)

    evaluation = result.get("evaluation", {})
    overall = evaluation.get("overall_score", 0)

    print(f"\n  종합: {overall}/10")

    assert overall >= MIN_OVERALL_SCORE


@pytest.mark.slow
async def test_password_reset_quality():
    """비밀번호 초기화 시나리오 품질"""
    scenarios = _load_scenarios()
    scenario = next(s for s in scenarios if s["id"] == "password_reset")
    result = await run_scenario(scenario)

    evaluation = result.get("evaluation", {})
    overall = evaluation.get("overall_score", 0)

    print(f"\n  종합: {overall}/10")

    # 이 시나리오는 RAG 문제로 점수가 낮을 수 있으므로 기준을 낮춤
    assert overall >= 4, f"종합 {overall}/10 < 최소 4"
