"""AICC 응답 속도 벤치마크

각 컴포넌트별 지연시간을 측정합니다:
1. RAG 검색 시간
2. LLM 첫 토큰 시간 (TTFT)
3. LLM 전체 응답 시간
4. 전체 파이프라인 시간 (RAG + LLM)

사용법:
    python -m tests.benchmark_latency
    python -m tests.benchmark_latency --rounds 10
"""

import asyncio
import time
import argparse
import statistics
from anthropic import Anthropic

from backend import config
from backend.db.database import init_db, list_personas
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import search_knowledge, load_documents_from_file


def measure_rag_latency(kb_id: str, query: str) -> dict:
    """RAG 검색 지연시간 측정"""
    start = time.perf_counter()
    results = search_knowledge(kb_id, query, n_results=3)
    elapsed = (time.perf_counter() - start) * 1000
    return {"rag_ms": round(elapsed, 1), "results_count": len(results)}


def measure_llm_latency(
    client: Anthropic,
    system_prompt: str,
    user_message: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """LLM 응답 지연시간 측정 (스트리밍)"""
    start = time.perf_counter()
    first_token_time = None
    full_response = ""

    with client.messages.stream(
        model=model,
        max_tokens=200,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = (time.perf_counter() - start) * 1000
            full_response += text

    total_time = (time.perf_counter() - start) * 1000

    return {
        "ttft_ms": round(first_token_time or 0, 1),
        "total_ms": round(total_time, 1),
        "response_length": len(full_response),
    }


def measure_full_pipeline(
    client: Anthropic,
    system_prompt: str,
    kb_id: str | None,
    user_message: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """전체 파이프라인 (RAG + LLM) 지연시간 측정"""
    pipeline_start = time.perf_counter()

    # RAG
    rag_result = {"rag_ms": 0, "results_count": 0}
    rag_context = ""
    if kb_id:
        rag_result = measure_rag_latency(kb_id, user_message)
        results = search_knowledge(kb_id, user_message, n_results=3)
        if results:
            rag_context = "\n\n## 참고 지식\n" + "\n\n".join(results)

    # LLM
    full_prompt = system_prompt + rag_context
    llm_result = measure_llm_latency(client, full_prompt, user_message, model)

    pipeline_total = (time.perf_counter() - pipeline_start) * 1000

    return {
        **rag_result,
        **llm_result,
        "pipeline_total_ms": round(pipeline_total, 1),
    }


# 테스트 질문들
TEST_QUERIES = [
    {"persona": "보험 상담사", "query": "실손보험 보장 범위가 어떻게 되나요?"},
    {"persona": "보험 상담사", "query": "보험금 청구하려면 서류가 뭐 필요해요?"},
    {"persona": "레스토랑 예약 안내", "query": "이번 주 토요일 저녁 2명 예약하고 싶어요"},
    {"persona": "레스토랑 예약 안내", "query": "디너 코스 가격이 얼마예요?"},
    {"persona": "IT 기술지원", "query": "VPN이 안 되는데 어떻게 해요?"},
    {"persona": "IT 기술지원", "query": "비밀번호를 잊어버렸어요"},
]


async def run_benchmark(rounds: int = 3):
    """벤치마크 실행"""
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # 초기화
    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()
    from backend.seed_knowledge import seed_knowledge_bases
    seed_knowledge_bases()

    personas = await list_personas()
    persona_map = {p.name: p for p in personas}

    print(f"\n{'='*70}")
    print("AICC 응답 속도 벤치마크")
    print(f"{'='*70}")
    print(f"라운드: {rounds}회 | 질문: {len(TEST_QUERIES)}개")
    print()

    all_results = []

    for q in TEST_QUERIES:
        persona = persona_map.get(q["persona"])
        if not persona:
            continue

        system_prompt = build_system_prompt(persona)
        round_results = []

        for r in range(rounds):
            result = measure_full_pipeline(
                client, system_prompt, persona.knowledge_base_id, q["query"], persona.llm_model
            )
            round_results.append(result)

        # 통계
        ttft_values = [r["ttft_ms"] for r in round_results]
        total_values = [r["total_ms"] for r in round_results]
        rag_values = [r["rag_ms"] for r in round_results]
        pipeline_values = [r["pipeline_total_ms"] for r in round_results]

        stats = {
            "persona": q["persona"],
            "query": q["query"][:30],
            "rag_avg": round(statistics.mean(rag_values), 1),
            "ttft_avg": round(statistics.mean(ttft_values), 1),
            "llm_total_avg": round(statistics.mean(total_values), 1),
            "pipeline_avg": round(statistics.mean(pipeline_values), 1),
        }
        all_results.append(stats)

        print(f"  [{stats['persona'][:6]}] \"{stats['query']}...\"")
        print(f"    RAG: {stats['rag_avg']}ms | TTFT: {stats['ttft_avg']}ms | LLM: {stats['llm_total_avg']}ms | Total: {stats['pipeline_avg']}ms")

    # 전체 요약
    print(f"\n{'─'*70}")
    print("전체 요약")
    print(f"{'─'*70}")

    avg_rag = round(statistics.mean([r["rag_avg"] for r in all_results]), 1)
    avg_ttft = round(statistics.mean([r["ttft_avg"] for r in all_results]), 1)
    avg_llm = round(statistics.mean([r["llm_total_avg"] for r in all_results]), 1)
    avg_pipeline = round(statistics.mean([r["pipeline_avg"] for r in all_results]), 1)

    print(f"  RAG 검색 평균:       {avg_rag}ms")
    print(f"  LLM 첫 토큰 (TTFT): {avg_ttft}ms")
    print(f"  LLM 전체 응답:       {avg_llm}ms")
    print(f"  파이프라인 전체:     {avg_pipeline}ms")

    # 실제 음성 파이프라인 예측
    print(f"\n{'─'*70}")
    print("실제 음성 파이프라인 지연 예측")
    print(f"{'─'*70}")
    stt_est = 300  # Google STT 스트리밍 평균
    tts_est = 200  # Google TTS 첫 오디오 청크
    voice_total = stt_est + avg_ttft + tts_est

    print(f"  STT (예측):          ~{stt_est}ms")
    print(f"  RAG + LLM TTFT:      ~{avg_rag + avg_ttft:.0f}ms")
    print(f"  TTS 첫 오디오 (예측): ~{tts_est}ms")
    print(f"  ────────────────────────────")
    print(f"  음성→음성 예측 합계: ~{voice_total:.0f}ms")

    if voice_total < 1000:
        print(f"  → 1초 이내 응답 가능! 자연스러운 대화 가능")
    elif voice_total < 1500:
        print(f"  → 1~1.5초 응답. 수용 가능하나 최적화 여지 있음")
    else:
        print(f"  → 1.5초 초과. 최적화 필요")

    print(f"\n{'─'*70}")
    print("응답 속도 개선 방법")
    print(f"{'─'*70}")
    print("  1. LLM 모델: Sonnet → Haiku (TTFT 50% 감소, 품질 트레이드오프)")
    print("  2. 프롬프트 길이 최소화 (시스템 프롬프트 + RAG 컨텍스트 축소)")
    print("  3. RAG: ChromaDB → 메모리 캐시 (반복 질문 캐싱)")
    print("  4. STT: 스트리밍 중간결과(interim)로 LLM 미리 호출")
    print("  5. TTS: 첫 문장 생성 즉시 재생 (나머지는 백그라운드)")
    print("  6. 추임새: '네~' 같은 짧은 오디오를 LLM 대기 중 즉시 재생")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="AICC 응답 속도 벤치마크")
    parser.add_argument("--rounds", type=int, default=3, help="측정 반복 횟수 (기본 3)")
    args = parser.parse_args()
    asyncio.run(run_benchmark(rounds=args.rounds))


if __name__ == "__main__":
    main()
