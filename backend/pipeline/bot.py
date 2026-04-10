"""Pipecat 음성 AI 파이프라인 — 핵심 모듈

build_pipeline(): 파이프라인 조립 (Transport 외부 주입 — 테스트 가능)
run_voice_agent(): 프로덕션 진입점 (DailyTransport 생성 + 실행)
"""

import json
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.anthropic import AnthropicLLMContext

from backend import config
from backend.db import database as db
from backend.pipeline.persona_loader import (
    build_stt_service,
    build_llm_service,
    build_tts_service,
    build_system_prompt,
)
from backend.pipeline.rag import search_knowledge
from backend.pipeline.filler import FillerProcessor


async def build_pipeline(persona, transport):
    """Pipecat 파이프라인 조립 — Transport를 외부에서 주입받음

    프로덕션: DailyTransport
    테스트:   TestTransport (파일 기반)

    Returns: (task, runner, components)
        components: 디버깅/테스트용 내부 컴포넌트 참조
    """

    # ── 서비스 컴포넌트 생성 ──
    stt = build_stt_service(persona)
    llm = build_llm_service(persona)
    tts = build_tts_service(persona)

    # ── RAG function calling 도구 정의 ──
    tools = []
    if persona.knowledge_base_id:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "페르소나의 전문 지식베이스에서 관련 정보를 검색합니다. 고객의 질문에 정확하게 답변하기 위해 사용하세요.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색할 질문 또는 키워드",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    # ── LLM 컨텍스트 (시스템 프롬프트 + 도구) ──
    system_prompt = build_system_prompt(persona)
    messages = [{"role": "system", "content": system_prompt}]

    if persona.greeting_message:
        messages.append({"role": "assistant", "content": persona.greeting_message})

    context = AnthropicLLMContext(messages=messages, tools=tools if tools else None)
    context_aggregator = llm.create_context_aggregator(context)

    # ── Function calling 핸들러 ──
    async def on_tool_call(function_name, tool_call_id, arguments, llm_instance, context, result_callback):
        if function_name == "search_knowledge" and persona.knowledge_base_id:
            query = arguments.get("query", "")
            results = search_knowledge(persona.knowledge_base_id, query)
            result_text = "\n\n".join(results) if results else "관련 정보를 찾을 수 없습니다."
            await result_callback(json.dumps({"results": result_text}, ensure_ascii=False))
        else:
            await result_callback(json.dumps({"error": "알 수 없는 도구입니다."}))

    llm.register_function("search_knowledge", on_tool_call)

    # ── 추임새 프로세서 ──
    filler = FillerProcessor(enabled=persona.filler_enabled)

    # ── 파이프라인 조립 ──
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            filler,
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = PipelineRunner(handle_sigint=False)

    components = {
        "stt": stt,
        "llm": llm,
        "tts": tts,
        "filler": filler,
        "context_aggregator": context_aggregator,
        "context": context,
    }

    return task, runner, components


async def run_voice_agent(room_url: str, persona_id: str, call_id: str):
    """프로덕션 진입점: DailyTransport 생성 + 파이프라인 실행"""

    persona = await db.get_persona(persona_id)
    if not persona:
        raise ValueError(f"페르소나를 찾을 수 없습니다: {persona_id}")

    # ── Daily WebRTC Transport ──
    transport = DailyTransport(
        room_url,
        None,
        "AICC Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=SileroVADAnalyzer.VADParams(
                    threshold=persona.vad_sensitivity,
                    min_silence_duration_ms=persona.interrupt_threshold_ms,
                )
            ),
            transcription_enabled=False,
        ),
    )

    task, runner, components = await build_pipeline(persona, transport)

    # ── 이벤트 핸들러 ──
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant(transport, participant):
        if persona.greeting_message:
            await task.queue_frames(
                [components["tts"].create_text_frame(persona.greeting_message)]
            )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()

    # ── 실행 ──
    await runner.run(task)
