"""페르소나 설정 → Pipecat 서비스 컴포넌트 매핑"""

from backend.models.persona import Persona
from backend import config


def build_stt_service(persona: Persona):
    """페르소나 설정에 따라 STT 서비스 생성"""
    if persona.stt_provider == "google":
        from pipecat.services.google.stt import GoogleSTTService
        return GoogleSTTService(
            language=persona.language,
        )
    else:
        raise ValueError(f"지원하지 않는 STT provider: {persona.stt_provider}")


def build_llm_service(persona: Persona):
    """페르소나 설정에 따라 LLM 서비스 생성"""
    from pipecat.services.anthropic import AnthropicLLMService

    return AnthropicLLMService(
        api_key=config.ANTHROPIC_API_KEY,
        model=persona.llm_model,
    )


def build_tts_service(persona: Persona):
    """페르소나 설정에 따라 TTS 서비스 생성"""
    if persona.tts_provider == "google":
        from pipecat.services.google.tts import GoogleTTSService
        return GoogleTTSService(
            language=persona.language,
            voice_name=persona.tts_voice_id,
        )
    elif persona.tts_provider == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
        return ElevenLabsTTSService(
            api_key=config.ELEVENLABS_API_KEY,
            voice_id=persona.tts_voice_id,
        )
    else:
        raise ValueError(f"지원하지 않는 TTS provider: {persona.tts_provider}")


def build_system_prompt(persona: Persona) -> str:
    """페르소나 시스템 프롬프트 구성"""
    base_prompt = persona.system_prompt

    # 공통 규칙: 모든 페르소나에 적용
    base_prompt += """

## 필수 응답 규칙 (절대 어기지 마세요)
1. **반드시 1-2문장으로만 답변하세요.** 3문장 이상은 절대 금지입니다.
2. 한 번에 하나의 정보만 전달하세요. 고객이 더 물어보면 그때 추가 설명하세요.
3. 불릿 포인트나 번호 목록을 사용하지 마세요. 대화체로 말하세요.
4. 고객의 문제가 해결되었는지 반드시 확인하세요. "이렇게 하시면 될 것 같은데, 해결되셨나요?"
5. 지식베이스에 있는 정보를 기반으로 구체적으로 답변하세요. 모르는 것은 모른다고 하되, 다른 부서로 넘기기 전에 아는 것부터 먼저 안내하세요.
"""

    if persona.filler_enabled:
        base_prompt += """
## 자연스러운 대화 스타일
- 답변 시작 시 자연스러운 추임새를 사용하세요: "네~", "아, 그렇군요", "그럼요"
- 완벽한 문장보다 자연스러운 구어체를 사용하세요.
"""

    return base_prompt
