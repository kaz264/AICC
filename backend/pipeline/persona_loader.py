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

    if persona.filler_enabled:
        base_prompt += """

## 자연스러운 대화 스타일
- 답변을 시작할 때 상황에 맞는 자연스러운 추임새를 사용하세요: "아, 네~", "음...", "아이고", "그렇군요"
- 완벽한 문장보다 자연스러운 구어체를 사용하세요
- 너무 길게 말하지 마세요. 한 번에 2-3문장 이내로 답변하세요.
"""

    return base_prompt
