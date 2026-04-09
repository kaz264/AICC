"""페르소나 데이터 모델"""

from pydantic import BaseModel, Field
from typing import Optional
import uuid


class PersonaBase(BaseModel):
    name: str = Field(..., description="페르소나 이름 (예: 보험 상담사)")
    system_prompt: str = Field(..., description="LLM 시스템 프롬프트")
    greeting_message: str = Field(
        default="안녕하세요, 무엇을 도와드릴까요?",
        description="통화 시작 시 인사말",
    )
    language: str = Field(default="ko-KR")
    stt_provider: str = Field(default="google", description="google | deepgram")
    tts_provider: str = Field(default="google", description="google | elevenlabs")
    tts_voice_id: str = Field(
        default="ko-KR-Neural2-C",
        description="TTS 음성 ID (provider별 상이)",
    )
    llm_model: str = Field(default="claude-sonnet-4-20250514")
    vad_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    interrupt_threshold_ms: int = Field(
        default=800, description="침묵 감지 후 응답까지 대기 시간(ms)"
    )
    filler_enabled: bool = Field(
        default=True, description="추임새 사용 여부 (네~, 음...)"
    )
    knowledge_base_id: Optional[str] = Field(
        default=None, description="ChromaDB 컬렉션 ID"
    )


class PersonaCreate(PersonaBase):
    pass


class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    greeting_message: Optional[str] = None
    language: Optional[str] = None
    stt_provider: Optional[str] = None
    tts_provider: Optional[str] = None
    tts_voice_id: Optional[str] = None
    llm_model: Optional[str] = None
    vad_sensitivity: Optional[float] = None
    interrupt_threshold_ms: Optional[int] = None
    filler_enabled: Optional[bool] = None
    knowledge_base_id: Optional[str] = None


class Persona(PersonaBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True
