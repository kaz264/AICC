"""통화 로그 데이터 모델"""

from pydantic import BaseModel, Field
from typing import Optional
import uuid


class CallLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona_id: str
    room_url: str = ""
    status: str = Field(default="created", description="created | active | ended")
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    latency_avg_ms: Optional[float] = None
