"""추임새(Filler) 프로세서

사용자 발화가 끝나고 LLM 응답을 기다리는 동안
"네~", "음...", "그렇군요" 같은 짧은 추임새를 즉시 TTS로 재생합니다.

효과: 체감 응답시간을 0에 가깝게 줄임 (LLM 대기 중에도 음성이 나옴)
"""

import random
from pipecat.frames.frames import TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# 상황별 추임새 목록
FILLERS_DEFAULT = [
    "네~",
    "네, 알겠습니다.",
    "아, 그렇군요.",
    "음...",
    "네네,",
    "그럼요,",
]

FILLERS_EMPATHY = [
    "아이고,",
    "아, 그러셨군요.",
    "네, 이해합니다.",
]


class FillerProcessor(FrameProcessor):
    """사용자 발화 종료 시 추임새를 즉시 삽입하는 프로세서

    파이프라인에서 STT 뒤, LLM 앞에 배치합니다:
    transport.input → stt → user_aggregator → [FillerProcessor] → llm → tts → ...
    """

    def __init__(self, enabled: bool = True, fillers: list[str] | None = None):
        super().__init__()
        self.enabled = enabled
        self.fillers = fillers or FILLERS_DEFAULT
        self._last_filler = ""

    async def process_frame(self, frame, direction: FrameDirection):
        """프레임 처리 — UserStoppedSpeaking 감지 시 추임새 삽입"""
        await self.push_frame(frame, direction)

        # 사용자 발화 완료 프레임 감지
        frame_name = type(frame).__name__
        if self.enabled and frame_name in (
            "UserStoppedSpeakingFrame",
            "TranscriptionFrame",
        ):
            filler = self._pick_filler()
            if filler:
                await self.push_frame(
                    TextFrame(text=filler),
                    FrameDirection.DOWNSTREAM,
                )

    def _pick_filler(self) -> str:
        """이전과 중복되지 않는 추임새 선택"""
        available = [f for f in self.fillers if f != self._last_filler]
        if not available:
            available = self.fillers
        filler = random.choice(available)
        self._last_filler = filler
        return filler
