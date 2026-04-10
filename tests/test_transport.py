"""Pipecat TestTransport — 파일 기반 오디오 입출력

프로덕션의 DailyTransport를 대체하여 파이프라인을 테스트합니다.
- input(): WAV/MP3 파일에서 오디오 프레임을 읽어 파이프라인에 주입
- output(): 파이프라인 출력 오디오 프레임을 캡처

실제 STT/LLM/TTS 서비스는 그대로 호출합니다 (mock 없음).
"""

import asyncio
import time
import struct
import wave
import io
from pathlib import Path
from dataclasses import dataclass, field

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    CancelFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    Frame,
)


@dataclass
class TransportMetrics:
    """테스트 결과 수집"""
    input_frames_sent: int = 0
    output_audio_frames: int = 0
    output_audio_bytes: bytearray = field(default_factory=bytearray)
    transcriptions: list[str] = field(default_factory=list)
    text_outputs: list[str] = field(default_factory=list)
    all_output_frames: list[Frame] = field(default_factory=list)
    first_output_audio_time: float | None = None
    start_time: float = 0
    ttfr_ms: float = 0  # Time to First Response


class TestInputProcessor(FrameProcessor):
    """오디오 파일을 프레임으로 변환하여 파이프라인에 주입"""

    def __init__(self, audio_bytes: bytes, sample_rate: int = 24000, chunk_ms: int = 20):
        super().__init__()
        self.audio_bytes = audio_bytes
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._started = False

    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)

        if isinstance(frame, StartFrame) and not self._started:
            self._started = True
            await self._send_audio()

    async def _send_audio(self):
        """오디오를 청크 단위로 전송"""
        # WAV 헤더 파싱
        if self.audio_bytes[:4] == b'RIFF':
            try:
                wav_io = io.BytesIO(self.audio_bytes)
                with wave.open(wav_io, 'rb') as wf:
                    self.sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    raw_data = wf.readframes(wf.getnframes())
                    # 모노로 변환
                    if n_channels == 2:
                        raw_data = self._stereo_to_mono(raw_data, sampwidth)
            except Exception:
                raw_data = self.audio_bytes[44:]  # 헤더 스킵 폴백
        else:
            raw_data = self.audio_bytes

        chunk_size = int(self.sample_rate * 2 * self.chunk_ms / 1000)  # 16bit = 2 bytes/sample
        for i in range(0, len(raw_data), chunk_size):
            chunk = raw_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))

            await self.push_frame(
                AudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.sleep(self.chunk_ms / 1000)  # 실시간 속도 시뮬레이션

    def _stereo_to_mono(self, data: bytes, sampwidth: int) -> bytes:
        """스테레오 → 모노 변환"""
        samples = []
        step = sampwidth * 2
        for i in range(0, len(data), step):
            left = int.from_bytes(data[i:i + sampwidth], 'little', signed=True)
            right = int.from_bytes(data[i + sampwidth:i + step], 'little', signed=True)
            mono = (left + right) // 2
            samples.append(mono.to_bytes(sampwidth, 'little', signed=True))
        return b''.join(samples)


class TestOutputProcessor(FrameProcessor):
    """파이프라인 출력 프레임 캡처"""

    def __init__(self, metrics: TransportMetrics):
        super().__init__()
        self.metrics = metrics

    async def process_frame(self, frame, direction):
        self.metrics.all_output_frames.append(frame)

        if isinstance(frame, AudioRawFrame):
            self.metrics.output_audio_frames += 1
            self.metrics.output_audio_bytes.extend(frame.audio)

            if self.metrics.first_output_audio_time is None:
                self.metrics.first_output_audio_time = time.perf_counter()
                self.metrics.ttfr_ms = (
                    self.metrics.first_output_audio_time - self.metrics.start_time
                ) * 1000

        elif isinstance(frame, TranscriptionFrame):
            self.metrics.transcriptions.append(frame.text)

        elif isinstance(frame, TextFrame):
            self.metrics.text_outputs.append(frame.text)

        await self.push_frame(frame, direction)


class TestTransport:
    """DailyTransport 대체 — 파일 기반 테스트용 Transport

    사용법:
        transport = TestTransport(audio_bytes=customer_wav_data)
        task, runner, components = await build_pipeline(persona, transport)
        await runner.run(task)
        print(transport.metrics.ttfr_ms)
    """

    def __init__(self, audio_bytes: bytes, sample_rate: int = 24000):
        self.metrics = TransportMetrics()
        self._input = TestInputProcessor(audio_bytes, sample_rate)
        self._output = TestOutputProcessor(self.metrics)

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    def start_timer(self):
        """타이머 시작 (파이프라인 실행 직전 호출)"""
        self.metrics.start_time = time.perf_counter()

    def event_handler(self, event_name):
        """이벤트 핸들러 데코레이터 (DailyTransport 호환)"""
        def decorator(func):
            return func
        return decorator
