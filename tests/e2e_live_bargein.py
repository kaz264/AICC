"""실제 Pipecat 라이브 Barge-in 테스트

AICC Bot: Pipecat 파이프라인 (별도 프로세스)
고객 Bot: daily-python SDK 직접 사용
"""

import asyncio
import aiohttp
import json
import time
import re
import multiprocessing
import wave
import io
import httpx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config


# -- Daily Room --

async def create_room():
    async with aiohttp.ClientSession() as s:
        async with s.post(
            "https://api.daily.co/v1/rooms",
            headers={"Authorization": f"Bearer {config.DAILY_API_KEY}"},
            json={"properties": {"enable_chat": False, "start_video_off": True}},
        ) as r:
            return await r.json()


async def delete_room(name):
    async with aiohttp.ClientSession() as s:
        await s.delete(
            f"https://api.daily.co/v1/rooms/{name}",
            headers={"Authorization": f"Bearer {config.DAILY_API_KEY}"},
        )


# -- AICC Bot (별도 프로세스) --

def _aicc_process(room_url, persona_id):
    import asyncio
    from backend.db.database import init_db, create_call_log
    from backend.pipeline.bot import run_voice_agent

    async def _run():
        await init_db()
        from backend.main import _seed_sample_personas
        await _seed_sample_personas()
        from backend.seed_knowledge import seed_knowledge_bases
        seed_knowledge_bases()
        log = await create_call_log(persona_id, room_url)
        print("  [AICC Bot] 파이프라인 시작", flush=True)
        await run_voice_agent(room_url, persona_id, log.id)

    try:
        asyncio.run(_run())
    except Exception as e:
        print(f"  [AICC Bot] Error: {e}", flush=True)


# -- TTS --

def tts(voice_id, text):
    text = re.sub(r"[~*#]", "", text).strip()
    r = httpx.post(
        "https://api.typecast.ai/v1/text-to-speech",
        headers={"X-API-KEY": config.TYPECAST_API_KEY, "Content-Type": "application/json"},
        json={
            "voice_id": voice_id, "text": text,
            "model": "ssfm-v30", "language": "kor",
            "output": {"format": "wav"},
        },
        timeout=30,
    )
    return r.content if r.status_code == 200 else b""


# -- 고객 Bot (daily-python SDK 직접 사용) --

def _customer_process(room_url, turns, voice_id):
    import daily
    import threading

    daily.Daily.init()

    print("  [Customer Bot] 음성 합성 중...", flush=True)
    audios = []
    for t in turns:
        txt = t.get("customer") or t.get("customer_interrupt", "")
        if txt:
            audio = tts(voice_id, txt)
            audios.append({"text": txt, "audio": audio, "type": t["type"]})
            time.sleep(0.5)
    print(f"  [Customer Bot] {len(audios)} turns ready", flush=True)

    # 가상 마이크 생성
    mic = daily.Daily.create_microphone_device(
        "customer-mic", sample_rate=24000, channels=1
    )

    # CallClient 생성 + Room 참가
    client = daily.CallClient()
    joined = threading.Event()

    def on_joined(data, error):
        if not error:
            joined.set()
            print("  [Customer Bot] Room 참가 완료", flush=True)
        else:
            print(f"  [Customer Bot] Join error: {error}", flush=True)

    client.join(room_url, completion=on_joined)

    joined.wait(timeout=10)
    if not joined.is_set():
        print("  [Customer Bot] Join failed!", flush=True)
        return

    # 마이크 입력 활성화
    client.update_inputs({
        "microphone": {
            "isEnabled": True,
            "settings": {"deviceId": "customer-mic"},
        },
        "camera": False,
    })

    # AICC 인사말 대기
    print("  [Customer Bot] Waiting for greeting (5s)...", flush=True)
    time.sleep(5)

    for i, item in enumerate(audios):
        print(f"\n  [Customer Bot] Turn {i+1}: {item['text'][:40]}...", flush=True)

        if item["type"] == "bargein":
            print("  [Customer Bot] BARGE-IN! (2s then interrupt)", flush=True)
            time.sleep(2)

        audio = item["audio"]
        if audio and audio[:4] == b"RIFF":
            wav_io = io.BytesIO(audio)
            with wave.open(wav_io, "rb") as wf:
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())

            chunk_size = sr * 2 // 50  # 20ms
            for j in range(0, len(raw), chunk_size):
                chunk = raw[j:j + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b"\x00" * (chunk_size - len(chunk))
                mic.write_frames(chunk)
                time.sleep(0.02)

        if item["type"] == "bargein":
            time.sleep(5)
        else:
            print("  [Customer Bot] Waiting for response (8s)...", flush=True)
            time.sleep(8)

    print("\n  [Customer Bot] Done, leaving", flush=True)
    time.sleep(2)
    client.leave()


# -- 시나리오 --

TURNS = [
    {"type": "normal", "customer": "안녕하세요, 이메일이 갑자기 안 돼요. 아웃룩에서 메일이 안 보내져요."},
    {"type": "bargein", "customer_interrupt": "아, 그건 아니고요. 보내기가 안 되는 거예요. 받는 건 돼요."},
    {"type": "normal", "customer": "보내기 버튼 누르면 서버 연결 실패라고 나와요."},
    {"type": "normal", "customer": "네, 해볼게요."},
    {"type": "normal", "customer": "됐어요, 감사합니다."},
]


# -- 메인 --

async def main():
    from backend.db.database import init_db, list_personas

    await init_db()
    from backend.main import _seed_sample_personas
    await _seed_sample_personas()

    personas = await list_personas()
    persona = next((p for p in personas if p.name == "IT 기술지원"), None)
    if not persona:
        print("IT 기술지원 없음")
        return

    sep = "=" * 70
    print(f"\n{sep}")
    print("Live Barge-in Test (Daily Room + Pipecat + daily-python)")
    print(f"Customer: Female | Agent: Male (IT Support)")
    print(f"{sep}")

    room = await create_room()
    room_url = room["url"]
    room_name = room["name"]
    print(f"\n  Room: {room_url}")

    try:
        print("  Starting AICC Bot...")
        aicc = multiprocessing.Process(target=_aicc_process, args=(room_url, persona.id))
        aicc.start()
        await asyncio.sleep(8)

        print("  Starting Customer Bot...")
        customer = multiprocessing.Process(
            target=_customer_process,
            args=(room_url, TURNS, "tc_68537c9420b646f2176890ba"),
        )
        customer.start()
        customer.join(timeout=90)

        print(f"\n  Test complete!")

    finally:
        if aicc.is_alive():
            aicc.terminate()
            aicc.join(timeout=5)
        await delete_room(room_name)
        print(f"  Room {room_name} deleted")

    print(f"{sep}")


if __name__ == "__main__":
    asyncio.run(main())
