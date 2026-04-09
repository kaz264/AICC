"""통화 시작/종료 API"""

import asyncio
import aiohttp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend import config
from backend.db import database as db

router = APIRouter(prefix="/api/calls", tags=["calls"])


class StartCallRequest(BaseModel):
    persona_id: str


class StartCallResponse(BaseModel):
    call_id: str
    room_url: str
    token: str


async def _create_daily_room() -> dict:
    """Daily.co에 새 룸 생성"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/rooms",
            headers={"Authorization": f"Bearer {config.DAILY_API_KEY}"},
            json={
                "properties": {
                    "exp": None,  # PoC에서는 만료 없음
                    "enable_chat": False,
                    "enable_knocking": False,
                    "start_video_off": True,
                    "start_audio_off": False,
                }
            },
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise HTTPException(status_code=500, detail=f"Daily room 생성 실패: {error}")
            return await resp.json()


async def _create_daily_token(room_name: str) -> str:
    """Daily.co 룸 참가용 토큰 생성"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers={"Authorization": f"Bearer {config.DAILY_API_KEY}"},
            json={
                "properties": {
                    "room_name": room_name,
                    "is_owner": False,
                }
            },
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise HTTPException(status_code=500, detail=f"Daily token 생성 실패: {error}")
            data = await resp.json()
            return data["token"]


@router.post("/start", response_model=StartCallResponse)
async def start_call(req: StartCallRequest):
    """통화 시작: Daily room 생성 → bot 프로세스 실행"""
    # 페르소나 존재 확인
    persona = await db.get_persona(req.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="페르소나를 찾을 수 없습니다")

    # Daily room 생성
    room = await _create_daily_room()
    room_url = room["url"]
    room_name = room["name"]

    # 사용자용 토큰 생성
    token = await _create_daily_token(room_name)

    # 통화 로그 기록
    call_log = await db.create_call_log(req.persona_id, room_url)

    # Bot 프로세스를 백그라운드로 실행
    asyncio.create_task(
        _run_bot(room_url, req.persona_id, call_log.id)
    )

    return StartCallResponse(
        call_id=call_log.id,
        room_url=room_url,
        token=token,
    )


async def _run_bot(room_url: str, persona_id: str, call_id: str):
    """Pipecat bot 실행 (백그라운드)"""
    from backend.pipeline.bot import run_voice_agent
    try:
        await run_voice_agent(room_url, persona_id, call_id)
    except Exception as e:
        print(f"[Bot Error] call_id={call_id}: {e}")
    finally:
        await db.end_call_log(call_id)


@router.get("/logs")
async def get_call_logs():
    return await db.list_call_logs()
