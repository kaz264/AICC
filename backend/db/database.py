"""SQLite 데이터베이스 관리"""

import aiosqlite
from datetime import datetime, timezone
from typing import Optional
from backend.config import DB_PATH
from backend.models.persona import Persona, PersonaCreate, PersonaUpdate
from backend.models.call_log import CallLog
import uuid


async def init_db():
    """데이터베이스 테이블 초기화"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS personas (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                greeting_message TEXT DEFAULT '안녕하세요, 무엇을 도와드릴까요?',
                language TEXT DEFAULT 'ko-KR',
                stt_provider TEXT DEFAULT 'google',
                tts_provider TEXT DEFAULT 'google',
                tts_voice_id TEXT DEFAULT 'ko-KR-Neural2-C',
                llm_model TEXT DEFAULT 'claude-sonnet-4-20250514',
                vad_sensitivity REAL DEFAULT 0.5,
                interrupt_threshold_ms INTEGER DEFAULT 800,
                filler_enabled BOOLEAN DEFAULT 1,
                knowledge_base_id TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS call_logs (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL,
                room_url TEXT DEFAULT '',
                status TEXT DEFAULT 'created',
                started_at TEXT,
                ended_at TEXT,
                duration_seconds INTEGER,
                latency_avg_ms REAL,
                FOREIGN KEY (persona_id) REFERENCES personas(id)
            )
        """)
        await db.commit()


# ── Persona CRUD ──

async def create_persona(data: PersonaCreate) -> Persona:
    now = datetime.now(timezone.utc).isoformat()
    persona_id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO personas
               (id, name, system_prompt, greeting_message, language,
                stt_provider, tts_provider, tts_voice_id, llm_model,
                vad_sensitivity, interrupt_threshold_ms, filler_enabled,
                knowledge_base_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                persona_id, data.name, data.system_prompt, data.greeting_message,
                data.language, data.stt_provider, data.tts_provider, data.tts_voice_id,
                data.llm_model, data.vad_sensitivity, data.interrupt_threshold_ms,
                data.filler_enabled, data.knowledge_base_id, now, now,
            ),
        )
        await db.commit()
    return await get_persona(persona_id)


async def get_persona(persona_id: str) -> Optional[Persona]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM personas WHERE id = ?", (persona_id,))
        row = await cursor.fetchone()
        if row:
            return Persona(**dict(row))
    return None


async def list_personas() -> list[Persona]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM personas ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [Persona(**dict(row)) for row in rows]


async def update_persona(persona_id: str, data: PersonaUpdate) -> Optional[Persona]:
    updates = data.model_dump(exclude_none=True)
    if not updates:
        return await get_persona(persona_id)
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [persona_id]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE personas SET {set_clause} WHERE id = ?", values
        )
        await db.commit()
    return await get_persona(persona_id)


async def delete_persona(persona_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
        await db.commit()
        return cursor.rowcount > 0


# ── Call Log CRUD ──

async def create_call_log(persona_id: str, room_url: str = "") -> CallLog:
    now = datetime.now(timezone.utc).isoformat()
    call_id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO call_logs (id, persona_id, room_url, status, started_at)
               VALUES (?, ?, ?, 'active', ?)""",
            (call_id, persona_id, room_url, now),
        )
        await db.commit()
    return CallLog(
        id=call_id, persona_id=persona_id,
        room_url=room_url, status="active", started_at=now,
    )


async def end_call_log(call_id: str, duration_seconds: int = 0, latency_avg_ms: float = 0):
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE call_logs
               SET status = 'ended', ended_at = ?, duration_seconds = ?, latency_avg_ms = ?
               WHERE id = ?""",
            (now, duration_seconds, latency_avg_ms, call_id),
        )
        await db.commit()


async def list_call_logs(limit: int = 50) -> list[CallLog]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM call_logs ORDER BY started_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [CallLog(**dict(row)) for row in rows]
