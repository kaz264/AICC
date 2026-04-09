"""페르소나 CRUD REST API"""

from fastapi import APIRouter, HTTPException
from backend.models.persona import PersonaCreate, PersonaUpdate, Persona
from backend.db import database as db

router = APIRouter(prefix="/api/personas", tags=["personas"])


@router.get("/", response_model=list[Persona])
async def list_personas():
    return await db.list_personas()


@router.get("/{persona_id}", response_model=Persona)
async def get_persona(persona_id: str):
    persona = await db.get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="페르소나를 찾을 수 없습니다")
    return persona


@router.post("/", response_model=Persona, status_code=201)
async def create_persona(data: PersonaCreate):
    return await db.create_persona(data)


@router.put("/{persona_id}", response_model=Persona)
async def update_persona(persona_id: str, data: PersonaUpdate):
    persona = await db.update_persona(persona_id, data)
    if not persona:
        raise HTTPException(status_code=404, detail="페르소나를 찾을 수 없습니다")
    return persona


@router.delete("/{persona_id}")
async def delete_persona(persona_id: str):
    deleted = await db.delete_persona(persona_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="페르소나를 찾을 수 없습니다")
    return {"message": "삭제되었습니다"}
