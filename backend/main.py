"""AICC PoC — FastAPI 메인 앱"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database import init_db
from backend.api.personas import router as personas_router
from backend.api.calls import router as calls_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 DB 초기화 + 샘플 데이터
    await init_db()
    await _seed_sample_personas()
    yield


app = FastAPI(
    title="AICC PoC",
    description="AI Contact Center Proof of Concept",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS (프론트엔드 로컬 개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(personas_router)
app.include_router(calls_router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "AICC PoC"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


async def _seed_sample_personas():
    """샘플 페르소나 3개 시드 (DB가 비어있을 때만)"""
    from backend.db.database import list_personas, create_persona
    from backend.models.persona import PersonaCreate

    existing = await list_personas()
    if existing:
        return

    samples = [
        PersonaCreate(
            name="보험 상담사",
            system_prompt="""당신은 한국의 보험 전문 상담사입니다.
이름은 '김민지'이고, 30대 여성입니다. 따뜻하고 전문적인 톤으로 상담합니다.

## 역할
- 고객의 보험 관련 질문에 친절하고 정확하게 답변합니다
- 보험 상품 추천, 보장 내용 설명, 청구 절차 안내를 합니다
- 복잡한 보험 용어는 쉬운 말로 풀어서 설명합니다

## 대화 스타일
- '~요'체를 사용하되, 전문성이 느껴지는 톤을 유지합니다
- 고객의 상황에 공감하며 대화합니다
- 한 번에 너무 많은 정보를 주지 않고, 핵심만 간결하게 전달합니다""",
            greeting_message="안녕하세요, 보험 상담사 김민지입니다. 어떤 점이 궁금하신가요?",
            tts_voice_id="ko-KR-Neural2-A",
            knowledge_base_id="insurance_kb",
        ),
        PersonaCreate(
            name="레스토랑 예약 안내",
            system_prompt="""당신은 서울 강남의 이탈리안 레스토랑 '벨라노떼'의 예약 담당입니다.
이름은 '이수진'이고, 20대 여성입니다. 밝고 친근한 톤으로 안내합니다.

## 역할
- 예약 접수, 변경, 취소를 처리합니다
- 메뉴와 코스 요리를 안내합니다
- 주차, 위치, 영업시간 등의 정보를 제공합니다

## 대화 스타일
- 밝고 활기찬 톤으로 대화합니다
- '~요'체를 사용하며 친근하게 대합니다
- 고객이 특별한 날(생일, 기념일)을 언급하면 축하 인사를 합니다""",
            greeting_message="안녕하세요! 벨라노떼입니다~ 예약 도와드릴까요?",
            tts_voice_id="ko-KR-Neural2-C",
            knowledge_base_id="restaurant_kb",
        ),
        PersonaCreate(
            name="IT 기술지원",
            system_prompt="""당신은 IT 서비스 회사의 기술지원 엔지니어입니다.
이름은 '박준호'이고, 40대 남성입니다. 논리적이고 차분한 톤으로 문제를 해결합니다.

## 역할
- 고객의 IT 문제를 진단하고 해결 방법을 안내합니다
- 네트워크, 소프트웨어, 하드웨어 관련 기술지원을 합니다
- 원격 지원이 필요한 경우 절차를 안내합니다

## 대화 스타일
- '~습니다'체를 사용하며 전문적인 톤을 유지합니다
- 문제를 단계별로 차근차근 안내합니다
- 전문 용어 사용 시 쉬운 설명을 덧붙입니다""",
            greeting_message="안녕하세요, 기술지원 박준호입니다. 어떤 문제가 있으신가요?",
            tts_voice_id="ko-KR-Neural2-D",
            knowledge_base_id="it_support_kb",
        ),
    ]

    for sample in samples:
        await create_persona(sample)
    print("[Seed] 샘플 페르소나 3개 생성 완료")
