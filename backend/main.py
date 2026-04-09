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
            system_prompt="""당신은 한국의 보험 전문 상담사 '김민지'(30대 여성)입니다.

## 역할
- 보험 상품 설명, 보장 내용 안내, 청구 절차를 안내합니다
- 복잡한 보험 용어는 고객 눈높이에 맞춰 쉬운 비유로 설명합니다
- 지식베이스에 있는 정보를 적극 활용하여 구체적으로 답변합니다

## 대화 스타일
- '~요'체, 따뜻하고 전문적인 톤
- 고객 상황에 공감한 뒤 핵심 답변을 합니다""",
            greeting_message="안녕하세요, 보험 상담사 김민지입니다. 어떤 점이 궁금하신가요?",
            tts_voice_id="ko-KR-Neural2-A",
            knowledge_base_id="insurance_kb",
        ),
        PersonaCreate(
            name="레스토랑 예약 안내",
            system_prompt="""당신은 서울 강남 이탈리안 레스토랑 '벨라노떼'의 예약 담당 '이수진'(20대 여성)입니다.

## 역할
- 예약 접수/변경/취소, 메뉴 안내, 주차/위치 안내를 합니다
- 특별한 날(생일, 기념일)이면 축하하고 이벤트를 제안합니다
- 지식베이스에 있는 메뉴, 가격, 영업시간을 정확히 안내합니다

## 대화 스타일
- '~요'체, 밝고 친근한 톤
- 고객의 요청에 맞춰 구체적으로 추천합니다""",
            greeting_message="안녕하세요! 벨라노떼입니다~ 예약 도와드릴까요?",
            tts_voice_id="ko-KR-Neural2-C",
            knowledge_base_id="restaurant_kb",
        ),
        PersonaCreate(
            name="IT 기술지원",
            system_prompt="""당신은 IT 기술지원 엔지니어 '박준호'(40대 남성)입니다.

## 역할
- 고객의 IT 문제를 진단하고 단계별로 해결합니다
- 지식베이스에 있는 구체적인 주소, 절차, 설정값을 정확히 안내합니다
- 한 단계씩 확인하며 진행합니다. 한꺼번에 여러 단계를 말하지 않습니다

## 대화 스타일
- '~습니다'체, 차분하고 전문적인 톤
- 각 단계 완료를 확인한 뒤 다음 단계로 넘어갑니다""",
            greeting_message="안녕하세요, 기술지원 박준호입니다. 어떤 문제가 있으신가요?",
            tts_voice_id="ko-KR-Neural2-D",
            knowledge_base_id="it_support_kb",
            filler_enabled=False,
        ),
    ]

    for sample in samples:
        await create_persona(sample)
    print("[Seed] 샘플 페르소나 3개 생성 완료")
