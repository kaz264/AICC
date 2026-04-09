"""시나리오 기반 대화 테스트

실제 LLM 호출 없이, 페르소나 설정이 올바르게 구성되는지와
RAG가 올바른 지식을 반환하는지를 검증합니다.

실제 LLM 응답 품질 테스트는 API 키가 있을 때만 실행됩니다.
"""

import os
import pytest
from backend.db.database import create_persona, init_db
from backend.models.persona import PersonaCreate
from backend.pipeline.persona_loader import build_system_prompt
from backend.pipeline.rag import add_documents, search_knowledge


# ── 시나리오 1: 보험 상담사 ──

class TestInsuranceScenario:
    """보험 상담사 페르소나 시나리오"""

    @pytest.fixture(autouse=True)
    async def setup(self):
        self.persona = await create_persona(PersonaCreate(
            name="보험 상담사",
            system_prompt="당신은 보험 전문 상담사입니다. 고객에게 친절하게 보험 상품을 안내합니다.",
            greeting_message="안녕하세요, 보험 상담사 김민지입니다.",
            tts_voice_id="ko-KR-Neural2-A",
            knowledge_base_id="test_insurance_scenario",
            filler_enabled=True,
        ))
        # 지식베이스 세팅
        add_documents("test_insurance_scenario", [
            "실손보험은 실제 병원 의료비를 보장하는 보험입니다. 급여 80%, 비급여 70% 보장.",
            "암보험 가입 후 90일간은 면책 기간으로 보장이 되지 않습니다.",
            "보험금 청구는 병원 영수증과 진료 확인서를 앱에서 업로드하면 됩니다.",
        ])

    async def test_persona_prompt_includes_insurance_context(self):
        prompt = build_system_prompt(self.persona)
        assert "보험 전문 상담사" in prompt

    async def test_rag_returns_relevant_insurance_info(self):
        results = search_knowledge("test_insurance_scenario", "실손보험 보장 범위")
        assert len(results) >= 1
        assert any("실손보험" in r for r in results)

    async def test_rag_returns_claim_info(self):
        results = search_knowledge("test_insurance_scenario", "보험금 청구 방법")
        assert len(results) >= 1
        assert any("영수증" in r or "청구" in r for r in results)


# ── 시나리오 2: 레스토랑 예약 ──

class TestRestaurantScenario:
    """레스토랑 예약 페르소나 시나리오"""

    @pytest.fixture(autouse=True)
    async def setup(self):
        self.persona = await create_persona(PersonaCreate(
            name="레스토랑 예약",
            system_prompt="당신은 레스토랑 예약 담당입니다. 밝고 친근한 톤으로 안내합니다.",
            greeting_message="안녕하세요! 벨라노떼입니다~",
            tts_voice_id="ko-KR-Neural2-C",
            knowledge_base_id="test_restaurant_scenario",
        ))
        add_documents("test_restaurant_scenario", [
            "런치 코스 A는 35,000원입니다. 수프, 샐러드, 파스타, 에스프레소 포함.",
            "디너 코스는 85,000원입니다. 와인 페어링 추가 시 45,000원.",
            "영업시간은 점심 11:30~15:00, 저녁 17:30~22:00, 월요일 휴무.",
            "프라이빗 룸은 6~8인 기준, 코스 주문 시 무료 이용.",
        ])

    async def test_rag_returns_menu_info(self):
        results = search_knowledge("test_restaurant_scenario", "런치 코스 가격")
        assert len(results) >= 1
        assert any("35,000" in r or "런치" in r for r in results)

    async def test_rag_returns_hours(self):
        results = search_knowledge("test_restaurant_scenario", "영업시간 언제")
        assert len(results) >= 1
        assert any("11:30" in r or "영업시간" in r for r in results)


# ── 시나리오 3: IT 기술지원 ──

class TestITSupportScenario:
    """IT 기술지원 페르소나 시나리오"""

    @pytest.fixture(autouse=True)
    async def setup(self):
        self.persona = await create_persona(PersonaCreate(
            name="IT 기술지원",
            system_prompt="당신은 IT 기술지원 엔지니어입니다. 논리적이고 차분하게 문제를 해결합니다.",
            greeting_message="안녕하세요, 기술지원 박준호입니다.",
            tts_voice_id="ko-KR-Neural2-D",
            knowledge_base_id="test_it_scenario",
            filler_enabled=False,  # IT는 추임새 없이 간결하게
        ))
        add_documents("test_it_scenario", [
            "VPN 연결 안 될 때: 인터넷 확인 → VPN 재시작 → 서버 주소 확인 → IT팀 요청.",
            "비밀번호 초기화: 사내 포털 접속 → 비밀번호 찾기 → 휴대폰 인증 → 새 비밀번호 설정.",
            "프린터 안 될 때: 전원 확인 → 용지/잉크 확인 → 기본 프린터 설정 → 대기열 취소.",
        ])

    async def test_it_prompt_no_filler(self):
        """IT 지원은 추임새를 사용하지 않아야 함"""
        prompt = build_system_prompt(self.persona)
        assert "아, 네~" not in prompt

    async def test_rag_returns_vpn_troubleshoot(self):
        results = search_knowledge("test_it_scenario", "VPN이 안 돼요")
        assert len(results) >= 1
        assert any("VPN" in r for r in results)

    async def test_rag_returns_password_reset(self):
        results = search_knowledge("test_it_scenario", "비밀번호를 잊어버렸어요")
        assert len(results) >= 1
        assert any("비밀번호" in r for r in results)


# ── 크로스 페르소나 격리 테스트 ──

class TestPersonaIsolation:
    """서로 다른 페르소나의 지식이 섞이지 않는지 확인"""

    @pytest.fixture(autouse=True)
    def setup(self):
        add_documents("test_iso_insurance", ["실손보험 보장 내용입니다."])
        add_documents("test_iso_restaurant", ["런치 코스 메뉴입니다."])

    def test_insurance_kb_doesnt_have_restaurant(self):
        results = search_knowledge("test_iso_insurance", "런치 코스 가격")
        # 보험 KB에서 레스토랑 정보가 나오면 안 됨
        for r in results:
            assert "런치 코스" not in r

    def test_restaurant_kb_doesnt_have_insurance(self):
        results = search_knowledge("test_iso_restaurant", "실손보험 보장")
        for r in results:
            assert "실손보험" not in r
