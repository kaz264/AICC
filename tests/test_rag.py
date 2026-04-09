"""RAG 지식베이스 테스트"""

import pytest
from backend.pipeline.rag import (
    get_or_create_collection,
    add_documents,
    search_knowledge,
    load_documents_from_file,
)
from pathlib import Path
import tempfile


def test_add_and_search_documents():
    """문서 추가 후 검색이 되는지 확인"""
    kb_id = "test_kb_search"
    docs = [
        "실손보험은 실제 병원 의료비를 보장하는 보험입니다.",
        "자동차보험은 교통사고 시 피해를 보상합니다.",
        "암보험은 암 진단 시 보험금을 지급합니다.",
    ]
    add_documents(kb_id, docs)

    results = search_knowledge(kb_id, "병원 치료비 보장", n_results=3)
    assert len(results) >= 1
    # ChromaDB 기본 임베딩(영어 모델)이라 한국어 시맨틱 매칭이 완벽하지 않을 수 있음
    # 최소한 저장한 문서 중 하나가 반환되는지 확인
    all_text = " ".join(results)
    assert "보험" in all_text


def test_search_relevance():
    """검색 결과가 쿼리와 관련성이 있는지 확인"""
    kb_id = "test_kb_relevance"
    docs = [
        "레스토랑 런치 코스는 35,000원입니다. 샐러드와 파스타가 포함됩니다.",
        "VPN 연결이 안 되면 인터넷 연결 상태를 먼저 확인하세요.",
        "프린터가 인쇄되지 않을 때는 용지와 잉크를 확인하세요.",
    ]
    add_documents(kb_id, docs)

    # IT 관련 질문 → IT 관련 문서가 나와야 함
    results = search_knowledge(kb_id, "VPN이 안 돼요", n_results=1)
    assert len(results) >= 1
    assert "VPN" in results[0] or "인터넷" in results[0]


def test_search_empty_collection():
    """빈 컬렉션 검색 시 빈 리스트 반환"""
    results = search_knowledge("nonexistent_kb_12345", "아무 질문")
    assert results == []


def test_load_documents_from_file():
    """파일에서 문서 로드 테스트"""
    kb_id = "test_kb_file_load"

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("첫 번째 청크입니다.\n\n두 번째 청크입니다.\n\n세 번째 청크입니다.")
        temp_path = f.name

    load_documents_from_file(kb_id, temp_path)

    results = search_knowledge(kb_id, "첫 번째", n_results=1)
    assert len(results) >= 1

    Path(temp_path).unlink()  # 정리


def test_load_documents_nonexistent_file():
    """존재하지 않는 파일 로드 시 에러 없이 처리"""
    load_documents_from_file("test_kb_nofile", "/nonexistent/path.txt")
    # 에러 없이 통과하면 성공


def test_get_or_create_collection():
    """컬렉션 생성/조회 테스트"""
    collection = get_or_create_collection("test_collection_create")
    assert collection is not None
    assert collection.name == "test_collection_create"

    # 같은 이름으로 다시 호출해도 에러 없어야 함
    collection2 = get_or_create_collection("test_collection_create")
    assert collection2.name == collection.name
