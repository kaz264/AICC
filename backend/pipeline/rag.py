"""ChromaDB 기반 페르소나별 지식베이스 RAG + LRU 캐싱"""

import chromadb
from functools import lru_cache
from hashlib import md5
from pathlib import Path
from backend import config


_client: chromadb.ClientAPI | None = None
_collections: dict[str, object] = {}

# ── 캐시: 동일한 (kb_id, query) 조합은 재계산하지 않음 ──
_search_cache: dict[str, list[str]] = {}
_CACHE_MAX_SIZE = 500


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    return _client


def get_or_create_collection(knowledge_base_id: str):
    """페르소나별 ChromaDB 컬렉션 가져오기/생성 (컬렉션 객체 캐싱)"""
    if knowledge_base_id not in _collections:
        client = get_chroma_client()
        _collections[knowledge_base_id] = client.get_or_create_collection(
            name=knowledge_base_id,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[knowledge_base_id]


def add_documents(knowledge_base_id: str, documents: list[str], ids: list[str] | None = None):
    """지식베이스에 문서 추가"""
    collection = get_or_create_collection(knowledge_base_id)
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=ids)
    # 문서 추가 시 해당 KB 캐시 무효화
    keys_to_remove = [k for k in _search_cache if k.startswith(knowledge_base_id + ":")]
    for k in keys_to_remove:
        del _search_cache[k]


def _cache_key(knowledge_base_id: str, query: str, n_results: int) -> str:
    """캐시 키 생성"""
    query_hash = md5(query.encode()).hexdigest()[:12]
    return f"{knowledge_base_id}:{query_hash}:{n_results}"


def search_knowledge(knowledge_base_id: str, query: str, n_results: int = 3) -> list[str]:
    """지식베이스에서 관련 문서 검색 (캐싱 적용)"""
    cache_key = _cache_key(knowledge_base_id, query, n_results)

    # 캐시 히트
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    # 캐시 미스 → 검색
    try:
        collection = get_or_create_collection(knowledge_base_id)
        results = collection.query(query_texts=[query], n_results=n_results)
        docs = results["documents"][0] if results["documents"] else []
    except Exception:
        docs = []

    # 캐시 저장 (크기 제한)
    if len(_search_cache) >= _CACHE_MAX_SIZE:
        # 가장 오래된 항목 50개 삭제
        keys = list(_search_cache.keys())[:50]
        for k in keys:
            del _search_cache[k]
    _search_cache[cache_key] = docs

    return docs


def clear_cache():
    """캐시 전체 삭제 (테스트용)"""
    _search_cache.clear()
    _collections.clear()


def load_documents_from_file(knowledge_base_id: str, file_path: str):
    """텍스트 파일에서 문서를 읽어 지식베이스에 추가"""
    path = Path(file_path)
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    # 빈 줄 기준으로 청크 분리
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if chunks:
        ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
        add_documents(knowledge_base_id, chunks, ids)
