"""ChromaDB 기반 페르소나별 지식베이스 RAG"""

import chromadb
from pathlib import Path
from backend import config


_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    return _client


def get_or_create_collection(knowledge_base_id: str):
    """페르소나별 ChromaDB 컬렉션 가져오기/생성"""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=knowledge_base_id,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(knowledge_base_id: str, documents: list[str], ids: list[str] | None = None):
    """지식베이스에 문서 추가"""
    collection = get_or_create_collection(knowledge_base_id)
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=ids)


def search_knowledge(knowledge_base_id: str, query: str, n_results: int = 3) -> list[str]:
    """지식베이스에서 관련 문서 검색"""
    try:
        collection = get_or_create_collection(knowledge_base_id)
        results = collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


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
