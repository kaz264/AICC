"""샘플 지식베이스 데이터 로드 스크립트"""

from backend.pipeline.rag import load_documents_from_file
from pathlib import Path


def seed_knowledge_bases():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

    mappings = {
        "insurance_kb": data_dir / "insurance_faq.txt",
        "restaurant_kb": data_dir / "restaurant_menu.txt",
        "it_support_kb": data_dir / "it_support_manual.txt",
    }

    for kb_id, file_path in mappings.items():
        print(f"[Seed] {kb_id} <- {file_path.name}")
        load_documents_from_file(kb_id, str(file_path))

    print("[Seed] 지식베이스 시드 완료")


if __name__ == "__main__":
    seed_knowledge_bases()
