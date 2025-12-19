import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def inspect_vector_db():
    """Vector DB의 상태를 확인하고 간단한 검색 테스트를 수행합니다."""
    db_path = "./chroma_db"

    # DB 폴더 존재 여부 확인
    if not os.path.exists(db_path):
        print(f"'{db_path}' 폴더가 없습니다. 먼저 build_vector_db.py를 실행하세요.")
        return

    print(f"'{db_path}' 연결 시도 중...")

    # Vector DB 로드
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"DB 로드 실패: {e}")
        return

    # 데이터 개수 확인
    try:
        count = vectorstore._collection.count()
        if count == 0:
            print("DB는 연결되었으나, 저장된 데이터가 없습니다 (Count: 0).")
            print("data 폴더에 PDF가 있는지 확인하세요.")
            return
        else:
            print(f"DB 상태 정상! 총 저장된 Chunk 개수: {count}개")
    except Exception as e:
        print(f"경고: 개수 확인 중 오류 발생 - {e}")

    # 샘플 검색 테스트
    print("\n[검색 테스트 수행]")
    query = "랭체인 메모리에 기반한 멀티턴 챗봇 파이프라인"
    print(f"질문: '{query}'")

    results = vectorstore.similarity_search(query, k=1)

    if results:
        doc = results[0]
        print(f"검색 성공! 가장 유사한 문서 발견:")
        print(f"------------------------------------------------")
        print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
        print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
        print(f"내용 일부: {doc.page_content[:500]}...")
        print(f"------------------------------------------------")
    else:
        print("검색 결과가 없습니다.")


if __name__ == "__main__":
    inspect_vector_db()