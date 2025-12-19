import os
from dotenv import load_dotenv
from data_loader import RAGLoader

# 환경 변수 로드
load_dotenv()

def main():
    # API 키 체크
    if not os.getenv("OPENAI_API_KEY"):
        print(".env 파일에 OPENAI_API_KEY가 없습니다.")
        return

    loader = RAGLoader()
    # ./data 폴더를 읽어서 ./chroma_db 생성
    loader.create_db_from_folder("./data")

if __name__ == "__main__":
    main()