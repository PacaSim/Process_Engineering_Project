# 프로세스 공학 RAG 챗봇

프로세스 공학 문서를 기반으로 질문에 답변하는 RAG (Retrieval-Augmented Generation) 챗봇입니다.

## 주요 기능

- PDF 문서 자동 로딩 및 벡터화
- 대화 기록 기반 질의 확장 (Query Augmentation)
- 문서 출처 및 페이지 번호 표시
- 멀티턴 대화 지원
- Streamlit 기반 직관적인 웹 인터페이스

## 시스템 아키텍처

```
사용자 질문
    ↓
질의 확장 (대화 기록 반영)
    ↓
임베딩 변환
    ↓
벡터 검색 (Top-k=3)
    ↓
관련 문서 Chunk 검색
    ↓
LLM + 문서 기반 프롬프트
    ↓
답변 생성
    ↓
답변 + 출처 표시
    ↓
Streamlit UI 출력
```

## 기술 스택

- **LangChain**: RAG 파이프라인 구축
- **OpenAI API**: 임베딩 (text-embedding-3-small) 및 LLM (gpt-4o-mini)
- **ChromaDB**: 벡터 데이터베이스
- **Streamlit**: 웹 인터페이스
- **PyPDFLoader**: PDF 문서 로딩

## 설치 방법

### 1. 저장소 클론 및 이동

```bash
cd Process_Engineering_Project
```

### 2. 가상 환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_api_key_here
```

## 사용 방법

### 1. PDF 문서 준비

`data/` 폴더에 프로세스 공학 관련 PDF 문서를 저장합니다:

```bash
mkdir -p data
# PDF 파일을 data/ 폴더에 복사
```

### 2. 벡터 DB 구축

```bash
cd rag_modules
python build_vector_db.py
```

### 3. DB 상태 확인 (선택사항)

```bash
python check_db.py
```

### 4. 챗봇 실행

프로젝트 루트 디렉토리에서:

```bash
streamlit run main.py
```

브라우저에서 자동으로 열리지 않으면 `http://localhost:8501`로 접속합니다.

## 프로젝트 구조

```
Process_Engineering_Project/
├── main.py                      # Streamlit 메인 애플리케이션
├── requirements.txt             # Python 의존성
├── .env                         # 환경 변수 (API 키)
├── .gitignore                   # Git 제외 파일
├── README.md                    # 프로젝트 문서
├── data/                        # PDF 문서 저장 폴더
│   └── *.pdf
├── chroma_db/                   # 벡터 DB 저장 폴더 (자동 생성)
└── rag_modules/                 # RAG 모듈
    ├── data_loader.py           # PDF 로딩 및 벡터화
    ├── retriever.py             # 질의 확장 및 검색
    ├── generator.py             # 답변 생성
    ├── build_vector_db.py       # DB 구축 스크립트
    └── check_db.py              # DB 확인 스크립트
```

## 주요 파일 설명

### `main.py`
Streamlit 기반 웹 인터페이스를 제공합니다. 사용자 입력을 받아 RAG 체인을 실행하고 결과를 표시합니다.

### `rag_modules/data_loader.py`
- PDF 문서를 로딩하고 텍스트를 추출합니다
- RecursiveCharacterTextSplitter로 문서를 청크 단위로 분할합니다 (chunk_size=1000, overlap=200)
- OpenAI 임베딩 모델로 벡터화하여 ChromaDB에 저장합니다

### `rag_modules/retriever.py`
- 대화 기록을 고려한 질의 확장을 수행합니다
- 모호한 질문을 독립적인 질문으로 재구성합니다
- Top-k=3으로 가장 관련성 높은 문서 청크를 검색합니다

### `rag_modules/generator.py`
- 검색된 문서를 기반으로 답변을 생성합니다
- 환각(Hallucination)을 방지하기 위해 문서 기반 답변만 생성합니다
- 간결한 답변 (최대 3문장)을 생성합니다

### `rag_modules/build_vector_db.py`
벡터 DB를 처음부터 생성하는 스크립트입니다. 기존 DB가 있으면 삭제하고 새로 생성합니다.

### `rag_modules/check_db.py`
벡터 DB의 상태를 확인하고 샘플 검색을 수행하는 유틸리티 스크립트입니다.

## 설정 커스터마이징

### 검색 결과 개수 변경

`rag_modules/retriever.py`의 20번째 줄을 수정합니다:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # k 값 변경
```

### 청크 크기 조정

`rag_modules/data_loader.py`의 45-48번째 줄을 수정합니다:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,    # 청크 크기
    chunk_overlap=300   # 오버랩 크기
)
```

### LLM 모델 변경

`rag_modules/retriever.py` 및 `rag_modules/generator.py`의 LLM 설정을 수정합니다:

```python
llm = ChatOpenAI(model="gpt-4", temperature=0)  # 모델명 변경
```

### 답변 길이 조정

`rag_modules/generator.py`의 시스템 프롬프트를 수정합니다:

```python
qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use five sentences maximum and keep the "  # 문장 수 변경
    "answer concise."
    ...
)
```

## 문제 해결

### API 키 오류
```
경고: .env 파일에 API Key가 설정되지 않았습니다.
```
- `.env` 파일이 프로젝트 루트에 있는지 확인
- `OPENAI_API_KEY=sk-...` 형식이 맞는지 확인

### Vector DB 없음
```
경고: Vector DB가 없습니다. 먼저 'python build_vector_db.py'를 실행하세요.
```
- `rag_modules/build_vector_db.py`를 실행하여 DB를 생성

### PDF 로딩 실패
```
경고: PDF 파일이 없습니다.
```
- `data/` 폴더에 PDF 파일이 있는지 확인
- PDF 파일이 손상되지 않았는지 확인

### 검색 결과 없음
- PDF 문서에 실제 텍스트가 있는지 확인 (이미지만 있는 스캔본이 아닌지)
- `check_db.py`를 실행하여 DB 상태 확인


## 참고 자료

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
