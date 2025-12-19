import os
import glob
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class RAGLoader:
    def __init__(self):
        self.db_path = "./chroma_db"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def create_db_from_folder(self, folder_path="./data"):
        """
        Data folder의 PDF 파일을 읽어서 Vector DB를 새로 생성합니다.
        기존 DB가 있으면 삭제하고 새로 생성합니다.
        """
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not pdf_files:
            print("경고: PDF 파일이 없습니다.")
            return None

        print(f"Vector DB 생성을 시작합니다 (대상 파일: {len(pdf_files)}개)")

        all_documents = []
        for file_path in pdf_files:
            filename = os.path.basename(file_path)
            try:
                print(f"로딩 중: {filename}")
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata['source'] = filename
                all_documents.extend(docs)
            except Exception as e:
                print(f"에러 발생 ({filename}): {e}")

        if not all_documents:
            return None

        # 문서를 청크 단위로 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_documents)

        # 기존 DB 삭제 및 새 DB 생성
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("Vector DB 생성 완료!")
        return vectorstore

    def load_existing_db(self):
        """
        이미 생성된 Vector DB를 로드합니다.
        DB가 없으면 None을 반환합니다.
        """
        if not os.path.exists(self.db_path):
            return None

        vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        return vectorstore