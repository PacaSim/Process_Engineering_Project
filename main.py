import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from rag_modules.data_loader import RAGLoader
from rag_modules.retriever import get_history_aware_retriever
from rag_modules.generator import create_rag_chain

load_dotenv()

st.set_page_config(page_title="프로세스 공학 RAG 챗봇", page_icon="Q")
st.title("프로세스 공학 기말 범위 Q&A 챗봇")

# 초기화 로직
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.chat_history = []

    # 앱 시작 시 DB 로드 시도
    if os.getenv("OPENAI_API_KEY"):
        loader = RAGLoader()
        vectorstore = loader.load_existing_db()

        if vectorstore:
            retriever = get_history_aware_retriever(vectorstore)
            st.session_state.rag_chain = create_rag_chain(retriever)
            st.toast("기존 지식 베이스를 성공적으로 불러왔습니다!")
        else:
            st.error("경고: Vector DB가 없습니다. 먼저 'python build_vector_db.py'를 실행하세요.")
    else:
        st.error("경고: .env 파일에 API Key가 설정되지 않았습니다.")

# 사이드바
with st.sidebar:
    st.header("상태 정보")
    if st.session_state.rag_chain:
        st.success("시스템 상태: 준비 완료")
        st.info("지식 베이스가 로드되었습니다.")
    else:
        st.error("시스템 상태: DB 없음")
        st.markdown("터미널에서 DB 구축 스크립트를 실행해주세요.")

# 채팅 인터페이스
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    if not st.session_state.rag_chain:
        st.error("시스템이 준비되지 않았습니다. DB 구축을 먼저 진행해주세요.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = st.session_state.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })

                answer = response['answer']
                sources = response['context']

                st.markdown(answer)

                if sources:
                    st.divider()
                    st.caption("참고 문서 출처:")
                    source_list = []
                    for doc in sources:
                        src_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        if 'page' in doc.metadata:
                            src_name += f" (Page {int(doc.metadata['page'])+1})"
                        source_list.append(f"- {src_name}")

                    for src in set(source_list):
                        st.caption(src)

        st.session_state.chat_history.append(AIMessage(content=answer))