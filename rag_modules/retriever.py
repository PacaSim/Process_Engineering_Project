from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def get_history_aware_retriever(vectorstore):
    """
    Args:
        vectorstore: 벡터 저장소 객체

    Returns:
        history_aware_retriever: 대화 기록을 고려하는 검색기
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 기본 검색기 (Top-k = 3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 질의 확장을 위한 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # History Aware Retriever 생성
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever