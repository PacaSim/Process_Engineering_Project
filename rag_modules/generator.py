from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def create_rag_chain(history_aware_retriever):
    """
    문서 기반 답변 생성 체인을 구성합니다.
    검색된 문서를 기반으로 정확한 답변을 생성하며, 환각을 방지합니다.

    Args:
        history_aware_retriever: 대화 기록을 고려하는 검색기

    Returns:
        rag_chain: 검색과 생성을 결합한 RAG 체인
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 답변 생성을 위한 시스템 프롬프트
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "Add a line break after every sentence to improve readability."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 문서 결합 체인
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 최종 RAG 체인 조립
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain