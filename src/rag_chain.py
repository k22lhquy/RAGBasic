# src/rag_chain.py
"""
Query Pipeline (Search thẳng - không transform query):
  Câu hỏi → Embed → Search VectorDB → Lấy chunks → LLM → Trả lời
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K
)


def get_llm():
    """Khởi tạo LLM qua OpenRouter"""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )


def format_docs(docs):
    """Gộp các chunks thành 1 đoạn context để đưa vào prompt"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (trang {page + 1})" if page != "" else ""
        formatted.append(f"[Đoạn {i} - {source}{page_info}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore):
    """
    Xây dựng RAG chain:
    câu hỏi → retrieve → format context → prompt → LLM → output
    """
    # Retriever: search thẳng theo similarity score
    retriever = vectorstore.as_retriever(
        search_type="similarity",   # search thẳng, không transform
        search_kwargs={"k": TOP_K}  # lấy top K chunks
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
Bạn là trợ lý thông minh. Hãy trả lời câu hỏi CHỈ DỰA TRÊN thông tin trong tài liệu được cung cấp.

Nếu thông tin không có trong tài liệu, hãy nói: "Tài liệu không đề cập đến vấn đề này."
Không được bịa đặt thông tin ngoài tài liệu.

=== NỘI DUNG TÀI LIỆU ===
{context}

=== CÂU HỎI ===
{question}

=== TRẢ LỜI ===
""")

    llm = get_llm()

    # RAG chain: LCEL (LangChain Expression Language)
    chain = (
        {
            "context": retriever | format_docs,  # retrieve → format
            "question": RunnablePassthrough()     # giữ nguyên câu hỏi
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain, retriever, question: str, show_sources: bool = True):
    """Hỏi và hiển thị kết quả + nguồn"""
    print(f"\n❓ Câu hỏi: {question}")
    print("-" * 50)

    # Lấy chunks để hiện sources
    if show_sources:
        docs = retriever.invoke(question)

    # Gọi chain để lấy câu trả lời
    answer = chain.invoke(question)

    print(f"💬 Trả lời:\n{answer}")

    if show_sources:
        print("\n📚 Nguồn tham khảo:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            score_info = f"trang {page + 1}" if page != "" else ""
            print(f"  [{i}] {source} {score_info}")
            print(f"      {doc.page_content[:100]}...")

    return answer