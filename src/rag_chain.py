# src/rag_chain.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    TOP_K, MAX_OUTPUT_TOKENS, CONTEXT_TOKEN_LIMIT, CHARS_PER_TOKEN
)


def get_llm():
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS
    )


def truncate_docs(docs):
    max_chars = CONTEXT_TOKEN_LIMIT * CHARS_PER_TOKEN
    total_chars = 0
    kept_docs = []

    for doc in docs:
        doc_len = len(doc.page_content)
        if total_chars + doc_len > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                doc.page_content = doc.page_content[:remaining] + "..."
                kept_docs.append(doc)
            break
        kept_docs.append(doc)
        total_chars += doc_len

    total_tokens_est = total_chars // CHARS_PER_TOKEN
    print(f"   📊 Context: {len(kept_docs)}/{len(docs)} chunks "
          f"| ~{total_tokens_est} tokens (giới hạn {CONTEXT_TOKEN_LIMIT})")
    return kept_docs


def format_docs(docs):
    docs = truncate_docs(docs)
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (trang {page + 1})" if page != "" else ""
        formatted.append(f"[Đoạn {i} - {source}{page_info}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore, mode: str = "multi_query"):

    if mode == "multi_query":
        from src.multi_query import build_multi_query_retriever
        print("🔍 Chế độ: Multi-Query Retriever")
        retriever = build_multi_query_retriever(vectorstore, llm=get_llm())
    else:
        print("🔍 Chế độ: Simple Retriever")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )

    # Prompt có thêm history
    prompt = ChatPromptTemplate.from_template("""
Bạn là trợ lý thông minh. Hãy trả lời câu hỏi CHỈ DỰA TRÊN thông tin trong tài liệu được cung cấp.

Nếu thông tin không có trong tài liệu, hãy nói: "Tài liệu không đề cập đến vấn đề này."
Không được bịa đặt thông tin ngoài tài liệu.

=== LỊCH SỬ HỘI THOẠI ===
{history}

=== NỘI DUNG TÀI LIỆU ===
{context}

=== CÂU HỎI ===
{question}

=== TRẢ LỜI ===
""")

    llm = get_llm()
    last_docs = {"docs": []}

    def retrieve_and_format(question: str) -> str:
        docs = retriever.invoke(question)
        last_docs["docs"] = docs
        context = format_docs(docs)
        print(f"\n   🧩 Context gửi LLM (200 ký tự đầu):")
        print(f"   {context[:200]}...")
        return context

    # Chain nhận dict gồm question + history
    chain = (
        {
            "context":  RunnableLambda(lambda x: retrieve_and_format(x["question"])),
            "question": RunnableLambda(lambda x: x["question"]),
            "history":  RunnableLambda(lambda x: x["history"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, last_docs


def ask(chain, last_docs: dict, question: str,
        history=None, rephrase_chain=None, show_sources: bool = True):
    """
    history        : ConversationHistory object (None = không dùng history)
    rephrase_chain : chain để rephrase câu hỏi (None = không rephrase)
    """
    from src.history import rephrase_question

    print(f"\n❓ Câu hỏi: {question}")
    print("-" * 50)

    # Rephrase câu hỏi nếu có history
    search_question = question
    if history is not None and rephrase_chain is not None:
        search_question = rephrase_question(rephrase_chain, history, question)

    history_text = history.format() if history else "(chưa có lịch sử)"

    # Gọi chain 1 lần duy nhất
    answer = chain.invoke({
        "question": search_question,
        "history": history_text
    })

    print(f"💬 Trả lời:\n{answer}")

    # Lưu vào history
    if history is not None:
        history.add(user=question, bot=answer)

    # Hiển thị sources
    if show_sources and last_docs["docs"]:
        seen = set()
        unique_docs = []
        for doc in last_docs["docs"]:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        print(f"\n📚 Nguồn tham khảo ({len(unique_docs)} chunks):")
        for i, doc in enumerate(unique_docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_info = f" trang {page + 1}" if page != "" else ""
            print(f"  [{i}] {source}{page_info}")
            print(f"      {doc.page_content[:120]}...")

    return answer