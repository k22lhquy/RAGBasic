# src/multi_query.py
"""
Multi-Query Retriever — tự implement, không phụ thuộc vào langchain internal imports.

Flow:
  Câu hỏi gốc → LLM sinh thêm 3 câu hỏi → search song song → gộp, bỏ trùng
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL


def build_multi_query_retriever(vectorstore, llm=None):
    """
    Trả về 1 hàm retriever(question) thay vì object retriever.
    Tự implement để tránh lỗi import langchain internal.
    """

    if llm is None:
        llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            temperature=0
        )

    # Prompt sinh câu hỏi
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""Tạo ra 3 câu hỏi khác nhau cùng ý nghĩa với câu hỏi gốc bên dưới.
Mục đích là tìm kiếm tài liệu từ nhiều góc độ khác nhau.

Yêu cầu:
- Viết bằng tiếng Việt
- Mỗi câu hỏi trên 1 dòng
- Không đánh số, không giải thích
- Chỉ trả về đúng 3 dòng

Câu hỏi gốc: {question}"""
    )

    # Chain sinh câu hỏi
    generate_chain = prompt | llm | StrOutputParser()

    # Base retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def retriever(question: str):
        """Search bằng câu gốc + 3 câu sinh thêm, gộp và bỏ trùng"""

        # Sinh thêm câu hỏi
        generated = generate_chain.invoke({"question": question})
        extra_questions = [q.strip() for q in generated.strip().split("\n") if q.strip()]

        all_questions = [question] + extra_questions
        print(f"\n🔍 Multi-Query — các câu hỏi được dùng để search:")
        for i, q in enumerate(all_questions, 1):
            print(f"   {i}. {q}")

        # Search song song từng câu hỏi
        all_docs = []
        for q in all_questions:
            docs = base_retriever.invoke(q)
            all_docs.extend(docs)

        # Bỏ trùng theo nội dung
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        print(f"   → Tìm được {len(all_docs)} chunks, còn {len(unique_docs)} sau khi bỏ trùng\n")
        return unique_docs

    # Gắn thêm method invoke() để dùng như LangChain retriever
    class MultiQueryRetrieverWrapper:
        def invoke(self, question):
            return retriever(question)

    return MultiQueryRetrieverWrapper()