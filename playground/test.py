from groq import Groq
import chromadb
import numpy as np
import os

# Config
API_KEY = os.getenv("GROQ_API_KEY")  # https://console.groq.com
client = Groq(api_key=API_KEY)

# Vector DB local - dùng embedding đơn giản không cần API
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="docs")

# Embedding đơn giản bằng hash (không cần API key)
def embed(text: str) -> list[float]:
    words = text.lower().split()
    vec = [0.0] * 384
    for i, word in enumerate(words):
        idx = hash(word) % 384
        vec[idx] += 1.0
    norm = sum(x**2 for x in vec) ** 0.5
    return [x / norm if norm > 0 else x for x in vec]

# Thêm tài liệu
def add_documents(docs: list[str]):
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    for i, doc in enumerate(docs):
        collection.add(
            documents=[doc],
            embeddings=[embed(doc)],
            ids=[f"doc_{i}"]
        )
    print(f"✅ Đã thêm {len(docs)} tài liệu")

# Tìm context liên quan
def retrieve(query: str, n=3) -> str:
    total = collection.count()
    if total == 0:
        return "Không có tài liệu nào."
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=min(n, total)
    )
    return "\n\n".join(results["documents"][0])

# Chat với history
chat_history = []

def chat(user_message: str) -> str:
    context = retrieve(user_message)

    augmented_message = f"""Dựa vào tài liệu sau để trả lời.
Nếu không có thông tin liên quan, hãy nói "Tôi không có thông tin về vấn đề này."

<context>
{context}
</context>

Câu hỏi: {user_message}"""

    chat_history.append({"role": "user", "content": augmented_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "Bạn là trợ lý hữu ích, trả lời bằng tiếng Việt."}] + chat_history
    )

    reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": reply})
    return reply

# --- Main ---
if __name__ == "__main__":
    add_documents([
        "Công ty chúng tôi thành lập năm 2020.",
        "Chính sách hoàn tiền trong 30 ngày kể từ ngày mua.",
        "Sản phẩm A có giá 500,000 VND, bảo hành 12 tháng.",
        "Hotline hỗ trợ: 1800-1234, hoạt động 8h-22h mỗi ngày.",
    ])

    print("🤖 Chatbot RAG sẵn sàng! Gõ 'exit' để thoát.\n")

    while True:
        user_input = input("Bạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("👋 Tạm biệt!")
            break
        reply = chat(user_input)
        print(f"Bot: {reply}\n")