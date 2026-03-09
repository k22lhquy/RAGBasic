# main.py
import os
import sys

from src.indexing import run_indexing, load_vectorstore
from src.embedding import get_embedding_model
from src.rag_chain import build_rag_chain, ask
from src.history import ConversationHistory, build_rephrase_chain


def main():
    print("=" * 50)
    print("📄 RAG - HỎI ĐÁP TỪ TÀI LIỆU")
    print("=" * 50)

    embeddings = get_embedding_model()

    if os.path.exists("vectorstore") and os.listdir("vectorstore"):
        print("⚡ Tìm thấy VectorDB có sẵn, đang load...")
        vectorstore = load_vectorstore(embeddings)
        print("✅ Load VectorDB thành công")
    else:
        print("📥 Chưa có VectorDB, bắt đầu indexing tài liệu trong 'data/'...")
        vectorstore = run_indexing(data_dir="data")
        if vectorstore is None:
            print("\n💡 Hướng dẫn: Hãy bỏ file PDF hoặc TXT vào thư mục 'data/' rồi chạy lại.")
            sys.exit(1)

    chain, last_docs = build_rag_chain(vectorstore)

    # Khởi tạo history và rephrase chain
    history = ConversationHistory(max_turns=5)
    rephrase_chain = build_rephrase_chain()

    print("\n✅ Sẵn sàng! Nhập câu hỏi (gõ 'exit' để thoát, 'clear' để xóa lịch sử)\n")
    while True:
        question = input("Bạn: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit", "thoát"]:
            print("👋 Tạm biệt!")
            break
        if question.lower() == "clear":
            history.clear()
            print("🗑️  Đã xóa lịch sử hội thoại\n")
            continue

        ask(chain, last_docs, question,
            history=history,
            rephrase_chain=rephrase_chain)
        print()


if __name__ == "__main__":
    main()