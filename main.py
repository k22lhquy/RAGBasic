# main.py
"""
Entry point - chạy indexing và chat với tài liệu
"""
import os
import sys

from src.indexing import run_indexing, load_vectorstore, get_embedding_model
from src.rag_chain import build_rag_chain, ask


def main():
    print("=" * 50)
    print("📄 RAG - HỎI ĐÁP TỪ TÀI LIỆU")
    print("=" * 50)

    embeddings = get_embedding_model()

    # Nếu VectorDB đã tồn tại thì load lại, không cần index lại
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

    # Build RAG chain
    chain, retriever = build_rag_chain(vectorstore)

    # Chat loop
    print("\n✅ Sẵn sàng! Nhập câu hỏi (gõ 'exit' để thoát)\n")
    while True:
        question = input("Bạn: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit", "thoát"]:
            print("👋 Tạm biệt!")
            break

        ask(chain, retriever, question)
        print()


if __name__ == "__main__":
    main()