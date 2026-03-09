# main.py
import os
import sys

from src.indexing import run_indexing, load_vectorstore
from src.embedding import get_embedding_model
from src.rag_chain import build_rag_chain, ask
from src.history import ConversationHistory, build_rephrase_chain, list_sessions


def main():
    print("=" * 50)
    print("📄 RAG - HỎI ĐÁP TỪ TÀI LIỆU")
    print("=" * 50)

    # Hỏi có muốn tiếp tục session cũ không
    sessions = list_sessions()
    session_id = None
    if sessions:
        print("\nGõ ID session để tiếp tục, hoặc Enter để tạo mới:")
        inp = input("Session ID: ").strip()
        if inp in sessions:
            session_id = inp
            print(f"✅ Tiếp tục session: {session_id}")
        else:
            print("✅ Tạo session mới")

    embeddings = get_embedding_model()

    if os.path.exists("vectorstore") and os.listdir("vectorstore"):
        print("⚡ Tìm thấy VectorDB có sẵn, đang load...")
        vectorstore = load_vectorstore(embeddings)
        print("✅ Load VectorDB thành công")
    else:
        print("📥 Chưa có VectorDB, bắt đầu indexing...")
        vectorstore = run_indexing(data_dir="data")
        if vectorstore is None:
            print("\n💡 Bỏ file PDF hoặc TXT vào thư mục 'data/' rồi chạy lại.")
            sys.exit(1)

    chain, last_docs = build_rag_chain(vectorstore, streaming=True)

    history        = ConversationHistory(max_turns=5, session_id=session_id)
    rephrase_chain = build_rephrase_chain()

    # Hiển thị lịch sử nếu có
    if not history.is_empty():
        history.show_all()

    print("\n✅ Sẵn sàng!")
    print("   Lệnh đặc biệt: 'exit' thoát | 'clear' xóa history | 'history' xem lịch sử\n")

    while True:
        question = input("Bạn: ").strip()
        if not question:
            continue

        if question.lower() in ["exit", "quit", "thoát"]:
            print(f"👋 Tạm biệt! Session '{history.session_id}' đã được lưu.")
            break

        if question.lower() == "clear":
            history.clear()
            continue

        if question.lower() == "history":
            history.show_all()
            continue

        ask(chain, last_docs, question,
            history=history,
            rephrase_chain=rephrase_chain,
            streaming=True)
        print()


if __name__ == "__main__":
    main()