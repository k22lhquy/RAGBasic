# src/reranker.py
"""
Reranking — chấm điểm lại chunks sau khi retrieve

Flow:
  Multi-Query → 7 chunks (cosine similarity)
              → Reranker đọc từng cặp (câu hỏi, chunk)
              → chấm điểm relevance mới
              → sắp xếp lại
              → lấy top K chất lượng cao nhất

Dùng cross-encoder model local — không cần API key.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
       Hỗ trợ tiếng Việt ở mức cơ bản, đủ dùng cho RAG.
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Model cross-encoder — nhỏ, nhanh, chạy được trên CPU
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Số chunks giữ lại sau khi rerank
RERANK_TOP_K = 3


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL, top_k: int = RERANK_TOP_K):
        print(f"⏳ Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        print("✅ Reranker loaded")

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        """
        Chấm điểm lại từng chunk dựa trên câu hỏi.

        CrossEncoder nhận cặp (câu hỏi, chunk) và trả về
        1 điểm số duy nhất — khác với bi-encoder (embedding)
        vốn embed riêng rồi so sánh.

        Ví dụ:
          question = "lương phát ngày mấy?"
          docs = [chunk về nghỉ phép, chunk về lương, chunk về KPI, ...]

          CrossEncoder(question, chunk_luong)  → score 8.2  ← cao
          CrossEncoder(question, chunk_nghi)   → score 1.1  ← thấp
          CrossEncoder(question, chunk_kpi)    → score 3.4  ← trung bình

          → sắp xếp lại → [chunk_luong, chunk_kpi, ...]
          → lấy top 3
        """
        if not docs:
            return docs

        # Tạo pairs (câu hỏi, nội dung chunk)
        pairs = [(question, doc.page_content) for doc in docs]

        # CrossEncoder chấm điểm tất cả pairs
        scores = self.model.predict(pairs)

        # Gắn score vào metadata để có thể debug
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = round(float(score), 4)

        # Sắp xếp theo score giảm dần, lấy top_k
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:self.top_k]]

        # Log để thấy sự khác biệt trước/sau rerank
        print(f"\n   📈 Reranking: {len(docs)} chunks → top {self.top_k}")
        for i, (score, doc) in enumerate(ranked[:self.top_k], 1):
            src = doc.metadata.get("source", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"      {i}. score={score:.2f} | {preview}...")

        return top_docs