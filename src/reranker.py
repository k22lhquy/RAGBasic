# src/reranker.py
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.config import RERANKER_MODEL, RERANK_TOP_K, RERANK_THRESHOLD


class Reranker:
    def __init__(self, model_name=RERANKER_MODEL, top_k=RERANK_TOP_K, threshold=RERANK_THRESHOLD):
        print(f"⏳ Loading reranker model: {model_name}")
        self.model     = CrossEncoder(model_name)
        self.top_k     = top_k
        self.threshold = threshold
        print(f"✅ Reranker loaded (top_k={top_k}, threshold={threshold})")

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs

        pairs  = [(question, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = round(float(score), 4)

        ranked  = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        passed  = [(s, d) for s, d in ranked if s >= self.threshold]
        removed = len(ranked) - len(passed)
        top     = passed[:self.top_k]

        print(f"\n   📈 Reranking: {len(docs)} chunks"
              f" → {len(passed)} pass threshold({self.threshold})"
              f" → top {len(top)}")
        for i, (score, doc) in enumerate(top, 1):
            preview = doc.page_content[:70].replace("\n", " ")
            print(f"      {i}. score={score:+.2f} | {preview}...")
        if removed:
            print(f"      ✂️  Loại {removed} chunks (score < {self.threshold})")

        # Fallback: luôn giữ ít nhất 1 chunk tốt nhất
        if not top:
            print("   ⚠️  Fallback: giữ chunk score cao nhất")
            top = ranked[:1]

        return [doc for _, doc in top]