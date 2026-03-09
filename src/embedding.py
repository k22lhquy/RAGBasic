# src/embedding.py
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL

MODELS_NEED_PREFIX = [
    "intfloat/multilingual-e5",
    "intfloat/e5-",
]

def _needs_prefix(model_name: str) -> bool:
    return any(p in model_name for p in MODELS_NEED_PREFIX)


class E5Embeddings(HuggingFaceEmbeddings):
    """
    Wrap HuggingFaceEmbeddings để tự động thêm prefix cho e5 models:
    - embed_documents → thêm "passage: " vào mỗi chunk
    - embed_query     → thêm "query: " vào câu hỏi
    """

    def embed_documents(self, texts):
        texts_with_prefix = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts_with_prefix)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def get_embedding_model():
    print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}")

    if _needs_prefix(EMBEDDING_MODEL):
        print("   ℹ️  Model này dùng prefix 'passage/query' tự động")
        embeddings = E5Embeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    print("✅ Embedding model loaded")
    return embeddings