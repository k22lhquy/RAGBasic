# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# === LLM ===
LLM_BASE_URL    = "https://openrouter.ai/api/v1"
LLM_API_KEY     = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL       = "nvidia/nemotron-3-nano-30b-a3b:free"
LLM_TEMPERATURE = 0.2

# === Token Control ===
MAX_OUTPUT_TOKENS   = 512
CONTEXT_TOKEN_LIMIT = 3000
CHARS_PER_TOKEN     = 4

# === Embedding ===
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# === Vector Store ===
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "documents"

# === Chunking ===
# Tăng lên để giữ nguyên từng mục lớn trong file (14.1, 14.2, 14.3...)
# Mỗi mục khoảng 500-800 ký tự, CHUNK_SIZE 2000 đảm bảo không cắt ngang mục
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 200

# === Retrieval ===
TOP_K = 4  # tăng lên 4 vì reranker sẽ lọc lại sau

# === Reranking ===
# Đổi sang model multilingual để chấm điểm tiếng Việt chính xác hơn
RERANKER_MODEL   = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"  # hỗ trợ đa ngôn ngữ tốt hơn
RERANK_TOP_K     = 3
RERANK_THRESHOLD = -5.0  # hạ ngưỡng xuống để không bị loại quá nhiều
                          # với model mới score range khác, -5.0 là an toàn