# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# === LLM ===

# --- OpenRouter (cũ) ---
# LLM_BASE_URL = "https://openrouter.ai/api/v1"
# LLM_API_KEY  = os.getenv("OPENROUTER_API_KEY")
# LLM_MODEL    = "google/gemma-3n-e4b-it:free"

# --- Groq (mới) ---
# Lấy key tại: https://console.groq.com/keys
# Danh sách model còn hoạt động (tháng 3/2026):
#   "llama-3.3-70b-versatile"   ← mạnh nhất, khuyên dùng
#   "llama-3.1-8b-instant"      ← nhanh nhất, nhẹ hơn
#   "mixtral-8x7b-32768"        ← context window lớn 32k
# Model đã bị khai tử: gemma2-9b-it, llama-4-scout-17b
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY  = os.getenv("GROQ_API_KEY")
LLM_MODEL    = "llama-3.3-70b-versatile"

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
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 200

# === Retrieval ===
TOP_K = 6

# === Reranking ===
RERANKER_MODEL   = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"
RERANK_TOP_K     = 3
RERANK_THRESHOLD = -5.0