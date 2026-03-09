# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# === LLM ===
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
LLM_TEMPERATURE = 0.2

# === Token Control ===
# Tổng token 1 request ≈ CONTEXT_TOKEN_LIMIT + MAX_OUTPUT_TOKENS + prompt overhead (~200)
# Gemma 3n free context window = 8192 tokens
MAX_OUTPUT_TOKENS = 512       # token LLM được phép trả lời
CONTEXT_TOKEN_LIMIT = 3000    # token tối đa dành cho context (chunks)
CHARS_PER_TOKEN = 4           # ước tính: 1 token ≈ 4 ký tự (tiếng Việt thì ~3-4)

# === Embedding ===
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# === Vector Store ===
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "documents"

# === Chunking ===
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# === Retrieval ===
TOP_K = 4  # lấy top K chunks — ảnh hưởng trực tiếp đến token đầu vào