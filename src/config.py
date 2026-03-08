# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# === LLM ===
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "google/gemma-3n-e4b-it:free"
LLM_TEMPERATURE = 0.2

# === Embedding ===
# Dùng model free local thay vì OpenAI để không cần key riêng
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === Vector Store ===
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "documents"

# === Chunking ===
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# === Retrieval ===
TOP_K = 4  # lấy top 4 chunks liên quan nhất