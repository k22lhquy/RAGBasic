# src/indexing.py
import os
from src.embedding import get_embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL, VECTORSTORE_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP
)


def load_documents(data_dir: str = "data"):
    documents = []

    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    print(f"📁 File tìm thấy: {all_files}")

    for fpath in all_files:
        if fpath.lower().endswith(".txt"):
            loaded = False
            for encoding in ["utf-8", "utf-8-sig", "cp1258", "cp1252", "latin-1"]:
                try:
                    with open(fpath, "r", encoding=encoding) as f:
                        text = f.read()
                    doc = Document(page_content=text, metadata={"source": fpath})
                    documents.append(doc)
                    print(f"  ✅ TXT [{encoding}]: {fpath}")
                    loaded = True
                    break
                except Exception as e:
                    continue
            if not loaded:
                print(f"  ❌ Không đọc được: {fpath}")

        elif fpath.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(fpath)
                docs = loader.load()
                documents.extend(docs)
                print(f"  ✅ PDF: {fpath} ({len(docs)} trang)")
            except Exception as e:
                print(f"  ❌ Lỗi PDF {fpath}: {e}")

    print(f"✅ Tổng cộng: {len(documents)} document")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Chunked: {len(chunks)} chunks")
    return chunks




def build_vectorstore(chunks, embeddings):
    print("⏳ Đang embed và lưu vào VectorDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"✅ Đã lưu {len(chunks)} chunks")
    return vectorstore


def load_vectorstore(embeddings):
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


def run_indexing(data_dir: str = "data"):
    print("=" * 50)
    print("🚀 BẮT ĐẦU INDEXING PIPELINE")
    print("=" * 50)

    documents = load_documents(data_dir)
    if not documents:
        print("❌ Không load được tài liệu nào!")
        return None

    chunks = chunk_documents(documents)
    embeddings = get_embedding_model()
    vectorstore = build_vectorstore(chunks, embeddings)

    print("=" * 50)
    print("✅ INDEXING HOÀN TẤT")
    print("=" * 50)
    return vectorstore