# api.py
"""
RAGCloude REST API — FastAPI

Chạy:
    uvicorn api:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import VECTORSTORE_DIR
from src.embedding import get_embedding_model
from src.indexing import (
    load_vectorstore, run_indexing,
    add_document, delete_document, list_indexed_files,
    build_vectorstore, chunk_documents, load_single_file,
    _load_registry, _save_registry, _file_md5
)
from src.rag_chain import build_rag_chain, ask
from src.history import ConversationHistory, build_rephrase_chain, list_sessions
import json

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAGCloude API",
    description="REST API cho chatbot hỏi đáp tài liệu nội bộ (RAG)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}

# ── Global resources ───────────────────────────────────────────────────────────

embeddings     = None
rephrase_chain = None

# ── Per-session state ──────────────────────────────────────────────────────────
# sessions_state[session_id] = {
#   "vectorstore_dir": str,   ← thư mục vectorstore riêng của session
#   "chain": ...,
#   "last_docs": {...},
#   "files": [filename, ...],  ← danh sách file đã upload vào session này
# }
sessions_state: dict = {}


@app.on_event("startup")
async def startup():
    global embeddings, rephrase_chain
    embeddings     = get_embedding_model()
    rephrase_chain = build_rephrase_chain()
    print("✅ API sẵn sàng!")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_session_dir(session_id: str) -> str:
    return os.path.join("session_stores", session_id)


def _init_session(session_id: str):
    """Khởi tạo state cho 1 session mới"""
    sessions_state[session_id] = {
        "vectorstore_dir": _get_session_dir(session_id),
        "chain":           None,
        "last_docs":       {"docs": []},
        "files":           [],
    }


def _get_session(session_id: str) -> dict:
    if session_id not in sessions_state:
        _init_session(session_id)
    return sessions_state[session_id]


def _rebuild_chain_for_session(session_id: str):
    """Build/rebuild RAG chain từ vectorstore của session"""
    state = _get_session(session_id)
    vs_dir = state["vectorstore_dir"]

    if not os.path.exists(vs_dir) or not os.listdir(vs_dir):
        state["chain"]     = None
        state["last_docs"] = {"docs": []}
        return

    from langchain_chroma import Chroma
    vs = Chroma(
        persist_directory=vs_dir,
        embedding_function=embeddings,
        collection_name="session_docs"
    )
    chain, last_docs = build_rag_chain(vs, streaming=False)
    state["chain"]     = chain
    state["last_docs"] = last_docs


def _add_file_to_session(session_id: str, filepath: str) -> dict:
    """Index 1 file vào vectorstore của session"""
    state    = _get_session(session_id)
    vs_dir   = state["vectorstore_dir"]
    filename = os.path.basename(filepath)

    docs = load_single_file(filepath)
    if not docs:
        return {"status": "error", "filename": filename, "message": "Không đọc được file"}

    chunks = chunk_documents(docs)

    from langchain_chroma import Chroma
    os.makedirs(vs_dir, exist_ok=True)

    if os.path.exists(vs_dir) and os.listdir(vs_dir):
        vs = Chroma(
            persist_directory=vs_dir,
            embedding_function=embeddings,
            collection_name="session_docs"
        )
    else:
        vs = Chroma(
            persist_directory=vs_dir,
            embedding_function=embeddings,
            collection_name="session_docs"
        )

    vs.add_documents(chunks)

    if filename not in state["files"]:
        state["files"].append(filename)

    return {"status": "added", "filename": filename, "chunks": len(chunks)}


# ── Schemas ───────────────────────────────────────────────────────────────────

class NewSessionResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    question:   str        = Field(..., example="Nhân viên được nghỉ phép bao nhiêu ngày?")
    session_id: str        = Field(..., example="abc123")

class ChatResponse(BaseModel):
    answer:     str
    session_id: str
    sources:    list[str]

class UploadResponse(BaseModel):
    status:   str
    filename: str
    chunks:   int  | None = None
    message:  str  | None = None

class SessionFilesResponse(BaseModel):
    session_id: str
    files:      list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status":           "ok",
        "active_sessions":  len(sessions_state),
    }


# ── Session management ────────────────────────────────────────────────────────

@app.post("/sessions/new", response_model=NewSessionResponse, tags=["Sessions"])
def new_session():
    """
    Tạo 1 session mới.
    Mỗi session có vectorstore riêng — hoàn toàn độc lập với các session khác.
    Upload file vào session nào thì chỉ session đó dùng được.
    """
    session_id = uuid.uuid4().hex[:12]
    _init_session(session_id)
    return NewSessionResponse(session_id=session_id)


@app.get("/sessions/{session_id}/files", response_model=SessionFilesResponse, tags=["Sessions"])
def get_session_files(session_id: str):
    """Danh sách file đã upload vào session này"""
    state = _get_session(session_id)
    return SessionFilesResponse(session_id=session_id, files=state["files"])


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: str):
    """Xóa toàn bộ session: vectorstore + lịch sử chat"""
    # Xóa vectorstore của session
    vs_dir = _get_session_dir(session_id)
    if os.path.exists(vs_dir):
        shutil.rmtree(vs_dir, ignore_errors=True)

    # Xóa khỏi RAM
    sessions_state.pop(session_id, None)

    # Xóa khỏi history file
    from src.history import HISTORY_FILE
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_sessions = json.load(f)
        all_sessions.pop(session_id, None)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(all_sessions, f, ensure_ascii=False, indent=2)

    return {"status": "deleted", "session_id": session_id}


@app.get("/sessions", tags=["Sessions"])
def get_all_sessions():
    """Danh sách tất cả session đang active"""
    result = []
    for sid, state in sessions_state.items():
        result.append({
            "session_id":  sid,
            "files":       state["files"],
            "has_chain":   state["chain"] is not None,
        })
    return result


# ── Upload (per-session) ──────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_file_to_session(
    session_id:       str,
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...)
):
    """
    Upload file vào 1 session cụ thể.
    File chỉ có tác dụng trong session đó, không ảnh hưởng session khác.
    Hỗ trợ: .txt, .pdf, .docx, .doc
    """
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{ext}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Đảm bảo session tồn tại
    _get_session(session_id)

    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f_out:
            f_out.write(content)

        result = _add_file_to_session(session_id, tmp_path)

        if result["status"] == "added":
            background_tasks.add_task(_rebuild_chain_for_session, session_id)

        return UploadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.delete("/sessions/{session_id}/files/{filename}", tags=["Documents"])
def remove_file_from_session(session_id: str, filename: str):
    """Xóa 1 file khỏi vectorstore của session"""
    state  = _get_session(session_id)
    vs_dir = state["vectorstore_dir"]

    if filename not in state["files"]:
        raise HTTPException(status_code=404, detail=f"File '{filename}' không có trong session này")

    if not os.path.exists(vs_dir):
        raise HTTPException(status_code=404, detail="Vectorstore của session chưa tồn tại")

    from langchain_chroma import Chroma
    vs      = Chroma(persist_directory=vs_dir, embedding_function=embeddings, collection_name="session_docs")
    results = vs.get()

    ids_to_delete = [
        doc_id for doc_id, meta in zip(results["ids"], results["metadatas"])
        if os.path.basename(meta.get("source", "")) == filename
    ]

    if ids_to_delete:
        vs.delete(ids=ids_to_delete)

    state["files"] = [f for f in state["files"] if f != filename]
    _rebuild_chain_for_session(session_id)

    return {"status": "deleted", "filename": filename, "deleted_chunks": len(ids_to_delete)}


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    """
    Gửi câu hỏi và nhận câu trả lời.
    session_id bắt buộc — tạo session mới trước bằng POST /sessions/new
    """
    state = _get_session(req.session_id)

    if state["chain"] is None:
        raise HTTPException(
            status_code=503,
            detail="Session này chưa có tài liệu nào. Vui lòng upload file trước."
        )

    history = ConversationHistory(max_turns=5, session_id=req.session_id)

    answer = ask(
        state["chain"],
        state["last_docs"],
        req.question,
        history=history,
        rephrase_chain=rephrase_chain,
        show_sources=False,
        streaming=False,
    )

    sources = []
    if state["last_docs"] and state["last_docs"].get("docs"):
        seen = set()
        for doc in state["last_docs"]["docs"]:
            src = doc.metadata.get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append(os.path.basename(src))

    return ChatResponse(
        answer=answer,
        session_id=req.session_id,
        sources=sources,
    )


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/history", tags=["Sessions"])
def get_session_history(session_id: str):
    """Lấy toàn bộ lịch sử hội thoại của session"""
    from src.history import HISTORY_FILE
    if not os.path.exists(HISTORY_FILE):
        return {"session_id": session_id, "turns": []}

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        all_sessions = json.load(f)

    session_data = all_sessions.get(session_id, {})
    return {
        "session_id": session_id,
        "turns":      session_data.get("turns", []),
        "updated_at": session_data.get("updated_at", ""),
    }