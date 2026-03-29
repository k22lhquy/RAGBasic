import streamlit as st
import os

from src.indexing import run_indexing, load_vectorstore
from src.embedding import get_embedding_model
from src.rag_chain import build_rag_chain, ask
from src.history import ConversationHistory, build_rephrase_chain

# ========================
# INIT
# ========================
st.set_page_config(page_title="RAG QA", layout="wide")

st.title("📄 RAG - Hỏi đáp tài liệu")

# ========================
# LOAD MODEL + VECTOR DB
# ========================
@st.cache_resource
def init_system():
    embeddings = get_embedding_model()

    if os.path.exists("vectorstore") and os.listdir("vectorstore"):
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = run_indexing(data_dir="data")

    chain, last_docs = build_rag_chain(vectorstore, streaming=False)
    rephrase_chain = build_rephrase_chain()

    return chain, last_docs, rephrase_chain

chain, last_docs, rephrase_chain = init_system()

# ========================
# SESSION STATE
# ========================
if "history" not in st.session_state:
    st.session_state.history = ConversationHistory(max_turns=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ========================
# SHOW CHAT
# ========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ========================
# INPUT
# ========================
if prompt := st.chat_input("Nhập câu hỏi..."):
    # show user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # gọi RAG
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            answer = ask(
                chain,
                last_docs,
                prompt,
                history=st.session_state.history,
                rephrase_chain=rephrase_chain,
                streaming=False
            )

            st.write(answer)

    # lưu lại
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.header("⚙️ Tùy chọn")

    if st.button("🗑 Clear history"):
        st.session_state.messages = []
        st.session_state.history.clear()
        st.success("Đã xóa!")

    st.markdown("---")
    st.write("📂 Folder data/: upload file vào đây rồi restart app")