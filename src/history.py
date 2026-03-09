# src/history.py
"""
Conversation History với persistent storage.

- Lưu lịch sử vào file JSON → giữ được sau khi tắt chương trình
- Mỗi session có ID riêng → có thể quản lý nhiều cuộc hội thoại
- Rephrase câu hỏi mới dựa trên context hội thoại trước
"""

import json
import os
import uuid
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

# File lưu toàn bộ lịch sử
HISTORY_FILE = "chat_history.json"


class ConversationHistory:
    def __init__(self, max_turns: int = 5, session_id: str = None):
        """
        max_turns  : giữ tối đa N lượt trong bộ nhớ khi gửi LLM
        session_id : ID phiên làm việc, tự tạo nếu không truyền vào
        """
        self.max_turns  = max_turns
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turns      = []   # lượt hội thoại trong RAM
        self._load()           # load lịch sử từ file nếu session_id đã tồn tại

    # ── Lưu / Load ──────────────────────────────────────────

    def _load(self):
        """Load lịch sử của session này từ file JSON"""
        if not os.path.exists(HISTORY_FILE):
            return
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                all_sessions = json.load(f)
            if self.session_id in all_sessions:
                self.turns = all_sessions[self.session_id]["turns"]
                print(f"   📂 Đã load {len(self.turns)} lượt hội thoại từ session '{self.session_id}'")
        except Exception as e:
            print(f"   ⚠️  Không load được history: {e}")

    def _save(self):
        """Ghi toàn bộ turns của session vào file JSON"""
        try:
            # Đọc file hiện tại (nếu có)
            all_sessions = {}
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    all_sessions = json.load(f)

            # Cập nhật session này
            all_sessions[self.session_id] = {
                "session_id":  self.session_id,
                "updated_at":  datetime.now().isoformat(),
                "turns":       self.turns
            }

            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(all_sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"   ⚠️  Không lưu được history: {e}")

    # ── CRUD ────────────────────────────────────────────────

    def add(self, user: str, bot: str):
        """Thêm 1 lượt hội thoại, lưu xuống file ngay"""
        self.turns.append({
            "user": user,
            "bot":  bot,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        self._save()

    def format(self) -> str:
        """Format N lượt gần nhất để nhét vào prompt"""
        recent = self.turns[-self.max_turns:]
        if not recent:
            return "(chưa có lịch sử hội thoại)"
        lines = []
        for t in recent:
            lines.append(f"User: {t['user']}")
            lines.append(f"Bot: {t['bot']}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def clear(self):
        """Xóa history trong RAM và trong file"""
        self.turns = []
        self._save()
        print(f"🗑️  Đã xóa lịch sử session '{self.session_id}'")

    def show_all(self):
        """In toàn bộ lịch sử session hiện tại"""
        if not self.turns:
            print("(Chưa có lịch sử)")
            return
        print(f"\n📜 Lịch sử session '{self.session_id}' ({len(self.turns)} lượt):")
        for i, t in enumerate(self.turns, 1):
            print(f"\n  [{i}] {t.get('time','')} User: {t['user']}")
            print(f"       Bot : {t['bot'][:100]}...")


def list_sessions():
    """Liệt kê tất cả sessions đã lưu"""
    if not os.path.exists(HISTORY_FILE):
        print("Chưa có lịch sử nào được lưu.")
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        all_sessions = json.load(f)
    sessions = list(all_sessions.values())
    print(f"\n📚 Có {len(sessions)} session đã lưu:")
    for s in sessions:
        n = len(s["turns"])
        print(f"  - {s['session_id']} | {n} lượt | cập nhật: {s['updated_at'][:16]}")
    return [s["session_id"] for s in sessions]


def build_rephrase_chain():
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0
    )
    prompt = PromptTemplate(
        input_variables=["history", "question"],
        template="""Dựa vào lịch sử hội thoại bên dưới, hãy viết lại câu hỏi mới nhất
thành 1 câu hỏi độc lập, đầy đủ nghĩa, không cần đọc lịch sử vẫn hiểu được.

Quy tắc:
- Nếu câu hỏi đã rõ ràng, độc lập → giữ nguyên
- Nếu câu hỏi dùng "thế còn", "còn", "vậy thì", đại từ "nó", "đó" → bổ sung context
- Chỉ trả về câu hỏi đã viết lại, không giải thích thêm

Lịch sử hội thoại:
{history}

Câu hỏi mới: {question}

Câu hỏi đã viết lại:"""
    )
    return prompt | llm | StrOutputParser()


def rephrase_question(rephrase_chain, history: ConversationHistory, question: str) -> str:
    if history.is_empty():
        return question
    rephrased = rephrase_chain.invoke({
        "history":  history.format(),
        "question": question
    }).strip()
    if rephrased != question:
        print(f"   🔄 Rephrase: '{question}' → '{rephrased}'")
    return rephrased