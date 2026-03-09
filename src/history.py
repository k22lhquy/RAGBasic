# src/history.py
"""
Conversation History:
  Lưu lịch sử hội thoại và rephrase câu hỏi mới
  dựa trên context hội thoại trước đó.

Flow:
  history + câu hỏi mới → LLM rephrase → câu hỏi độc lập → retrieve → trả lời
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL


class ConversationHistory:
    """Lưu và quản lý lịch sử hội thoại"""

    def __init__(self, max_turns: int = 5):
        """
        max_turns: giữ tối đa bao nhiêu lượt hội thoại gần nhất
        Không giữ quá nhiều vì tốn token và dễ nhiễu
        """
        self.max_turns = max_turns
        self.turns = []   # list of {"user": ..., "bot": ...}

    def add(self, user: str, bot: str):
        """Thêm 1 lượt hội thoại vào history"""
        self.turns.append({"user": user, "bot": bot})
        # Chỉ giữ max_turns lượt gần nhất
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def format(self) -> str:
        """Format history thành chuỗi để nhét vào prompt"""
        if not self.turns:
            return "(chưa có lịch sử hội thoại)"
        lines = []
        for i, turn in enumerate(self.turns, 1):
            lines.append(f"User: {turn['user']}")
            lines.append(f"Bot: {turn['bot']}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def clear(self):
        self.turns = []


def build_rephrase_chain():
    """
    Chain dùng LLM để rephrase câu hỏi mới thành câu độc lập
    dựa trên lịch sử hội thoại.

    Ví dụ:
      history: User hỏi về tháng 4 nghỉ lễ, Bot trả lời 4 ngày
      câu mới: "thế còn tháng 5?"
      → rephrase: "tháng 5 có mấy ngày nghỉ lễ?"
    """
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
- Nếu câu hỏi đã rõ ràng, độc lập → giữ nguyên, không thay đổi
- Nếu câu hỏi dùng "thế còn", "còn", "vậy thì", đại từ như "nó", "đó" → bổ sung context từ lịch sử
- Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm

Lịch sử hội thoại:
{history}

Câu hỏi mới: {question}

Câu hỏi đã viết lại:"""
    )

    return prompt | llm | StrOutputParser()


def rephrase_question(rephrase_chain, history: ConversationHistory, question: str) -> str:
    """
    Nếu có history → rephrase câu hỏi
    Nếu chưa có history → dùng câu hỏi gốc luôn (không cần gọi LLM)
    """
    if history.is_empty():
        return question   # câu đầu tiên, không cần rephrase

    rephrased = rephrase_chain.invoke({
        "history": history.format(),
        "question": question
    }).strip()

    if rephrased != question:
        print(f"   🔄 Rephrase: '{question}' → '{rephrased}'")

    return rephrased