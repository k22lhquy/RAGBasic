"""
DEMO: Token counting — token là gì và cách đếm

Chạy: python 01_token_counting.py
Cài : pip install tiktoken
"""

import tiktoken

# ============================================================
# PHẦN 1: Token là gì?
# ============================================================
print("=" * 55)
print("PHẦN 1: Token là gì?")
print("=" * 55)

enc = tiktoken.get_encoding("cl100k_base")  # encoding của GPT-4 / Claude

examples = [
    "Hello",
    "Hello World",
    "Xin chào",
    "Chính sách hoàn tiền",
    "tháng 4 nghỉ mấy ngày?",
    "Khách hàng có thể yêu cầu hoàn tiền trong vòng 30 ngày kể từ ngày mua hàng.",
]

print(f"\n{'Văn bản':<50} | Tokens | Ký tự | Tỉ lệ")
print("-" * 75)
for text in examples:
    tokens = enc.encode(text)
    ratio = len(text) / len(tokens) if tokens else 0
    print(f"{text:<50} | {len(tokens):>6} | {len(text):>5} | {ratio:.1f} ký tự/token")

print("\n→ Tiếng Anh: ~4 ký tự/token")
print("→ Tiếng Việt: ~3-4 ký tự/token (do dấu thanh tốn thêm token)")


# ============================================================
# PHẦN 2: Visualize token — text được cắt như thế nào?
# ============================================================
print("\n" + "=" * 55)
print("PHẦN 2: Text được tokenize như thế nào?")
print("=" * 55)

texts_to_visualize = [
    "hoàn tiền",
    "Chính sách hoàn tiền 30 ngày",
    "RAG retrieval augmented generation",
]

for text in texts_to_visualize:
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]
    print(f"\n'{text}'")
    print(f"  → {len(tokens)} tokens: {decoded}")


# ============================================================
# PHẦN 3: Ước tính chi phí 1 request RAG
# ============================================================
print("\n" + "=" * 55)
print("PHẦN 3: Ước tính token trong 1 request RAG")
print("=" * 55)

PROMPT_TEMPLATE = """Bạn là trợ lý thông minh. Hãy trả lời câu hỏi CHỈ DỰA TRÊN thông tin trong tài liệu.
Nếu thông tin không có trong tài liệu, hãy nói: Tài liệu không đề cập đến vấn đề này.

=== NỘI DUNG TÀI LIỆU ===
{context}

=== CÂU HỎI ===
{question}

=== TRẢ LỜI ==="""

SAMPLE_CONTEXT = """[Đoạn 1 - data/policy.txt]
Khách hàng có thể yêu cầu hoàn tiền trong vòng 30 ngày kể từ ngày mua hàng.
Điều kiện hoàn tiền: sản phẩm phải còn nguyên vẹn, chưa qua sử dụng.

[Đoạn 2 - data/policy.txt]
Tất cả sản phẩm điện tử được bảo hành 12 tháng kể từ ngày mua.
Khách hàng cần xuất trình hóa đơn mua hàng để được bảo hành."""

SAMPLE_QUESTION = "Chính sách hoàn tiền là bao nhiêu ngày?"
SAMPLE_ANSWER = "Theo tài liệu, khách hàng có thể yêu cầu hoàn tiền trong vòng 30 ngày kể từ ngày mua hàng, với điều kiện sản phẩm còn nguyên vẹn và chưa qua sử dụng."

full_prompt = PROMPT_TEMPLATE.format(context=SAMPLE_CONTEXT, question=SAMPLE_QUESTION)

prompt_tokens  = len(enc.encode(full_prompt))
answer_tokens  = len(enc.encode(SAMPLE_ANSWER))
total_tokens   = prompt_tokens + answer_tokens

print(f"\nPrompt template (cố định): ~{len(enc.encode(PROMPT_TEMPLATE.format(context='', question='')))} tokens")
print(f"Context (4 chunks)       : ~{len(enc.encode(SAMPLE_CONTEXT))} tokens")
print(f"Câu hỏi                  : ~{len(enc.encode(SAMPLE_QUESTION))} tokens")
print(f"{'─'*40}")
print(f"Tổng đầu vào (input)     : ~{prompt_tokens} tokens")
print(f"Câu trả lời (output)     : ~{answer_tokens} tokens")
print(f"TỔNG 1 REQUEST           : ~{total_tokens} tokens")

print(f"\n→ Gemma free: 8192 token limit")
print(f"→ Request này dùng {total_tokens/8192*100:.1f}% context window")
print(f"→ Còn trống: {8192 - total_tokens} tokens")