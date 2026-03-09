"""
DEMO: So sánh 2 embedding model — cái nào tốt hơn cho tiếng Việt?

Chạy: python 02_compare_models.py
Cài : pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# Bộ test cases: (câu hỏi, câu liên quan, câu không liên quan)
# ============================================================
TEST_CASES = [
    (
        "tháng 4 nghỉ mấy ngày?",
        "Nghỉ lễ Giỗ Tổ Hùng Vương ngày 18 tháng 4 và nghỉ 30/4",   # liên quan
        "Hướng dẫn cài đặt phần mềm diệt virus trên Windows",         # không liên quan
    ),
    (
        "chính sách hoàn tiền như thế nào?",
        "Khách hàng được hoàn tiền trong vòng 30 ngày kể từ ngày mua", # liên quan
        "Dự báo thời tiết hôm nay có mưa to ở miền Bắc",              # không liên quan
    ),
    (
        "bảo hành bao lâu?",
        "Sản phẩm được bảo hành 12 tháng kể từ ngày mua hàng",        # liên quan
        "Công thức nấu phở bò truyền thống Hà Nội",                   # không liên quan
    ),
]


def evaluate_model(model_name, use_prefix=False):
    """
    Chạy toàn bộ test cases với 1 model.
    Trả về hit_rate và avg_gap (khoảng cách score liên quan vs không liên quan).
    """
    print(f"\n{'='*55}")
    print(f"Model: {model_name}")
    print(f"{'='*55}")

    model = SentenceTransformer(model_name)
    hits = 0
    gaps = []

    for query, related, unrelated in TEST_CASES:
        if use_prefix:
            q_vec   = model.encode(f"query: {query}")
            r_vec   = model.encode(f"passage: {related}")
            ur_vec  = model.encode(f"passage: {unrelated}")
        else:
            q_vec   = model.encode(query)
            r_vec   = model.encode(related)
            ur_vec  = model.encode(unrelated)

        score_related   = cosine_similarity(q_vec, r_vec)
        score_unrelated = cosine_similarity(q_vec, ur_vec)
        gap = score_related - score_unrelated
        gaps.append(gap)

        hit = score_related > score_unrelated
        hits += hit
        status = "✅" if hit else "❌"
        print(f"\n  {status} Q: '{query}'")
        print(f"     liên quan   : {score_related:.3f}  '{related[:45]}...'")
        print(f"     ko liên quan: {score_unrelated:.3f}  '{unrelated[:45]}...'")
        print(f"     gap         : {gap:+.3f}  {'← tốt' if gap > 0.2 else '← yếu'}")

    hit_rate = hits / len(TEST_CASES) * 100
    avg_gap  = np.mean(gaps)
    print(f"\n  📊 Hit Rate: {hits}/{len(TEST_CASES)} = {hit_rate:.0f}%")
    print(f"  📊 Avg Gap : {avg_gap:.3f}  (càng cao càng phân biệt tốt)")
    return hit_rate, avg_gap


# Chạy so sánh
results = {}

hit1, gap1 = evaluate_model("sentence-transformers/all-MiniLM-L6-v2", use_prefix=False)
results["all-MiniLM-L6-v2"] = (hit1, gap1)

hit2, gap2 = evaluate_model("intfloat/multilingual-e5-base", use_prefix=True)
results["multilingual-e5-base"] = (hit2, gap2)


# Bảng tổng kết
print(f"\n{'='*55}")
print("TỔNG KẾT SO SÁNH")
print(f"{'='*55}")
print(f"{'Model':<28} | {'Hit Rate':>8} | {'Avg Gap':>8}")
print("-" * 55)
for model, (hit, gap) in results.items():
    winner = " ← tốt hơn" if gap == max(g for _, g in results.values()) else ""
    print(f"{model:<28} | {hit:>7.0f}% | {gap:>8.3f}{winner}")