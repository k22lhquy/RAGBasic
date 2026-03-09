"""
DEMO: Cosine Similarity — đo độ giống nhau giữa 2 vector

Chạy: python 01_cosine_similarity.py
Cài : pip install numpy
"""

import numpy as np


def cosine_similarity(a, b):
    """
    Công thức: cos(θ) = (A · B) / (|A| × |B|)

    Kết quả:
      1.0  = giống hệt nhau
      0.0  = không liên quan
     -1.0  = đối nghĩa hoàn toàn
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ============================================================
# PHẦN 1: Hiểu bằng vector đơn giản
# ============================================================
print("=" * 55)
print("PHẦN 1: Cosine similarity với vector đơn giản")
print("=" * 55)

# Mỗi chiều = 1 đặc trưng ngữ nghĩa
# [mùa_hè, thời_tiết, nóng, lạnh, tuyết, biển]
vec_hot    = [0.9, 0.8, 0.9, 0.0, 0.0, 0.5]
vec_summer = [0.8, 0.7, 0.7, 0.1, 0.0, 0.6]
vec_snow   = [0.1, 0.6, 0.0, 0.9, 0.9, 0.0]

print(f"\nvec_hot vs vec_summer : {cosine_similarity(vec_hot, vec_summer):.3f}  ← nên cao")
print(f"vec_hot vs vec_snow   : {cosine_similarity(vec_hot, vec_snow):.3f}  ← nên thấp")
print(f"vec_summer vs vec_snow: {cosine_similarity(vec_summer, vec_snow):.3f}  ← nên thấp")


# ============================================================
# PHẦN 2: Tại sao dùng cosine thay vì Euclidean?
# ============================================================
print("\n" + "=" * 55)
print("PHẦN 2: Cosine vs Euclidean distance")
print("=" * 55)

# Câu ngắn và câu dài nhưng cùng nghĩa
vec_short = np.array([1.0, 1.0, 0.0])  # "con mèo"
vec_long  = np.array([4.0, 4.0, 0.0])  # "con mèo con mèo con mèo con mèo"
vec_diff  = np.array([0.0, 0.0, 1.0])  # "xe hơi"

print(f"\nEuclidean (cùng nghĩa, khác độ dài): {np.linalg.norm(vec_short - vec_long):.2f}  ← sai! xa nhau")
print(f"Euclidean (khác nghĩa hoàn toàn)   : {np.linalg.norm(vec_short - vec_diff):.2f}  ← gần hơn dù khác nghĩa")
print(f"\nCosine    (cùng nghĩa, khác độ dài): {cosine_similarity(vec_short, vec_long):.3f}  ← đúng! giống hệt")
print(f"Cosine    (khác nghĩa hoàn toàn)   : {cosine_similarity(vec_short, vec_diff):.3f}  ← đúng! không liên quan")
print("\n→ Cosine đo GÓC giữa 2 vector, không bị ảnh hưởng bởi độ dài văn bản")


# ============================================================
# PHẦN 3: Mô phỏng search trong vector DB
# ============================================================
print("\n" + "=" * 55)
print("PHẦN 3: Mô phỏng tìm chunk liên quan nhất")
print("=" * 55)

# Vector 4 chiều: [nghỉ_lễ, tháng, tiền, sản_phẩm]
chunks = {
    "Tháng 4 nghỉ Giỗ Tổ Hùng Vương":  [0.9, 0.9, 0.0, 0.0],
    "Tháng 9 nghỉ Quốc Khánh 2/9":      [0.8, 0.7, 0.0, 0.0],
    "Hoàn tiền trong vòng 30 ngày":     [0.0, 0.3, 0.9, 0.7],
    "Bảo hành sản phẩm 12 tháng":       [0.0, 0.5, 0.3, 0.9],
    "Giá vé máy bay tháng 4 tăng mạnh": [0.1, 0.8, 0.4, 0.0],
}

query_vec = [0.9, 0.8, 0.0, 0.0]  # "tháng 4 có ngày nghỉ không?"
print(f"\nQuery vector: {query_vec}  → 'tháng 4 có ngày nghỉ không?'")
print("\nScore với từng chunk:")

results = sorted(
    [(cosine_similarity(query_vec, vec), text) for text, vec in chunks.items()],
    reverse=True
)
for i, (score, text) in enumerate(results, 1):
    marker = " ← được chọn" if i == 1 else ""
    print(f"  {i}. score={score:.3f} | {text}{marker}")