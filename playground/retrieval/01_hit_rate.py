"""
DEMO: Retrieval Evaluation — đo chất lượng search bằng Hit Rate

Chạy: python 01_hit_rate.py
Cài : pip install langchain-huggingface langchain-chroma chromadb sentence-transformers
"""

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ============================================================
# Dữ liệu mẫu — giả lập tài liệu chính sách
# ============================================================
DOCUMENTS = [
    "Khách hàng có thể yêu cầu hoàn tiền trong vòng 30 ngày kể từ ngày mua hàng.",
    "Điều kiện hoàn tiền: sản phẩm phải còn nguyên vẹn, chưa qua sử dụng và còn đầy đủ bao bì.",
    "Để yêu cầu hoàn tiền, khách hàng cần liên hệ bộ phận hỗ trợ qua email support@example.com.",
    "Tất cả sản phẩm điện tử được bảo hành 12 tháng kể từ ngày mua.",
    "Bảo hành không áp dụng cho các trường hợp hư hỏng do người dùng hoặc ngấm nước.",
    "Giao hàng miễn phí cho đơn hàng từ 500.000 đồng trở lên.",
    "Thời gian giao hàng nội thành: 2-3 ngày làm việc.",
    "Thời gian giao hàng tỉnh thành khác: 4-7 ngày làm việc.",
    "Thanh toán COD áp dụng cho đơn hàng dưới 10 triệu đồng.",
    "Chấp nhận thanh toán qua thẻ tín dụng, ví điện tử MoMo và ZaloPay.",
]

# Test cases: (câu hỏi, index của document đúng trong DOCUMENTS)
TEST_CASES = [
    ("Chính sách hoàn tiền bao nhiêu ngày?",        0),
    ("Điều kiện để được hoàn tiền là gì?",           1),
    ("Liên hệ hoàn tiền ở đâu?",                    2),
    ("Bảo hành sản phẩm điện tử bao lâu?",          3),
    ("Bảo hành có áp dụng cho hàng bị ngấm nước?",  4),
    ("Đơn hàng bao nhiêu thì được free ship?",      5),
    ("Giao hàng nội thành mất mấy ngày?",           6),
    ("Có thể thanh toán bằng MoMo không?",          9),
]


def build_vectorstore(embedding_model, use_prefix=False):
    """Tạo vectorstore từ DOCUMENTS"""
    if use_prefix:
        texts = [f"passage: {doc}" for doc in DOCUMENTS]
    else:
        texts = DOCUMENTS

    docs = [Document(page_content=t, metadata={"original": DOCUMENTS[i], "idx": i})
            for i, t in enumerate(texts)]

    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name="eval_test"
    )


def evaluate(model_name, use_prefix=False, top_k=4):
    """Đo hit rate của 1 model"""
    print(f"\n{'='*55}")
    print(f"Đánh giá: {model_name}  (top_k={top_k})")
    print(f"{'='*55}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vs = build_vectorstore(embeddings, use_prefix)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    hits = 0
    for question, correct_idx in TEST_CASES:
        query = f"query: {question}" if use_prefix else question
        results = retriever.invoke(query)

        # Lấy index của các doc tìm được
        found_indices = [doc.metadata.get("idx") for doc in results]
        hit = correct_idx in found_indices
        hits += hit

        status = "✅" if hit else "❌"
        print(f"  {status} '{question[:45]}'")
        if not hit:
            print(f"     Cần doc [{correct_idx}], tìm được: {found_indices}")

    hit_rate = hits / len(TEST_CASES) * 100
    print(f"\n  📊 Hit Rate: {hits}/{len(TEST_CASES)} = {hit_rate:.0f}%")
    return hit_rate


# So sánh 2 model
r1 = evaluate("sentence-transformers/all-MiniLM-L6-v2", use_prefix=False)
r2 = evaluate("intfloat/multilingual-e5-base", use_prefix=True)

print(f"\n{'='*55}")
print("KẾT QUẢ SO SÁNH")
print(f"{'='*55}")
print(f"  all-MiniLM-L6-v2     : {r1:.0f}%")
print(f"  multilingual-e5-base : {r2:.0f}%")
winner = "multilingual-e5-base" if r2 > r1 else "all-MiniLM-L6-v2"
print(f"\n  → {winner} tốt hơn cho tài liệu tiếng Việt này")