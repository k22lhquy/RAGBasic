"""
DEMO: Chunking Strategies — cách chia nhỏ văn bản ảnh hưởng đến RAG

Chạy: python 01_chunking_strategies.py
Cài : pip install langchain-text-splitters
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

SAMPLE_TEXT = """
CHÍNH SÁCH BÁN HÀNG VÀ HỖ TRỢ KHÁCH HÀNG

1. CHÍNH SÁCH HOÀN TIỀN
Khách hàng có thể yêu cầu hoàn tiền trong vòng 30 ngày kể từ ngày mua hàng.
Điều kiện hoàn tiền: sản phẩm phải còn nguyên vẹn, chưa qua sử dụng và còn đầy đủ bao bì.
Để yêu cầu hoàn tiền, khách hàng cần liên hệ bộ phận hỗ trợ qua email support@example.com.
Thời gian xử lý hoàn tiền là 5-7 ngày làm việc sau khi nhận được sản phẩm trả lại.

2. CHÍNH SÁCH BẢO HÀNH
Tất cả sản phẩm điện tử được bảo hành 12 tháng kể từ ngày mua.
Bảo hành không áp dụng cho các trường hợp: hư hỏng do người dùng, ngấm nước.
Khách hàng cần xuất trình hóa đơn mua hàng để được bảo hành.

3. CHÍNH SÁCH GIAO HÀNG
Giao hàng miễn phí cho đơn hàng từ 500.000 đồng trở lên.
Thời gian giao hàng: 2-3 ngày làm việc trong nội thành, 4-7 ngày cho tỉnh thành khác.
""".strip()


def show_chunks(name, chunks, show_content=True):
    print(f"\n{'─'*55}")
    print(f"📦 {name}: {len(chunks)} chunks")
    print(f"{'─'*55}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  [Chunk {i}] độ dài = {len(chunk)} ký tự")
        if show_content:
            preview = chunk[:120].replace('\n', '↵')
            print(f"  '{preview}...'")


# ============================================================
# CÁCH 1: Fixed-size — cắt đúng X ký tự bất kể nội dung
# ============================================================
print("=" * 55)
print("CÁCH 1: Fixed-size (CharacterTextSplitter)")
print("Cắt đúng 300 ký tự, không quan tâm câu đang dở")
print("=" * 55)

splitter1 = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separator=""           # cắt tại bất kỳ ký tự nào
)
chunks1 = splitter1.split_text(SAMPLE_TEXT)
show_chunks("Fixed-size 300", chunks1)

print("\n⚠️  Vấn đề: có thể cắt giữa chừng 1 câu, mất ngữ cảnh")


# ============================================================
# CÁCH 2: Recursive — ưu tiên tách theo đoạn → câu → từ
# ============================================================
print("\n" + "=" * 55)
print("CÁCH 2: Recursive (RecursiveCharacterTextSplitter)")
print("Ưu tiên tách tại: \\n\\n → \\n → . → space")
print("=" * 55)

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks2 = splitter2.split_text(SAMPLE_TEXT)
show_chunks("Recursive 300/50", chunks2)

print("\n✅ Tốt hơn: ưu tiên giữ nguyên đoạn văn, câu hoàn chỉnh")


# ============================================================
# PHẦN 3: Overlap ảnh hưởng thế nào?
# ============================================================
print("\n" + "=" * 55)
print("PHẦN 3: Tại sao cần chunk_overlap?")
print("=" * 55)

# Không overlap
s_no_overlap = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
# Có overlap
s_with_overlap = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=80)

c_no  = s_no_overlap.split_text(SAMPLE_TEXT)
c_yes = s_with_overlap.split_text(SAMPLE_TEXT)

print(f"\nKhông overlap: {len(c_no)} chunks")
print(f"Có overlap 80: {len(c_yes)} chunks (nhiều hơn vì có phần lặp)")

# Hiển thị chỗ nối giữa chunk 1 và 2 để thấy overlap
if len(c_no) >= 2:
    print(f"\n[Không overlap]")
    print(f"  Chunk 1 kết thúc : '...{c_no[0][-60:]}'")
    print(f"  Chunk 2 bắt đầu  : '{c_no[1][:60]}...'")
    print(f"  → Mất thông tin ở chỗ cắt!")

if len(c_yes) >= 2:
    print(f"\n[Có overlap 80 ký tự]")
    print(f"  Chunk 1 kết thúc : '...{c_yes[0][-60:]}'")
    print(f"  Chunk 2 bắt đầu  : '{c_yes[1][:60]}...'")
    print(f"  → Chunk 2 lặp lại phần cuối chunk 1, không bị mất ngữ cảnh")


# ============================================================
# PHẦN 4: Chunk size ảnh hưởng đến chất lượng RAG
# ============================================================
print("\n" + "=" * 55)
print("PHẦN 4: Chunk size nhỏ vs lớn — đánh đổi gì?")
print("=" * 55)

for size in [100, 300, 600]:
    s = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=20)
    c = s.split_text(SAMPLE_TEXT)
    print(f"\nChunk size {size}: → {len(c)} chunks")
    print(f"  Ưu điểm : {'search chính xác hơn, ít token hơn' if size < 300 else 'giữ được ngữ cảnh đầy đủ hơn'}")
    print(f"  Nhược   : {'dễ mất ngữ cảnh, câu bị cắt đứt' if size < 300 else 'nhiều token hơn, có thể nhiễu'}")