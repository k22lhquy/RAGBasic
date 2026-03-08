# reindex.py
"""
Chạy script này khi muốn index lại từ đầu (thêm tài liệu mới)
"""
import shutil
import os
from src.indexing import run_indexing

if os.path.exists("vectorstore"):
    print("🗑️  Xóa VectorDB cũ...")
    shutil.rmtree("vectorstore")

run_indexing(data_dir="data")
print("\n💡 Chạy 'python main.py' để bắt đầu hỏi đáp")