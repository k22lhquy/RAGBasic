# Playground — Hiểu sâu từng concept RAG

Folder này độc lập với project chính, dùng để thực hành và hiểu rõ từng phần.
Mỗi file chạy được độc lập, có giải thích chi tiết trong code.

```
playground/
├── embedding/
│   ├── 01_cosine_similarity.py   ← Cosine similarity là gì, tại sao dùng nó
│   └── 02_compare_models.py      ← So sánh 2 embedding model, cái nào tốt hơn
│
├── chunking/
│   └── 01_chunking_strategies.py ← Fixed-size vs Recursive, overlap ảnh hưởng gì
│
├── retrieval/
│   └── 01_hit_rate.py            ← Đo Hit Rate để biết search tốt hay không
│
└── token/
    └── 01_token_counting.py      ← Token là gì, đếm token, ước tính chi phí
```

## Cài đặt

```bash
pip install numpy sentence-transformers langchain-text-splitters
pip install langchain-huggingface langchain-chroma chromadb tiktoken
```

## Thứ tự nên chạy

```
1. embedding/01_cosine_similarity.py   → hiểu nền tảng vector search
2. chunking/01_chunking_strategies.py  → hiểu cách chia văn bản
3. token/01_token_counting.py          → hiểu token và chi phí
4. embedding/02_compare_models.py      → so sánh model thực tế
5. retrieval/01_hit_rate.py            → đo chất lượng search
```