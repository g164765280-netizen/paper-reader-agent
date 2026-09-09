# -*- coding: utf-8 -*-
"""用 LlamaIndex 的 SentenceSplitter + HuggingFaceEmbedding 重建全文索引。

与 build_fulltext_index.py 输出同格式（chunks.jsonl / bm25.pkl / embeddings.npy / faiss.index），
rag_pipeline.py 无需改动即可读取；区别是切块用 LlamaIndex 成熟实现（带 overlap）。
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")
EMBED = "/media/sdb1/gzj/weights/llm/rscd/bge-m3"


def tokenize(text: str):
    try:
        import jieba  # type: ignore
        return [t for t in jieba.cut(text.lower()) if t.strip()]
    except Exception:
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def table_to_md(rows) -> str:
    if not rows:
        return ""
    lines = []
    for r in rows[:30]:
        cells = [str(c).replace("\n", " ").strip() if c is not None else "" for c in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import Document
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    split = json.loads((CORPUS / "split.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_ids"])

    idx_dir = CORPUS / "index_llama"
    idx_dir.mkdir(parents=True, exist_ok=True)

    splitter = SentenceSplitter(chunk_size=180, chunk_overlap=40)
    embed_model = HuggingFaceEmbedding(model_name=EMBED, device="cuda:1", trust_remote_code=False)

    chunks = []
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        if d["arxiv_id"] not in train_ids:
            continue
        title = d.get("title", "")
        # LlamaIndex 切块
        for pno, page_text in enumerate(d.get("pages_text", [])):
            for node in splitter.get_nodes_from_documents([Document(text=page_text)]):
                t = node.text.strip()
                if len(t) > 30:
                    chunks.append({"paper_id": d["arxiv_id"], "title": title, "page": pno,
                                   "type": "text", "text": t})
        # 表格
        for tb in d.get("tables", []):
            md = table_to_md(tb.get("rows", []))
            if md and len(md) > 20:
                chunks.append({"paper_id": d["arxiv_id"], "title": title, "page": tb.get("page", 0),
                               "type": "table", "text": f"[表格]\n{md}"})

    print(f"llama chunks: {len(chunks)}", flush=True)
    (idx_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    with open(idx_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    texts = [c["text"] for c in chunks]
    embs = []
    B = 32
    for i in range(0, len(texts), B):
        embs.append(embed_model.get_text_embedding_batch(texts[i:i + B], show_progress=False))
    embs = np.vstack(embs).astype("float32")
    # 归一化（内积=余弦）
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    np.save(idx_dir / "embeddings.npy", embs)

    import faiss
    d = embs.shape[1]
    fi = faiss.IndexFlatIP(d)
    fi.add(embs)
    faiss.write_index(fi, str(idx_dir / "faiss.index"))
    print(f"DONE: {len(chunks)} llama chunks indexed", flush=True)


if __name__ == "__main__":
    main()
