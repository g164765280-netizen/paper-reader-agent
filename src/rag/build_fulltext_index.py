# -*- coding: utf-8 -*-
"""把解析出的全文+表格分块并建索引（BM25 + bge-m3 + FAISS），输出到 index_fulltext/。

复用 split.json 的留出/训练切分，保证与摘要级评测一致。
chunk 类型：text（正文段落）/ table（表格转 markdown）。
"""
from __future__ import annotations

import hashlib
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
    for r in rows[:30]:  # 限行数，避免超长表格
        cells = [str(c).replace("\n", " ").strip() if c is not None else "" for c in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def chunk_text(text: str, size: int = 150) -> list[str]:
    # 句子级切块：按句末标点/换行切，合并成 ~150 字符的小块，便于精确命中"含数字那句话"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= size:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf.strip():
        chunks.append(buf)
    return chunks


def main():
    split = json.loads((CORPUS / "split.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_ids"])

    idx_dir = CORPUS / "index_fulltext"
    idx_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        if d["arxiv_id"] not in train_ids:
            continue  # 留出论文不进索引
        title = d.get("title", "")
        # 正文分块
        for pno, page_text in enumerate(d.get("pages_text", [])):
            for c in chunk_text(page_text):
                if len(c) > 40:
                    chunks.append({"paper_id": d["arxiv_id"], "title": title, "page": pno,
                                   "type": "text", "text": c})
        # 表格
        for t in d.get("tables", []):
            md = table_to_md(t.get("rows", []))
            if md and len(md) > 20:
                chunks.append({"paper_id": d["arxiv_id"], "title": title, "page": t.get("page", 0),
                               "type": "table", "text": f"[表格]\n{md}"})

    print(f"fulltext chunks: {len(chunks)}", flush=True)
    (idx_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    with open(idx_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    from sentence_transformers import SentenceTransformer
    print("loading bge-m3 ...", flush=True)
    model = SentenceTransformer(EMBED, device="cuda:1")
    embs = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                        batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    np.save(idx_dir / "embeddings.npy", embs.astype("float32"))

    import faiss
    d = embs.shape[1]
    fi = faiss.IndexFlatIP(d)
    fi.add(embs.astype("float32"))
    faiss.write_index(fi, str(idx_dir / "faiss.index"))
    print(f"DONE: {len(chunks)} chunks indexed (train papers only)", flush=True)


if __name__ == "__main__":
    main()
