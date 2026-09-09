# -*- coding: utf-8 -*-
"""用摘要（metadata.jsonl）快速建索引，跳过 PDF，立即跑通 RAG + 评测。

与 prepare_corpus.py 产出相同结构（split.json + index/），供后续脚本复用。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np


def tokenize(text: str):
    try:
        import jieba  # type: ignore
        return [t for t in jieba.cut(text.lower()) if t.strip()]
    except Exception:
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--heldout-ratio", type=float, default=0.40)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    index_dir = corpus / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = corpus / "metadata_abstracts.jsonl"
    if not meta.exists():
        meta = corpus / "metadata.jsonl"
    papers = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("abstract"):
                papers.append(rec)
    print(f"papers with abstract: {len(papers)}", flush=True)

    heldout, train = [], []
    for p in papers:
        h = int(hashlib.sha256(p["arxiv_id"].encode()).hexdigest(), 16) % 1000
        (heldout if h < int(args.heldout_ratio * 1000) else train).append(p)
    print(f"heldout {len(heldout)} / train {len(train)}", flush=True)

    split = {
        "heldout_ratio": args.heldout_ratio,
        "heldout_ids": sorted(p["arxiv_id"] for p in heldout),
        "train_ids": sorted(p["arxiv_id"] for p in train),
        "heldout_meta": heldout,
        "train_meta": train,
    }
    (corpus / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    chunks = [{"paper_id": p["arxiv_id"], "title": p["title"], "page": 0,
               "text": f"Title: {p['title']}\nAbstract: {p['abstract']}"} for p in train]
    (index_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    from sentence_transformers import SentenceTransformer
    print("loading bge-m3 ...", flush=True)
    model = SentenceTransformer(args.embedding)
    embs = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                        batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    np.save(index_dir / "embeddings.npy", embs.astype("float32"))

    import faiss
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embs.astype("float32"))
    faiss.write_index(idx, str(index_dir / "faiss.index"))
    print(f"DONE: {len(chunks)} abstract chunks indexed", flush=True)


if __name__ == "__main__":
    main()
