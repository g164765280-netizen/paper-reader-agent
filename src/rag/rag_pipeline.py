# -*- coding: utf-8 -*-
"""RAG 检索 + 生成管线（可被评测脚本复用；支持 oracle 模式注入金证据）。

用法：
  python rag_pipeline.py --corpus ... --embedding ... --reranker ... --model ... \
      --query "LEVIR-CD 数据集分辨率多少？" [--oracle-evidence "..." ]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

RAG_SYSTEM = (
    "你是遥感变化检测领域的论文问答助手。请严格基于【证据】回答，逐条用 [1][2]… 标注出处；"
    "答案中的数字、单位、数据集规模必须与证据一致，不得改写或编造；"
    "若证据不足以回答，直接说“根据已有论文无法回答”，不要编造。请使用中文。"
)


def tokenize(text: str):
    try:
        import jieba  # type: ignore
        return [t for t in jieba.cut(text.lower()) if t.strip()]
    except Exception:
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class RAGPipeline:
    def __init__(self, corpus: Path, embedding_dir: str, reranker_dir: str, model_dir: str,
                 adapter_dir: str | None = None):
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self.index_dir = corpus / "index_llama"
        self.chunks = [json.loads(l) for l in (self.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        with open(self.index_dir / "bm25.pkl", "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        self.faiss = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.embs = np.load(self.index_dir / "embeddings.npy")

        print("loading embedding ...", flush=True)
        self.embedder = SentenceTransformer(embedding_dir, device="cuda:1")
        print("loading reranker ...", flush=True)
        self.reranker = CrossEncoder(reranker_dir, device="cuda:1")
        print("loading generator ...", flush=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map={"": 0})
        if adapter_dir:
            from peft import PeftModel
            print(f"loading adapter {adapter_dir} ...", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    def retrieve(self, query: str, top_k: int = 8, rerank_top: int = 4):
        print("  retrieve: dense encode ...", flush=True)
        # dense
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        print("  retrieve: dense done", flush=True)
        _, dense_idx = self.faiss.search(q.astype("float32"), top_k * 4)
        dense_idx = dense_idx[0].tolist()
        # sparse
        bm_scores = self.bm25.get_scores(tokenize(query))
        sparse_idx = np.argsort(-bm_scores)[: top_k * 4].tolist()
        # merge
        cand = list(dict.fromkeys(dense_idx + sparse_idx))
        if not cand:
            return []
        pairs = [(self.chunks[i]["text"], i) for i in cand if 0 <= i < len(self.chunks)]
        if self.reranker and pairs:
            try:
                print(f"  rerank {len(pairs)} pairs ...", flush=True)
                scores = self.reranker.predict([(query, t) for t, _ in pairs])
                order = np.argsort(-np.asarray(scores))[:rerank_top]
                return [self.chunks[pairs[i][1]] for i in order]
            except Exception as ex:
                print(f"  rerank failed ({ex}), fallback to hybrid", flush=True)
        return [self.chunks[i] for _, i in pairs[:rerank_top]]

    def generate(self, query: str, contexts: list[dict], max_new_tokens: int = 512) -> str:
        import torch
        print("  generate: template ...", flush=True)
        if not contexts:
            evidence = "（无检索证据）"
        else:
            evidence = "\n\n".join(
                f"[{i+1}] {c['title']} (p.{c['page']}): {c['text']}" for i, c in enumerate(contexts))
        msgs = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"问题：{query}\n\n证据：\n{evidence}\n\n请回答："},
        ]
        input_ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        print("  generate: model.generate ...", flush=True)
        with torch.inference_mode():
            out = self.model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        print("  generate: decode ...", flush=True)
        ans = self.tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return ans.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--oracle-evidence", default="")
    args = ap.parse_args()

    pipe = RAGPipeline(Path(args.corpus), args.embedding, args.reranker, args.model)
    if args.oracle_evidence:
        ctx = [{"title": "oracle", "page": 0, "text": args.oracle_evidence}]
    else:
        ctx = pipe.retrieve(args.query)
    print("=== retrieved ===")
    for c in ctx:
        print(f"- {c['title']} p.{c['page']}: {c['text'][:120]}")
    print("=== answer ===")
    print(pipe.generate(args.query, ctx))


if __name__ == "__main__":
    main()
