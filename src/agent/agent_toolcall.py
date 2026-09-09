# -*- coding: utf-8 -*-
"""自主 tool-calling agent（显存优化版）：Qwen3-8B 只加载一次，VLM 惰性加载。"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")
EMB = "/media/sdb1/gzj/weights/llm/rscd/bge-m3"
RERANK = "/media/sdb1/gzj/weights/llm/rscd/bge-reranker-v2-m3"
TEXT_MODEL = "/media/sdb1/yjl/ollama-models/Qwen3-8B"

SYSTEM = "你是遥感领域助手，可以调用工具回答问题。涉及影像必须调用影像工具；查论文指标用 lookup_metric；否则 search_papers。"

TOOLS = [
    {"type": "function", "function": {"name": "search_papers", "description": "检索论文回答文本问题",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "lookup_metric", "description": "查论文指标数值",
        "parameters": {"type": "object", "properties": {"paper_id": {"type": "string"}, "field": {"type": "string"}}, "required": ["paper_id", "field"]}}},
    {"type": "function", "function": {"name": "analyze_image", "description": "描述单张遥感影像",
        "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]}}},
    {"type": "function", "function": {"name": "detect_change", "description": "对比两张影像说明变化",
        "parameters": {"type": "object", "properties": {"image_a": {"type": "string"}, "image_b": {"type": "string"}}, "required": ["image_a", "image_b"]}}},
    {"type": "function", "function": {"name": "fuse", "description": "论文知识+影像融合解读",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "image_path": {"type": "string"}}, "required": ["question", "image_path"]}}},
    {"type": "function", "function": {"name": "generate_report", "description": "变化图→实验报告",
        "parameters": {"type": "object", "properties": {"method": {"type": "string"}, "dataset": {"type": "string"}, "metrics": {"type": "string"}, "change_map": {"type": "string"}}, "required": ["method", "dataset", "metrics", "change_map"]}}},
]


def _tokenize(t):
    try:
        import jieba
        return [x for x in jieba.cut(t.lower()) if x.strip()]
    except Exception:
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", t.lower())


class ToolCallAgent:
    def __init__(self):
        # 1. Qwen3-8B 只加载一次（orchestrator + 生成器）
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        # 2. 检索（bge + faiss + bm25，无 Qwen）
        idx = CORPUS / "index_llama"
        self.chunks = [json.loads(l) for l in (idx / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        with open(idx / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        import faiss
        self.faiss = faiss.read_index(str(idx / "faiss.index"))
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.embedder = SentenceTransformer(EMB, device="cuda:1")
        self.reranker = CrossEncoder(RERANK, device="cuda:1")
        # 3. 指标表
        self.lookup = json.loads((CORPUS / "metric_lookup.json").read_text(encoding="utf-8")) if (CORPUS / "metric_lookup.json").exists() else {}
        # 4. VLM 惰性加载
        self._vlm = None
        # 5. 多轮对话历史
        self.history: list = []

    @property
    def vlm(self):
        if self._vlm is None:
            from agent import VLM
            self._vlm = VLM()
        return self._vlm

    # ---- 检索 + 生成 ----
    def _retrieve(self, q, top_k=4):
        qv = self.embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        _, di = self.faiss.search(qv.astype("float32"), top_k * 4)
        di = di[0].tolist()
        bs = self.bm25.get_scores(_tokenize(q))
        si = np.argsort(-bs)[: top_k * 4].tolist()
        cand = list(dict.fromkeys(di + si))
        pairs = [(self.chunks[i]["text"], i) for i in cand if 0 <= i < len(self.chunks)]
        sc = self.reranker.predict([(q, t) for t, _ in pairs])
        order = np.argsort(-np.asarray(sc))[:top_k]
        return [self.chunks[pairs[i][1]] for i in order]

    def _generate(self, q, contexts):
        ev = "\n\n".join(f"[{i+1}] {c['title']} (p.{c['page']}): {c['text']}" for i, c in enumerate(contexts)) if contexts else "（无证据）"
        msgs = [{"role": "system", "content": "你是遥感论文问答助手，严格基于证据，逐条标[n]，数字照抄，答不出说无法回答。"},
                {"role": "user", "content": f"问题：{q}\n\n证据：\n{ev}"}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=256, do_sample=False)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    # ---- 工具 ----
    def _exec(self, name, args):
        if name == "search_papers":
            return self._generate(args.get("question", ""), self._retrieve(args.get("question", "")))
        if name == "lookup_metric":
            e = self.lookup.get(args.get("paper_id", ""), {}).get(args.get("field", "").lower(), [])
            return "；".join(f"{x['number']}" for x in e[:3]) if e else "未找到"
        if name == "analyze_image":
            return self.vlm("请描述这张遥感影像里有什么，属于什么地表类型。", [args.get("image_path", "")])
        if name == "detect_change":
            return self.vlm("这是同一区域T1/T2两张影像，请说明发生了什么变化及位置。", [args.get("image_a", ""), args.get("image_b", "")])
        if name == "fuse":
            ctx = self._retrieve(args.get("question", ""), top_k=3)
            ev = "\n".join(f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(ctx))
            return self.vlm("结合论文知识解读影像：\n" + ev + "\n\n问题：" + args.get("question", ""), [args.get("image_path", "")])
        if name == "generate_report":
            from PIL import Image
            m = Image.open(args.get("change_map", "")).convert("L")
            rgb = Image.new("RGB", m.size, (235, 235, 235)); px, mp = rgb.load(), m.load()
            for y in range(m.size[1]):
                for x in range(m.size[0]):
                    if mp[x, y] > 128: px[x, y] = (255, 60, 60)
            p = "/tmp/cm.png"; rgb.save(p)
            return self.vlm(f"方法{args.get('method')}在{args.get('dataset')}上指标{args.get('metrics')}。红色是变化区域，写规范结果分析报告。", [p])
        return f"未知工具 {name}"

    def _parse(self, resp):
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", resp, re.DOTALL) or re.search(r"(\{\"name\":.*?\})", resp, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(1)); return d.get("name"), d.get("arguments", {})
            except Exception: pass
        return None, None

    def chat(self, q, max_turns=5, use_history=True):
        if not self.history:
            self.history = [{"role": "system", "content": SYSTEM}]
        msgs = list(self.history)
        msgs.append({"role": "user", "content": q})
        resp = ""
        for _ in range(max_turns):
            text = self.tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)
            ids = self.tok(text, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(**ids, max_new_tokens=512, do_sample=False)
            resp = self.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            name, args = self._parse(resp)
            if name:
                r = self._exec(name, args)
                msgs.append({"role": "assistant", "content": resp})
                msgs.append({"role": "tool", "content": r, "name": name})
                continue
            break
        # 持久化历史（供下一轮多轮对话）
        if use_history:
            self.history = msgs + [{"role": "assistant", "content": resp}]
        return resp.strip()

    def reset(self):
        self.history = []


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "LEVIR-CD 数据集的训练样本数是多少？"
    print("=== 问题 ===", q)
    print("=== 回答 ===")
    print(ToolCallAgent().chat(q))
