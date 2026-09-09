# -*- coding: utf-8 -*-
"""遥感 RAG Agent：路由式多模态融合 + 工具封装。

输入：问题 + 可选影像（0/1/2 张）
路由：
  - 2 张影像 → detect_change（双时相变化检测）
  - 1 张影像 → analyze_image（影像描述/解读）
  - 无影像  → search_papers（文本 RAG 检索论文）
工具（后续可封装为 MCP server）：
  search_papers / lookup_metric / analyze_image / detect_change
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from PIL import Image

VL_MODEL = "/media/sdb1/yjl/ollama-models/Qwen3-VL-8B-Instruct"


class VLM:
    """Qwen3-VL 封装：看单张/多张影像。"""

    def __init__(self, model_path: str = VL_MODEL):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

    def __call__(self, prompt: str, images: list[str], max_new_tokens: int = 512) -> str:
        imgs = [Image.open(p).convert("RGB") for p in images]
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0], skip_special_tokens=True)


class RS_Agent:
    """遥感 RAG Agent：路由 + 工具。"""

    def __init__(self, rag_pipeline, vlm: VLM, metric_lookup: dict):
        self.rag = rag_pipeline
        self.vlm = vlm
        self.metric_lookup = metric_lookup

    # ---- 工具 ----
    def search_papers(self, question: str, top_k: int = 4) -> str:
        ctx = self.rag.retrieve(question, top_k=top_k)
        return self.rag.generate(question, ctx)

    def lookup_metric(self, paper_id: str, field: str) -> str:
        entries = self.metric_lookup.get(paper_id, {}).get(field.lower(), [])
        if not entries:
            return f"未在论文 {paper_id} 中找到 {field} 指标。"
        parts = [f"{e['number']}（{e['sentence'][:60]}）" for e in entries[:3]]
        return f"论文 {paper_id} 的 {field} 值：" + "；".join(parts)

    def analyze_image(self, image: str, prompt: str | None = None) -> str:
        p = prompt or "请描述这张遥感影像里有什么，属于什么地表类型。"
        return self.vlm(p, [image])

    def detect_change(self, img_a: str, img_b: str) -> str:
        p = ("这是同一区域不同时间的两张遥感影像（第一张 T1，第二张 T2）。"
             "请对比说明两期之间发生了什么变化（建筑新增/拆除、道路、植被等），并指出变化位置。")
        return self.vlm(p, [img_a, img_b])

    def fuse(self, question: str, image: str) -> str:
        """深度融合：检索论文知识 + 看影像，用 Qwen3-VL 综合解读。"""
        ctx = self.rag.retrieve(question, top_k=3)
        evidence = "\n".join(f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(ctx))
        prompt = (
            "你是遥感领域专家。结合以下论文知识与这张遥感影像，回答用户问题。"
            "论文证据：\n" + evidence + "\n\n问题：" + question
        )
        return self.vlm(prompt, [image])

    @staticmethod
    def _colorize(mask_path: str, out: str = "/tmp/change_map_color.png") -> str:
        from PIL import Image
        m = Image.open(mask_path).convert("L")
        rgb = Image.new("RGB", m.size, (235, 235, 235))
        px, mp = rgb.load(), m.load()
        for y in range(m.size[1]):
            for x in range(m.size[0]):
                if mp[x, y] > 128:
                    px[x, y] = (255, 60, 60)
        rgb.save(out)
        return out

    def generate_report(self, method: str, dataset: str, metrics: str, change_map: str) -> str:
        """变化图 → 自动实验报告：看变化图（空间）+ 指标（定量）→ 完整分析报告。"""
        cm = self._colorize(change_map)
        prompt = (
            f"方法 {method} 在 {dataset} 数据集上，检测指标为 {metrics}。"
            "下面这张图里，红色区域是模型检测出的变化区域。"
            "请写一段规范的结果分析报告，包含：总体结论、指标解读（Precision/Recall 权衡）、"
            "空间变化分布描述、误差诊断、改进建议。"
        )
        return self.vlm(prompt, [cm])

    # ---- 路由 ----
    def answer(self, question: str, images: list[str] | None = None) -> dict:
        images = images or []
        if len(images) >= 2:
            tool, out = "detect_change", self.detect_change(images[0], images[1])
        elif len(images) == 1:
            tool, out = "analyze_image", self.analyze_image(images[0], question)
        else:
            tool, out = "search_papers", self.search_papers(question)
        return {"tool": tool, "answer": out}

    def list_tools(self) -> list[dict]:
        return [
            {"name": "search_papers", "desc": "检索论文回答文本问题", "args": ["question"]},
            {"name": "lookup_metric", "desc": "查论文指标数字", "args": ["paper_id", "field"]},
            {"name": "analyze_image", "desc": "描述/解读单张遥感影像", "args": ["image"]},
            {"name": "detect_change", "desc": "双时相变化检测", "args": ["img_a", "img_b"]},
        ]


def load_metric_lookup(corpus: Path) -> dict:
    p = corpus / "metric_lookup.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
