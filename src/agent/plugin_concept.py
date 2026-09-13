# -*- coding: utf-8 -*-
"""概念题插件：PaperQA2 检索增强生成（本地 vLLM 8B + bge-m3 + 权威词条）。

- can_handle：概念题（问"为什么/是什么/区别/怎么"等）高分，数字查表题低分。
- run：调 PaperQA2 aquery，返回带出处的答案。
- 置信度：根据引用的 context 条数与相关度估算（占位，后续按 P0 收紧）。
"""
from __future__ import annotations

import asyncio
import re

from plugin_base import PluginResult

# 概念题关键词
CONCEPT_WORDS = ["为什么", "是什么", "区别", "怎么", "如何", "核心思想", "原理",
                 "动机", "本质", "意义", "特点", "概念"]


class ConceptPlugin:
    name = "answer_concept"
    description = "基于论文语料做概念性问答（带出处）"

    def __init__(self):
        # 惰性加载 paperqa（较重，只在真正调用时构建）
        self._docs = None
        self._settings = None
        self._ready = False

    def _ensure_ready(self):
        if self._ready:
            return
        from paperqa import Docs, Settings
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-no-key-required")
        MODEL = "Qwen3-8B"
        llm_config = {"model_list": [{
            "model_name": MODEL,
            "litellm_params": {
                "model": f"hosted_vllm/{MODEL}",
                "api_base": "http://localhost:8000/v1",
                "api_key": "sk-no-key-required",
                "temperature": 0.1,
                "max_tokens": 1024,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        }]}
        self._settings = Settings(
            llm=MODEL, llm_config=llm_config, summary_llm=MODEL, summary_llm_config=llm_config,
            embedding="st-/media/sdb1/gzj/weights/llm/rscd/bge-m3")
        self._settings.parsing.use_doc_details = False
        self._docs = Docs()
        self._ready = True

    def can_handle(self, question: str) -> float:
        q = question.lower()
        has_concept = any(w in question for w in CONCEPT_WORDS)
        has_dataset = any(w in q for w in ["levir", "whu", "sysu", "cdd", "dsifn", "second"])
        has_metric = any(w in q for w in ["f1", "iou", "oa", "precision", "recall", "指标", "值是多少"])
        # 纯问数字/指标 → 低分（让数字题插件接管）
        if has_dataset and has_metric and not has_concept:
            return 0.1
        if has_concept:
            return 0.9
        return 0.5

    async def _aquery(self, question: str) -> PluginResult:
        self._ensure_ready()
        try:
            s = await self._docs.aquery(question, settings=self._settings)
            ans = getattr(s, "answer", None)
            ans_str = str(ans) if ans else ""
            contexts = getattr(s, "contexts", [])
            sources = []
            for ctx in contexts:
                t = getattr(ctx, "text", None)
                name = getattr(t, "name", "") if t else ""
                if name:
                    sources.append(name)
            # 简单置信度：有引用且有内容 → 较高；引用少 → 较低；拒答 → 0
            ref_text = ans_str.lower()
            if "cannot answer" in ref_text or "无法" in ans_str or "查不到" in ans_str:
                return PluginResult(answer=ans_str, sources=sources, confidence=0.0,
                                    refused=True)
            conf = 0.7 if len(sources) >= 3 else (0.5 if sources else 0.3)
            return PluginResult(answer=ans_str, sources=sources, confidence=conf,
                                refused=False, meta={"n_contexts": len(contexts)})
        except Exception as e:
            return PluginResult(answer=f"概念问答出错：{str(e)[:120]}", confidence=0.0,
                                refused=True)

    def run(self, question: str, **kwargs) -> PluginResult:
        return asyncio.run(self._aquery(question))