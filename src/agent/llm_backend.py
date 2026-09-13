# -*- coding: utf-8 -*-
"""可插拔 LLM 后端抽象（一切皆插件思想的核心）。

层2/3/4（归纳/创见/多agent）都需要 LLM 推理，但「用本地 8B 还是 API 大模型」
应可一键切换，不改上层逻辑。本模块定义统一接口：

  LLMBackend.generate(messages) -> str

两个实现：
  - LocalVLLMBackend：本地 Qwen3-8B（vLLM :8000）
  - OpenAICompatBackend：任意 OpenAI 兼容 API（qwen3.8-max 等）

上层只用 LLMBackend 抽象，不关心具体后端。
"""
from __future__ import annotations

from typing import Protocol


class LLMBackend(Protocol):
    name: str

    def generate(self, messages: list[dict], max_tokens: int = 1024,
                 temperature: float = 0.1) -> str:
        """输入 messages（OpenAI 格式 list[dict]），返回文本。"""
        ...


class LocalVLLMBackend:
    """本地 Qwen3-8B（vLLM），关思考。"""
    name = "local-qwen3-8b"

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen3-8B"):
        self.base_url = base_url
        self.model = model

    def generate(self, messages, max_tokens=1024, temperature=0.1) -> str:
        import litellm
        try:
            r = litellm.completion(
                model=f"hosted_vllm/{self.model}",
                api_base=self.base_url,
                api_key="sk-no-key-required",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM 调用失败] {str(e)[:200]}"


class OpenAICompatBackend:
    """OpenAI 兼容 API 后端（可插拔，之后换大模型只改这里）。"""
    name = "api"

    def __init__(self, base_url: str = "", api_key: str = "",
                 model: str = "qwen3.8-max"):
        import os
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

    def generate(self, messages, max_tokens=1024, temperature=0.1) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            r = client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                extra_body={"enable_thinking": False},
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM 调用失败] {str(e)[:200]}"


def get_backend(kind: str = "local", **kw) -> LLMBackend:
    """工厂：按 kind 返回后端。kind in {"local","api"}。"""
    if kind == "api":
        return OpenAICompatBackend(**kw)
    return LocalVLLMBackend(**kw)