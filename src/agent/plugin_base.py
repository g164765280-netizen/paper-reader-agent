# -*- coding: utf-8 -*-
"""插件基类：数字题 / 概念题两条通路统一封装为「插件」，由路由器按需调用。

借鉴「一切皆插件」思想：每个插件是独立、可替换、契约统一的模块。
统一契约：
  - name: 插件名
  - description: 能力描述（路由器据此分流）
  - can_handle(question) -> float: 返回 0-1 的「适合度」评分
  - run(question, **kwargs) -> PluginResult: 执行并返回统一结果

PluginResult 统一输出：
  - answer: 答案正文
  - sources: 出处列表（论文/chunk/pqac-id）
  - confidence: 置信度 0-1
  - refused: 是否拒答（信息不足时诚实拒答）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class PluginResult:
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    # 附加信息（插件自定义，如数字题的匹配值、概念题的引用数）
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "refused": self.refused,
            "meta": self.meta,
        }


class Plugin(Protocol):
    name: str
    description: str

    def can_handle(self, question: str) -> float:
        """返回 0-1，表示对该问题的适合度。路由器取最高分者调用。"""
        ...

    def run(self, question: str, **kwargs) -> PluginResult:
        """执行插件，返回统一结果。"""
        ...