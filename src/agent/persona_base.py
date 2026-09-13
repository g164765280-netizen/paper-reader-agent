# -*- coding: utf-8 -*-
"""人格（Persona）契约 —— 一切皆插件的对话角色抽象。

人格是「有立场、能多轮对话、内部可调能力插件」的对话角色。
与能力插件（Plugin）的区别：
  - 能力插件：无状态，被调一次答一次（lookup_metric / answer_concept）
  - 人格：有 system_prompt 定「它是谁」、维护对话历史、按需调能力插件

契约：
  - name: 人格名
  - system_prompt: 人格系统提示词
  - can_activate(context) -> float: 判断该人格是否应激活（0-1）
  - respond(history, question) -> str: 生成回答
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class PersonaReply:
    text: str
    persona: str = ""
    tool_used: str | None = None  # 内部调用了哪个能力插件
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "persona": self.persona,
                "tool_used": self.tool_used, "meta": self.meta}


class Persona(Protocol):
    name: str
    system_prompt: str

    def can_activate(self, question: str) -> float:
        """判断该人格是否应激活（0-1）。"""
        ...

    def respond(self, history: list[dict], question: str) -> PersonaReply:
        """基于历史 + 问题，生成回答。"""
        ...