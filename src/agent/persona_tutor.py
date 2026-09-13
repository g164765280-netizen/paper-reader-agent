# -*- coding: utf-8 -*-
"""导师人格（Tutor）：常驻、默认人格，接已有能力插件回答。

- 角色：懂遥感变化检测的专家导师。
- 内部：持有一个 Router（能力插件），按问题调 lookup_metric / answer_concept / summarize_compare。
- 多轮：维护对话历史（history），记住聊过什么。
- can_activate：普通问题都归它（几乎总激活，除非明显该走 Panel/Ideator）。
"""
from __future__ import annotations

from persona_base import PersonaReply
from router import Router
from catalog import build_plugins


class TutorPersona:
    name = "tutor"
    system_prompt = (
        "你是遥感变化检测领域的专家导师。用中文回答，简洁专业、循序渐进。"
        "用户可能是研一新生，先讲清概念再展开；涉及论文实验数据要给出准确数值和出处。"
        "不知道就说不知道，不要编造。"
    )

    def __init__(self, router: Router | None = None):
        self.router = router or Router(build_plugins())
        self.history: list[dict] = []

    def can_activate(self, question: str) -> float:
        # 导师默认兜底，几乎总激活
        return 0.8

    def respond(self, history: list[dict] | None, question: str) -> PersonaReply:
        # 1. 用路由器（能力插件）拿到带出处的回答
        result = self.router.answer(question)
        tool = result.meta.get("routed_tool")
        # 2. 如果能力插件答不出/拒答，退回用导师人格泛答（LLM 基模）
        if result.refused or not result.answer.strip():
            text = self._generic_answer(question)
            return PersonaReply(text=text, persona=self.name, tool_used=tool, meta=result.meta)
        return PersonaReply(text=result.answer, persona=self.name,
                            tool_used=tool,
                            meta={"sources": result.sources, "confidence": result.confidence})

    def _generic_answer(self, question: str) -> str:
        """能力插件兜不住时，用 LLM 基模 + 历史简单回答。"""
        from llm_backend import get_backend
        llm = get_backend("local")
        msgs = [{"role": "system", "content": self.system_prompt}]
        for h in self.history[-4:]:
            msgs.append(h)
        msgs.append({"role": "user", "content": question})
        return llm.generate(msgs, max_tokens=600)