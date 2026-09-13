# -*- coding: utf-8 -*-
"""调度器（Router + Orchestrator）：聪明的架构。

两层：
1. Router.route：先用各插件 can_handle 打分取最高（快、稳、可解释）。
   当最高分与次高分接近（歧义），回退用 LLM 看目录再定（聪明）。
2. Orchestrator.answer：单步直接答；若判为「多步」，可后续扩展为先查数再归纳。

「一切皆插件」：只认 Plugin 契约 + catalog，不关心插件内部。
新增能力 = 在 catalog.build_plugins 注册一条，此处无需改。
"""
from __future__ import annotations

from plugin_base import PluginResult
from llm_backend import get_backend


class Router:
    def __init__(self, plugins: list, llm_backend=None):
        self.plugins = {p.name: p for p in plugins}
        self.llm = llm_backend or get_backend("local")

    def _score_all(self, question: str) -> list[tuple[str, float]]:
        return [(p.name, p.can_handle(question)) for p in self.plugins.values()]

    def route(self, question: str) -> tuple[str, float, str]:
        """返回 (插件名, 分数, 决策方式 rule|llm)。"""
        scores = sorted(self._score_all(question), key=lambda x: -x[1])
        top_name, top_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else -1.0
        # 分数接近（歧义）→ 用 LLM 看目录重新决定
        if top_score - second_score < 0.3 or top_score < 0.5:
            name = self._llm_route(question)
            if name in self.plugins:
                return name, top_score, "llm"
        return top_name, top_score, "rule"

    def _llm_route(self, question: str) -> str:
        """LLM 看能力目录，判断该用哪个工具。失败则返回空（回退规则）。"""
        if self.llm is None:
            return ""
        try:
            cat = "\n".join(
                f"- {name}: {p.description}" for name, p in self.plugins.items()
            )
            prompt = (
                "你是调度器。下面是一些能力的目录，请判断用户问题该用哪个能力。\n"
                "只回答能力名（纯名字，不要解释），从下列名字里选：\n"
                + "\n".join(self.plugins.keys())
                + f"\n\n能力目录：\n{cat}\n\n用户问题：{question}\n\n能力名："
            )
            return self.llm.generate(
                [{"role": "user", "content": prompt}], max_tokens=32, temperature=0
            ).strip()
        except Exception:
            return ""

    def answer(self, question: str, **kwargs) -> PluginResult:
        name, score, how = self.route(question)
        p = self.plugins.get(name)
        if not p:
            return PluginResult(answer="没有匹配的能力", refused=True,
                                meta={"routed_tool": None, "route_how": how})
        r = p.run(question, **kwargs)
        r.meta["routed_tool"] = name
        r.meta["route_score"] = round(score, 3)
        r.meta["route_how"] = how
        return r