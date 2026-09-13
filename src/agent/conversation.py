# -*- coding: utf-8 -*-
"""对话入口：人格调度 + 多轮对话。

人格调度：根据问题选人格（先只 Tutor，后续加 Panel/Ideator）。
调度策略：
  - 普通问题 → Tutor
  - 对比/争议 → Panel（未实现，占位回退 Tutor）
  - 求新思路 → Ideator（未实现，占位回退 Tutor）

多轮：维护全局 history，用户追问能衔接上文。
"""
from __future__ import annotations

from persona_tutor import TutorPersona
from persona_panel import PanelPersona
from persona_ideator import IdeatorPersona


class ConversationalAgent:
    def __init__(self):
        self.tutor = TutorPersona()
        self.panel = PanelPersona()
        self.ideator = IdeatorPersona()
        self.personas = [self.tutor, self.panel, self.ideator]
        self.history: list[dict] = []

    def _route_persona(self, question: str):
        """选人格：创见 → Ideator；对比/争议 → Panel；否则 → Tutor。"""
        ideator_score = self.ideator.can_activate(question)
        if ideator_score >= 0.7:
            return self.ideator, ideator_score
        panel_score = self.panel.can_activate(question)
        if panel_score >= 0.7:
            return self.panel, panel_score
        return self.tutor, self.tutor.can_activate(question)

    def chat(self, question: str) -> str:
        persona, score = self._route_persona(question)
        reply = persona.respond(self.history, question)
        # 更新全局历史
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": reply.text})
        # 打印人格标记（跑通逻辑用，便于看到路由）
        prefix = f"[{persona.name}]"
        if reply.tool_used:
            prefix += f" (调用了 {reply.tool_used})"
        return f"{prefix}\n{reply.text}"

    def reset(self):
        self.history = []
        self.tutor.history = []


if __name__ == "__main__":
    agent = ConversationalAgent()
    print("遥感领域专家已就绪，输入问题（输入 q 退出）：\n")
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "退出", "exit"):
            break
        ans = agent.chat(q)
        print(f"\n{ans}\n")