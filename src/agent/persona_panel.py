# -*- coding: utf-8 -*-
"""多专家会诊人格（Panel）：4 视角专家 + 主持人，单轮会诊。

触发：用户问「XX 和 YY 哪个好 / 有什么区别 / 对比」这类需要多视角的问题。

4 视角：
  - 方法派：只看方法设计思路
  - 指标派：只看实验结果数据
  - 综述派：看领域演进脉络
  - 质疑派：挑刺、找反例
主持人：汇总 4 视角，给结论。

可插拔：LLM 走 LLMBackend（本地 8B / API 切换）。
"""
from __future__ import annotations

from persona_base import PersonaReply
from llm_backend import get_backend

PANEL_VIEWS = [
    ("方法派", "你从【方法设计思路】角度分析：各自的核心机制、创新点、实现复杂度。"),
    ("指标派", "你从【实验结果数据】角度分析：各自在数据集上的指标表现、谁更好、差多少。"),
    ("综述派", "你从【领域演进脉络】角度分析：这两个方法在变化检测发展史里的位置、承上启下关系。"),
    ("质疑派", "你从【挑刺】角度分析：各自的局限、被忽略的问题、可能的反例或翻车场景。"),
]

MODERATOR_PROMPT = """你是会诊主持人。下面 4 位专家从不同角度分析了一个对比问题，请综合他们的观点，给出最终结论。

【问题】{question}

【4 位专家观点】
{views}

请输出：
1. 【结论】一句话回答用户的问题（哪个更好/核心区别是什么）
2. 【关键分歧】专家之间最主要的争议点
3. 【建议】如果用户要选，该怎么选（给出判断依据）"""


class PanelPersona:
    name = "panel"
    system_prompt = "你是多专家会诊主持人，综合方法派/指标派/综述派/质疑派的观点给结论。"

    def __init__(self, backend=None):
        self.llm = backend or get_backend("local")

    def can_activate(self, question: str) -> float:
        kw = ["对比", "比较", "哪个", "区别", "差异", "优缺点", "vs", "versus",
              "谁更好", "优劣", "异同", "更好", "更强", "哪个好"]
        if any(k in question for k in kw):
            return 0.85
        return 0.2

    def respond(self, history: list[dict] | None, question: str) -> PersonaReply:
        # 1. 4 视角各自发言
        views = []
        for view_name, view_instr in PANEL_VIEWS:
            prompt = (
                f"你是遥感变化检测领域的{view_name}专家。\n{view_instr}\n\n"
                f"问题：{question}\n\n"
                f"请用 2-4 句话给出你的分析，直接说观点，不要客套。"
            )
            text = self.llm.generate(
                [{"role": "user", "content": prompt}], max_tokens=300, temperature=0.3
            )
            views.append(f"【{view_name}】{text}")

        # 2. 主持人汇总
        views_text = "\n\n".join(views)
        final = self.llm.generate(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": MODERATOR_PROMPT.format(
                 question=question, views=views_text)}],
            max_tokens=800, temperature=0.2
        )
        # 拼接：主持人结论 + 4 视角原文
        full = final + "\n\n---\n\n### 各专家视角原文\n\n" + views_text
        return PersonaReply(text=full, persona=self.name, tool_used=None,
                            meta={"n_views": len(views)})