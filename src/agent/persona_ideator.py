# -*- coding: utf-8 -*-
"""创见人格（Ideator）：发挥 LLM 基模创造力，提新解决思路。

触发：用户问「还能怎么改进 / 有什么新方向 / 换一种思路 / 未来趋势」。
借鉴 AI co-scientist 的假设生成：先归纳现状，再提出若干可行假设/思路。

可插拔：LLM 走 LLMBackend（本地 8B / API 切换）。
"""
from __future__ import annotations

from persona_base import PersonaReply
from llm_backend import get_backend

IDEATOR_PROMPT = """你是遥感变化检测领域的研究者，要基于领域现状提出【新的解决思路】。

【用户关注的问题】
{question}

【领域现状参考】（可选，来自已有归纳）
{context}

请发挥你的专业知识和创造力，提出：
1. 【现状瓶颈】这个领域当前最核心的未解决难点（1-2 句）
2. 【新思路】提出 2-3 个有依据的新解决方向，每个说明：
   - 核心想法（一句话）
   - 为什么可能有效（依据什么原理/趋势）
   - 相比现有方法的差异点
3. 【可行性排序】按「新颖性 × 可行性」给这 2-3 个思路排个序，说明理由

要求：思路要具体、有依据，不要空喊口号；诚实标注哪些是已有方向的延伸、哪些是真新想法。"""


class IdeatorPersona:
    name = "ideator"
    system_prompt = "你是遥感变化检测领域的研究者，善于基于现状提出有依据的新思路。"

    def __init__(self, backend=None):
        self.llm = backend or get_backend("local")

    def can_activate(self, question: str) -> float:
        kw = ["还能怎么", "改进", "新方向", "新思路", "创新点", "未来", "趋势",
              "换一种思路", "有什么想法", "还能做", "下一步", "拓展"]
        if any(k in question for k in kw):
            return 0.85
        return 0.15

    def respond(self, history: list[dict] | None, question: str,
                context: str = "") -> PersonaReply:
        prompt = IDEATOR_PROMPT.format(question=question, context=context or "（无，请凭专业知识回答）")
        text = self.llm.generate(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": prompt}],
            max_tokens=1000, temperature=0.6  # 创见需要更高温度激发多样性
        )
        return PersonaReply(text=text, persona=self.name, tool_used=None,
                            meta={"context_used": bool(context)})