# -*- coding: utf-8 -*-
"""遥感科研导师 Agent（第一版骨架）。

把设计文档落成可跑的代码，模型无关（LiteLLM 统一接 Claude / GPT-4o / 本地模型）。
三大块：
  1. 问题理解器（九元组结构化）
  2. 路由（按 goal + domain 决定走哪条路）
  3. 教学式对话（学习状态记忆 + 递进 + 确认理解）

对应设计文档：
  - docs/遥感问题理解器.md
  - docs/教学式对话.md
  - docs/集成方案.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---- 提示词 ----

UNDERSTAND_SYSTEM = """你是遥感科研导师的"问题理解器"。把用户的问题结构化成一个 JSON 九元组。
只输出 JSON，不要解释。

字段（字符串，枚举值用英文小写）：
- domain: change_detection / semantic_segmentation / object_detection / image_classification / super_resolution / other
- task: 一句话任务描述（中文）
- scenario: building / vegetation / water / urban / farmland / snow / other / unspecified
- modality: optical / sar / multispectral / hyperspectral / lidar / multimodal / unspecified
- supervision: fully_supervised / semi_supervised / unsupervised / weakly_supervised / cross_domain / open_vocabulary / unspecified
- input: bitemporal / single_temporal / time_series / unspecified
- output: binary_change_map / semantic_change_map / change_caption / unspecified
- difficulty: small_object / class_imbalance / registration / cloud / pseudo_change / compute / unspecified
- metric: f1 / iou / oa / precision / recall / unspecified
- goal: understand_concept / choose_method / reproduce / improve_method / literature_review / write_paper / other

规则：
1. 用户没说清的就填 unspecified，不要猜。
2. goal 是最重要的字段，必须从用户意图推断。
3. 一句话里混多个意图时，取最主要的 goal。
"""

TEACH_SYSTEM = """你是遥感科研导师，面向研究生新生。按以下原则教学：

1. 先了解背景：首次对话先问知识水平/目标，再决定讲多深。
2. 难度递进：从基础到进阶，一步不跳。
3. 确认理解：讲完反问一个具体问题验证。
4. 可执行：建议具体到"复现 XX / 读 XX 论文 / 用什么指标"。
5. 诚实：不确定时明说"我不确定，建议查 XX"，绝不编造方法细节。
6. 有结构：先总览，再分层展开。
7. 溯源：提到方法/结论时给出处（论文/方法名）。

领域知识参考这份知识地图的核心结论：
- 变化检测三种基础融合：早融合(FC-EF)/孪生拼接(FC-Siam-conc)/孪生差分(FC-Siam-diff)。
- Transformer 路线：BIT(token化+全局注意力)、ChangeFormer(层次化+多尺度)。
- 指标：F1 是主指标；OA 因类别不平衡会严重失真，不能当主指标。
- 实验坑：配准误差→伪变化、变化像素占比小→OA失真、小目标→多尺度、季节变化→伪变化。
"""

ROUTE_MAP = {
    "understand_concept": "方法知识地图 + 教学式讲解",
    "choose_method": "方法知识地图 + 文献调研",
    "improve_method": "实验陷阱诊断 + 研究想法迭代",
    "literature_review": "文献调研 + 证据检索",
    "reproduce": "复现指导（读论文→跑代码→对比）",
    "write_paper": "论文工作流",
    "other": "通用答疑",
}

# ---- 学习状态记忆 ----

STATE_TEMPLATE = """# 用户学习状态

## 基本信息
知识水平：{level}
当前目标：{goal}

## 已解释过的概念
{explained}

## 当前卡点 / 未回答的问题
{stuck}

## 推荐下一步
{next_step}
"""


class MentorAgent:
    """模型无关的科研导师 agent。"""

    def __init__(self, model: str | None = None, memory_dir: str = "memory",
                 api_key: str | None = None, api_base: str | None = None):
        # 默认从环境变量读，便于用任意 OpenAI 兼容端点（如 DeepSeek）
        # 注意：deepseek-v4-pro 是重推理模型，思考链会吃光 max_tokens 导致 content 为空；
        #       qwen3.8-flash 返回干净 content，更适合教学式对话。
        self.model = model or os.getenv("MENTOR_MODEL", "openai/qwen3.8-flash")
        self.api_key = api_key or os.getenv("MENTOR_API_KEY", "")
        self.api_base = api_base or os.getenv("MENTOR_API_BASE", "")
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ---- 模型调用（LiteLLM，模型无关）----
    def _chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        try:
            import litellm
        except ImportError:
            raise RuntimeError("需要安装 litellm：pip install litellm")
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,  # 限长，避免第三方网关 504 超时
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        resp = litellm.completion(**kwargs)
        msg = resp.choices[0].message
        # 推理模型 content 可能为 None，回退到 reasoning_content
        text = getattr(msg, "content", None) or getattr(msg, "reasoning_content", "") or ""
        return text.strip()

    # ---- ① 问题理解器 ----
    def understand(self, question: str) -> dict:
        raw = self._chat(UNDERSTAND_SYSTEM, question, temperature=0.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 解析失败兜底
            return {"domain": "other", "task": question, "goal": "other",
                    "scenario": "unspecified", "modality": "unspecified",
                    "supervision": "unspecified", "input": "unspecified",
                    "output": "unspecified", "difficulty": "unspecified",
                    "metric": "unspecified"}

    # ---- ② 路由 ----
    def route(self, parsed: dict) -> str:
        return ROUTE_MAP.get(parsed.get("goal", "other"), ROUTE_MAP["other"])

    # ---- ③ 教学式对话 ----
    def chat(self, user_id: str, question: str) -> str:
        state = self._load_state(user_id)
        parsed = self.understand(question)
        route = self.route(parsed)

        # 首次对话：先问背景
        if state.get("level") is None:
            return ("你好！我是遥感科研导师。在开始之前，想先了解一下："
                    "你之前接触过深度学习和计算机视觉吗？现在是想理解某个概念、"
                    "选一个方法、还是改进你自己的模型？")

        # 组装上下文：学习状态 + 路由方向 + 问题
        context = (
            f"用户知识水平：{state.get('level', '未知')}\n"
            f"当前目标：{state.get('goal', '未知')}\n"
            f"已解释过的概念：{state.get('explained', '无')}\n"
            f"当前卡点：{state.get('stuck', '无')}\n"
            f"本次路由方向：{route}\n"
            f"九元组：{json.dumps(parsed, ensure_ascii=False)}\n\n"
            f"用户问题：{question}\n\n"
            f"请按教学原则回答（先看是否要先确认理解，再递进讲解；"
            f"结尾给出一个可执行的下一步建议）。"
        )
        answer = self._chat(TEACH_SYSTEM, context)

        # 更新记忆（简化版：记录本次问题为卡点，需人工确认）
        self._update_state(user_id, parsed, question)
        return answer

    # ---- 记忆读写 ----
    def _state_path(self, user_id: str) -> Path:
        return self.memory_dir / f"{user_id}.md"

    def _load_state(self, user_id: str) -> dict:
        p = self._state_path(user_id)
        if not p.exists():
            return {}
        # 简化：把 markdown 存成 JSON 附件，或直接读原样文本
        return {"raw": p.read_text(encoding="utf-8")}

    def _update_state(self, user_id: str, parsed: dict, question: str):
        p = self._state_path(user_id)
        existing = self._load_state(user_id).get("raw", "")
        goal = parsed.get("goal", "other")
        new_block = f"- 问题「{question}」（goal={goal}，待确认理解）\n"
        p.write_text(existing + new_block, encoding="utf-8")


if __name__ == "__main__":
    import sys
    agent = MentorAgent(model=os.getenv("MENTOR_MODEL", "gpt-4o"))
    q = sys.argv[1] if len(sys.argv) > 1 else "变化检测用什么指标评估？"
    print(agent.chat("demo_user", q))
