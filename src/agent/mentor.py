# -*- coding: utf-8 -*-
"""遥感科研导师 Agent（第一版）。

模型无关（LiteLLM 接任意 OpenAI 兼容端点），三块能力：
  1. 问题理解器（九元组结构化）
  2. 路由（按 goal + domain 决定走哪条路）
  3. 教学式连续对话（JSON 学习状态记忆：记住用户水平/目标/历史，递进 + 可执行下一步）

对应设计：docs/遥感问题理解器.md / docs/教学式对话.md / docs/集成方案.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path

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

规则：没说清就填 unspecified；goal 从意图推断；多个意图取最主要的。
"""

TEACH_SYSTEM = """你是遥感科研导师，面向研究生新生。按以下原则教学：

1. 先了解背景：首次对话先问知识水平/目标，再决定讲多深。
2. 难度递进：从基础到进阶，一步不跳。
3. 确认理解：讲完反问一个具体问题验证。
4. 可执行：建议具体到"复现 XX / 读 XX 论文 / 用什么指标"。
5. 诚实：不确定时明说"我不确定，建议查 XX"，绝不编造方法细节。
6. 有结构：先总览，再分层展开。
7. 溯源：提到方法/结论时给出处（论文/方法名）。

领域知识参考（变化检测）：
- 三种基础融合：早融合(FC-EF) / 孪生拼接(FC-Siam-conc) / 孪生差分(FC-Siam-diff)。
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

# 空状态模板
EMPTY_STATE = {"level": None, "goal": None, "history": []}


class MentorAgent:
    def __init__(self, model: str | None = None, memory_dir: str = "memory",
                 api_key: str | None = None, api_base: str | None = None):
        # qwen3.8-flash 返回干净 content；deepseek-v4-pro 重推理会吃光 max_tokens
        self.model = model or os.getenv("MENTOR_MODEL", "openai/qwen3.8-flash")
        self.api_key = api_key or os.getenv("MENTOR_API_KEY", "")
        self.api_base = api_base or os.getenv("MENTOR_API_BASE", "")
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ---- 模型调用 ----
    def _chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        import litellm
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"enable_thinking": False},  # 关思考，返回干净 content
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        resp = litellm.completion(**kwargs)
        msg = resp.choices[0].message
        text = getattr(msg, "content", None) or getattr(msg, "reasoning_content", "") or ""
        return text.strip()

    # ---- ① 问题理解器 ----
    def understand(self, question: str) -> dict:
        raw = self._chat(UNDERSTAND_SYSTEM, question, temperature=0.0, max_tokens=300)
        try:
            return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except (json.JSONDecodeError, ValueError):
            return {"domain": "other", "task": question, "goal": "other",
                    "scenario": "unspecified", "modality": "unspecified",
                    "supervision": "unspecified", "input": "unspecified",
                    "output": "unspecified", "difficulty": "unspecified", "metric": "unspecified"}

    # ---- ② 路由 ----
    def route(self, parsed: dict) -> str:
        return ROUTE_MAP.get(parsed.get("goal", "other"), ROUTE_MAP["other"])

    # ---- ③ 教学式连续对话 ----
    def chat(self, user_id: str, question: str) -> str:
        state = self._load_state(user_id)

        # 首次：还没有用户水平 → 先问背景
        if state.get("level") is None:
            return ("你好！我是遥感科研导师。开始之前，想先了解你的情况：\n"
                    "1) 你之前接触过深度学习和计算机视觉吗？大概什么水平？\n"
                    "2) 你现在最想做什么（理解概念 / 选方法 / 复现论文 / 改进自己的模型 / 写论文）？")

        # 老用户：带状态教学
        parsed = self.understand(question)
        route = self.route(parsed)
        context = (
            f"用户知识水平：{state.get('level')}\n"
            f"当前目标：{state.get('goal')}\n"
            f"最近对话历史（最多3条）：\n"
            + "".join(f"- 问「{h['q'][:40]}」→ 答「{h['a'][:60]}…」\n" for h in state.get("history", [])[-3:])
            + f"\n本次路由：{route}\n"
            f"九元组：{json.dumps(parsed, ensure_ascii=False)}\n\n"
            f"用户问题：{question}\n\n"
            f"请基于以上历史，递进地讲解（不要重复已讲过的内容），结尾给一个可执行的下一步。"
        )
        answer = self._chat(TEACH_SYSTEM, context)
        self._update_state(user_id, state, question, answer, parsed)
        return answer

    # ---- 记忆（JSON 持久化）----
    def set_background(self, user_id: str, level: str, goal: str):
        state = self._load_state(user_id)
        state["level"] = level
        state["goal"] = goal
        self._save_state(user_id, state)

    def _state_path(self, user_id: str) -> Path:
        return self.memory_dir / f"{user_id}.json"

    def _load_state(self, user_id: str) -> dict:
        p = self._state_path(user_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return dict(EMPTY_STATE)

    def _save_state(self, user_id: str, state: dict):
        self._state_path(user_id).write_text(
            json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    def _update_state(self, user_id: str, state: dict, question: str, answer: str, parsed: dict):
        state.setdefault("history", []).append({"q": question, "a": answer,
                                                 "goal": parsed.get("goal")})
        state["history"] = state["history"][-20:]  # 只保留最近 20 条
        self._save_state(user_id, state)


if __name__ == "__main__":
    import sys
    agent = MentorAgent()
    q = sys.argv[1] if len(sys.argv) > 1 else "变化检测用什么指标评估？"
    print(agent.chat("demo_user", q))
