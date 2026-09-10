# -*- coding: utf-8 -*-
"""教学题重跑（max_tokens=800）+ 7 维 rubric 打分（LLM-as-judge）。"""
from __future__ import annotations

import json
import time

from mentor import MentorAgent, TEACH_SYSTEM

TEACHING = [
    ("变化检测和普通图像分割有什么区别？", "研一新生，只学过DL基础没接触遥感"),
    ("我刚学完U-Net，想做变化检测，从哪开始？", "会CV没做过变化检测"),
    ("我的模型小目标漏检严重，怎么排查？", "做变化检测半年"),
    ("什么是Transformer？为什么变化检测用它？", "研一新生"),
    ("BIT和ChangeFormer我该用哪个？", "会CV"),
    ("我没有标注数据，能做变化检测吗？", "新手"),
    ("为什么大家用F1不用OA？", "研一新生"),
    ("变化检测还有什么可做的创新点？", "会CV要发论文"),
    ("我看不懂变化检测论文里的指标表", "新手"),
    ("我不懂什么叫语义变化检测", "半途转行"),
]

JUDGE_PROMPT = """你是导师回复质量的评审。对下面的导师回复，从 7 个维度各打 0-3 分（3=好，1=差，0=缺失）。
只输出 JSON，格式：{{"先了解背景":n,"难度递进":n,"确认理解":n,"可执行":n,"诚实":n,"有结构":n,"溯源":n}}

7 维标准：
- 先了解背景：是否先问/确认用户知识水平与目标
- 难度递进：是否从基础到进阶、不跳步
- 确认理解：是否用反问/例子验证用户是否懂
- 可执行：建议是否具体（复现XX/读XX论文/用什么指标）
- 诚实：不确定时是否明说，不编造
- 有结构：是否先总览再分层，有脉络
- 溯源：提到方法/结论是否给出处

用户画像：{persona}
问题：{q}
导师回复：
{answer}
"""


def main():
    agent = MentorAgent()
    t0 = time.time()
    results = []
    for i, (q, persona) in enumerate(TEACHING, 1):
        ctx = f"用户画像：{persona}。问题：{q} 请按教学原则回答（递进、确认理解、可执行下一步）。"
        ans = agent._chat(TEACH_SYSTEM, ctx, max_tokens=800)
        # 判分
        judge_in = JUDGE_PROMPT.format(persona=persona, q=q, answer=ans[:1500])
        try:
            jraw = agent._chat("你是评审。", judge_in, max_tokens=200, temperature=0.0)
            jd = json.loads(jraw[jraw.find("{"):jraw.rfind("}") + 1])
        except Exception:
            jd = {}
        scores = [jd.get(k, 0) for k in ["先了解背景", "难度递进", "确认理解", "可执行", "诚实", "有结构", "溯源"]]
        avg = sum(scores) / 7 if scores else 0
        results.append({"q": q, "persona": persona, "answer": ans, "scores": jd, "avg": round(avg, 2)})
        print(f"[{i}/10] avg={avg:.2f} | {q[:30]}", flush=True)
        print(f"    A: {ans[:120]}", flush=True)
    overall = sum(r["avg"] for r in results) / len(results)
    print(f"\n教学题 7 维平均分: {overall:.2f} / 3.0", flush=True)
    __import__("pathlib").Path("/media/sdb1/gzj/generated_data/llm/rscd/mentor_teaching_scored.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"总耗时 {time.time()-t0:.0f}s，结果已存", flush=True)


if __name__ == "__main__":
    main()
