# -*- coding: utf-8 -*-
"""导师评测基准跑分：20 条知识题 + 10 条教学题。

知识题：金标准关键词命中判定（客观）。
教学题：输出原始回答，供 LLM-as-judge / 人工 7 维 rubric 打分（本轮先收集）。

用法：
  source .mentor_env && python mentor_bench.py
"""
from __future__ import annotations

import json
import time

from mentor import MentorAgent, TEACH_SYSTEM

# 20 条知识题（问题, 金标准关键点）
KNOWLEDGE = [
    ("变化检测为什么不用 OA 当主指标？", ["不平衡", "占比"]),
    ("BIT 的核心思想是什么？", ["token", "Transformer", "注意力"]),
    ("ChangeFormer 和 BIT 的本质区别？", ["层次化", "多尺度"]),
    ("双时相影像为什么必须先配准？", ["伪变化"]),
    ("三种基础变化检测融合范式是什么？", ["早融合", "孪生", "差分"]),
    ("SNUNet 相比普通孪生网络改进了什么？", ["密集", "多尺度"]),
    ("变化检测里 Precision 高 Recall 低说明什么？", ["漏检"]),
    ("半监督和无监督变化检测的本质区别？", ["标注", "标签"]),
    ("弱监督变化检测用什么标注？", ["图像级", "弱"]),
    ("跨域变化检测要解决什么问题？", ["分布", "域"]),
    ("LEVIR-CD 是什么类型的数据集？", ["建筑", "变化检测"]),
    ("SECOND 和 LEVIR-CD 的任务区别？", ["语义", "二值"]),
    ("小目标变化漏检严重，常见怎么处理？", ["多尺度"]),
    ("配准误差会导致什么？", ["伪变化"]),
    ("全监督变化检测需要什么标注？", ["像素级"]),
    ("Transformer 相比 CNN 在变化检测里的优势？", ["全局", "长距离"]),
    ("季节变化导致的误检是什么问题？", ["伪变化"]),
    ("F1 和 IoU 的区别？", ["调和", "交并比"]),
    ("多模态变化检测（光学+SAR）的动机？", ["云", "全天候"]),
    ("变化检测为什么不用 OA 做对比指标？", ["不平衡"]),
]

# 10 条教学题（问题, 画像）
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


def judge(answer: str, keywords: list) -> int:
    hit = sum(1 for k in keywords if k.lower() in answer.lower())
    return hit


def main():
    agent = MentorAgent()
    t0 = time.time()

    # 知识题（自动判分）
    print("=" * 60)
    print("【知识题】20 条，金标准关键词命中")
    print("=" * 60)
    ok = 0
    for i, (q, kws) in enumerate(KNOWLEDGE, 1):
        ctx = f"用户是遥感新手。问题：{q} 请直接简洁回答，1-3 句话。"
        ans = agent._chat(TEACH_SYSTEM, ctx, max_tokens=300)
        hit = judge(ans, kws)
        ok += 1 if hit >= 1 else 0
        print(f"[{i}/20] hit={hit}/{len(kws)} {'✓' if hit else '✗'} | {q[:30]}", flush=True)
        print(f"    A: {ans[:100]}", flush=True)
    print(f"\n知识题通过率: {ok}/20 = {ok/20*100:.0f}%", flush=True)

    # 教学题（收集原始回答，本轮不自动判分）
    print("\n" + "=" * 60)
    print("【教学题】10 条，收集原始回答（供 rubric 打分）")
    print("=" * 60)
    teaching_out = []
    for i, (q, persona) in enumerate(TEACHING, 1):
        ctx = f"用户画像：{persona}。问题：{q} 请按教学原则回答（递进、确认理解、可执行下一步）。"
        ans = agent._chat(TEACH_SYSTEM, ctx, max_tokens=800)
        teaching_out.append({"q": q, "persona": persona, "answer": ans})
        print(f"[{i}/10] {q[:30]}", flush=True)
        print(f"    A: {ans[:150]}", flush=True)

    Path = __import__("pathlib").Path
    Path("/media/sdb1/gzj/generated_data/llm/rscd/mentor_teaching_raw.json").write_text(
        json.dumps(teaching_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n总耗时 {time.time()-t0:.0f}s，教学题原始回答已存", flush=True)


if __name__ == "__main__":
    main()
