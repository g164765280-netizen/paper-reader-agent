# -*- coding: utf-8 -*-
"""积累蒸馏数据：3 个不同用户画像 × 各 2-3 轮对话，存成 SFT 训练数据格式。"""
from __future__ import annotations

import json
import pathlib
import time

from mentor import MentorAgent

# (画像, level, goal, 对话问题序列)
PERSONAS = [
    ("会CV没做过遥感", "会CV，没做过变化检测", "理解概念",
     ["变化检测和图像分割有什么区别？", "我想动手做，先复现哪篇？"]),
    ("做变化检测半年想改进", "做变化检测半年", "改进自己的模型",
     ["我的模型小目标漏检严重，怎么排查？", "多尺度特征融合具体怎么做？"]),
    ("要发论文找创新点", "会CV，要发论文", "找创新点",
     ["变化检测现在还有什么可做的创新点？", "跨域变化检测值得做吗？"]),
]


def main():
    agent = MentorAgent(memory_dir="/tmp/mentor_distill")
    t0 = time.time()
    all_data = []  # SFT 格式: {"messages": [{"role":"user",...},{"role":"assistant",...}]}
    for persona, level, goal, questions in PERSONAS:
        uid = f"u_{persona[:4]}"
        print("=" * 60)
        print(f"画像：{persona}（{level}，目标：{goal}）")
        print("=" * 60)
        agent.set_background(uid, level, goal)
        for q in questions:
            ans = agent.chat(uid, q)
            all_data.append({"messages": [
                {"role": "system", "content": "你是遥感科研导师，面向研究生新生。"},
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans},
            ]})
            print(f"  Q: {q}")
            print(f"  A: {ans[:100]}...")
            print()
    out = pathlib.Path("/media/sdb1/gzj/generated_data/llm/rscd/mentor_distill_sft.jsonl")
    out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in all_data), encoding="utf-8")
    print(f"累计 {len(all_data)} 条 SFT 对话，耗时 {time.time()-t0:.0f}s，存至 {out}")


if __name__ == "__main__":
    main()
