# -*- coding: utf-8 -*-
"""跑通体验：模拟一场真实的新生→导师对话，验证连续对话 + 积累蒸馏数据。"""
from __future__ import annotations

import json
import time

from mentor import MentorAgent

DIALOGUE = [
    # (用户消息, 是否先设背景)
    ("我想学遥感变化检测，但不知道怎么开始", None),
    # 导师会问背景 → 用户回答背景
    ("我学过深度学习基础，但没接触过遥感，现在想理解概念", "set_bg"),
    ("变化检测和普通图像分割有什么区别？", None),
    ("那评估变化检测用什么指标？", None),
    ("我想复现一篇经典论文练手，选哪个好？", None),
]


def main():
    agent = MentorAgent(memory_dir="/tmp/mentor_demo")
    uid = "demo_newcomer"
    history = []
    t0 = time.time()
    for msg, flag in DIALOGUE:
        print("=" * 60)
        print(f"👤 用户：{msg}")
        print("-" * 60)
        if flag == "set_bg":
            agent.set_background(uid, "研一新生，学过DL基础，没接触遥感", "理解概念")
            print("（用户已告知背景）")
            continue
        ans = agent.chat(uid, msg)
        print(f"🎓 导师：{ans}")
        history.append({"user": msg, "assistant": ans})
        print()
    # 存对话（将来蒸馏原料）
    import pathlib
    pathlib.Path("/media/sdb1/gzj/generated_data/llm/rscd/mentor_demo_dialogue.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"对话已存，耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
