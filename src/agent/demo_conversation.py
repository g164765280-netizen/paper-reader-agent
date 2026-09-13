# -*- coding: utf-8 -*-
"""S3 对话入口端到端验证：自动问几个问题，测 Tutor 人格 + 能力插件完整链路。

用法：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python demo_conversation.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/media/sdb1/gzj/support/llm/rscd")

from conversation import ConversationalAgent


def main():
    agent = ConversationalAgent()
    questions = [
        "论文《BIT》在 LEVIR-CD 上的 F1 是多少？",      # 数字题 → lookup_metric
        "变化检测为什么不用 OA 当主指标？",             # 概念题 → answer_concept
    ]
    for q in questions:
        print("=" * 70)
        print("用户：", q)
        print("-" * 70)
        ans = agent.chat(q)
        print(ans)
        print()


if __name__ == "__main__":
    main()