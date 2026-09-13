# -*- coding: utf-8 -*-
"""层2 归纳对比 端到端验证：用真实指标库 + 本地 8B 归纳。

用法：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python demo_summarize_compare.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/media/sdb1/gzj/support/llm/rscd")

from plugin_summarize_compare import SummarizeComparePlugin
from llm_backend import get_backend


def main():
    backend = get_backend("local")  # 本地 8B
    plugin = SummarizeComparePlugin(backend=backend)

    questions = [
        "Transformer 类变化检测方法（BIT、ChangeFormer、Changer）各自的核心思路和实验结果对比",
        "Mamba 类方法相比 Transformer 类方法在变化检测里的思路和指标差异",
    ]
    for q in questions:
        print("=" * 70)
        print("问题：", q)
        print("=" * 70)
        r = plugin.run(q)
        print(r.answer)
        print(f"\n[出处 {len(r.sources)} 条] [置信度 {r.confidence}] [拒答 {r.refused}]")
        print()


if __name__ == "__main__":
    main()