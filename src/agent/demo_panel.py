# -*- coding: utf-8 -*-
"""S4 Panel 多专家会诊端到端验证。

用法：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python demo_panel.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/media/sdb1/gzj/support/llm/rscd")

from persona_panel import PanelPersona


def main():
    panel = PanelPersona()
    q = "BIT 和 ChangeFormer 在变化检测里哪个更好？"
    print("问题：", q)
    print("=" * 70)
    reply = panel.respond([], q)
    print(reply.text)


if __name__ == "__main__":
    main()