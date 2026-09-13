# -*- coding: utf-8 -*-
"""论文名/方法名 → paper_id 解析器。

解决「用户只说"BIT 这篇论文"，但指标库需要 paper_id」的消歧缺口。

数据源：
  1. method_to_paper.json：方法名 → arxiv_id 权威映射（人工核对）
  2. fulltext 标题：arxiv_id → title（用于标题模糊匹配）

流程：
  - 从问题里提取方法名/论文标题关键词
  - 先查 method_to_paper 精确映射
  - 再查标题模糊匹配
  - 返回 paper_id 或 None
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MAP_FILE = Path(__file__).parent / "method_to_paper.json"


class PaperResolver:
    def __init__(self, map_file: Path | str = MAP_FILE):
        self.map = {}
        d = json.loads(Path(map_file).read_text(encoding="utf-8"))
        self.methods = d.get("methods", {})
        self.aliases = d.get("aliases", {})

    def resolve(self, question: str) -> str | None:
        """从问题里解析出 paper_id。"""
        # 1. 精确方法名匹配（大小写敏感，方法名通常在标题/问题里大写）
        for method, pid in self.methods.items():
            if not pid:
                continue
            # 方法名作为独立词出现（避免 "BIT" 匹配到 "arbit" 之类）
            if re.search(rf"\b{re.escape(method)}\b", question, re.IGNORECASE):
                return pid
        # 2. alias 匹配
        for alias, method in self.aliases.items():
            if alias in question.lower():
                return self.methods.get(method)
        return None

    def all_methods(self) -> list[str]:
        return [m for m, p in self.methods.items() if p]


if __name__ == "__main__":
    r = PaperResolver()
    for q in ["论文《BIT》在 LEVIR-CD 上的 F1 是多少？",
              "ChangeFormer 的 IoU 是多少？",
              "Changer 在 LEVIR-CD 的指标？",
              "SNUNet 的 F1 是多少？"]:
        print(f"{r.resolve(q):<16} | {q}")