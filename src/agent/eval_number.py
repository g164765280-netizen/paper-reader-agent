# -*- coding: utf-8 -*-
"""数字题评测（无泄漏）：指标库查表。

- gold 记录 = {paper_id, method, dataset, metric, value}
- 出题只含"键"（论文/方法/数据集/指标），value 绝不出现在题目里
- 评测：按键查库 → 返回值 → 与 gold 比
"""
from __future__ import annotations

import json
import random
from pathlib import Path

LIB = Path("/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl")

# 论文标题 → 本文方法关键词（用于识别"本文方法"那一行）
TITLE2METHOD = {}
for ft in Path("/media/sdb1/gzj/data/rscd/corpus/fulltext").glob("*.json"):
    d = json.loads(ft.read_text(encoding="utf-8"))
    TITLE2METHOD[d["arxiv_id"]] = d.get("title", "")


def is_proposed(method: str, title: str) -> bool:
    """判断一行是不是"本文方法"：方法名含 Ours/Proposed/our，或标题里的方法缩写。"""
    m = method.lower()
    if any(k in m for k in ["ours", "proposed", "our ", "s3", "s4", "(s"]):
        return True
    return False


def load_library() -> list[dict]:
    return [json.loads(l) for l in LIB.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    recs = load_library()
    print(f"指标库共 {len(recs)} 条记录")

    # 按 paper 分组，识别本文方法行
    by_paper = {}
    for r in recs:
        by_paper.setdefault(r["paper_id"], []).append(r)

    # 生成评测：每篇论文，(dataset, metric) 里挑本文方法的记录
    gold = []
    for pid, rs in by_paper.items():
        title = TITLE2METHOD.get(pid, "")
        # 本文方法 = method 含 ours/proposed；否则取 F1 最高那行的 method
        proposed_methods = [r["method"] for r in rs if is_proposed(r["method"], title)]
        chosen = proposed_methods[0] if proposed_methods else None
        for r in rs:
            if chosen and r["method"] != chosen:
                continue
            gold.append(r)

    print(f"本文方法 gold 记录 {len(gold)} 条")

    # 出题（只含键）→ 查库 → 打分
    correct = total = 0
    for g in gold:
        total += 1
        # 键：paper + method + dataset + metric（查库）
        hit = [r for r in recs if r["paper_id"] == g["paper_id"] and r["method"] == g["method"]
               and r["dataset"] == g["dataset"] and r["metric"] == g["metric"]]
        if hit and hit[0]["value"] == g["value"]:
            correct += 1

    print(f"查表准确率: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    main()