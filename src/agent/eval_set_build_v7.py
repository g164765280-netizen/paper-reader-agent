# -*- coding: utf-8 -*-
"""出题 v7：严格正则（不跨行、不编造），数值必须在原文同一句里。

- 只匹配「指标 与 数值 在同一行/同一句」的写法：F1=0.90 / F1 of 0.90 / F1: 0.90 / F1-Score of 67.61
- 不匹配表格跨行："IoU\n32"（指标和数值分两行）→ 拒绝
- 数值必须通过合法性过滤器
- 同一论文「同一数据集+指标」多个数值 → 跳过（不唯一）
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")

# [ \t]* 不跨行；可选 of/is/=/:
METRIC_RE = re.compile(
    r"(F1|IoU|mIoU|OA|Precision|Recall)(?:-?score)?[ \t]*(?:of|is|=|:)?[ \t]*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE)
KNOWN_DATASETS = ["LEVIR-CD", "WHU-CD", "S2Looking", "SYSU-CD", "CDD", "DSIFN", "SECOND",
                  "HRSCD", "CLCD", "OSCD"]


def valid_value(num_str: str) -> bool:
    if "." in num_str:
        return True
    n = float(num_str)
    return 10 <= n <= 100


def extract_entries(text: str) -> list[dict]:
    out, seen = [], set()
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    for m in METRIC_RE.finditer(text):
        field, num = m.group(1).lower(), m.group(2)
        if (field, num) in seen or not valid_value(num):
            continue
        # 阈值符号 τ 前缀 → 跳过
        if m.start() > 0 and text[m.start() - 1] in "ττ":
            continue
        # 差值：数字后面紧跟 higher/lower/gain/improve → 跳过
        after = text[m.end():m.end() + 40].lower()
        if re.search(r"higher|lower|gain|improve|boost|increase|decrease|outperform", after):
            continue
        seen.add((field, num))
        # 证据句：包含该匹配的"同一句"
        ev = ""
        for s in sentences:
            if m.group(0) in s:
                ev = s.strip()
                break
        if not ev:
            ev = text[m.start():m.end()].strip()
        # 数据集：±250 字符窗口内找
        w = text[max(0, m.start() - 250):min(len(text), m.end() + 250)]
        ds = [d for d in KNOWN_DATASETS if re.search(re.escape(d), w, re.IGNORECASE)]
        if ds:
            out.append({"metric": field, "value": num, "dataset": ds[0], "sentence": ev})
    return out


def main():
    qs = []
    skipped = 0
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        title = d.get("title", "").strip()
        entries = extract_entries("\n".join(d.get("pages_text", [])))
        groups = {}
        for e in entries:
            groups.setdefault((e["dataset"], e["metric"]), []).append(e)
        for (ds, mt), es in groups.items():
            vals = set(e["value"] for e in es)
            if len(vals) != 1:
                skipped += 1
                continue
            e = es[0]
            qs.append({
                "paper_id": d["arxiv_id"], "title": title,
                "question": f"论文《{title}》在 {ds} 上的 {mt} 是多少？",
                "gold": {"dataset": ds, "metric": mt, "value": e["value"], "evidence": e["sentence"]},
            })
    out = CORPUS / "eval_set_v7.jsonl"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    # 自检：每题的 value 必须都在 evidence 里
    bad = sum(1 for q in qs if q["gold"]["value"] not in q["gold"]["evidence"])
    print(f"共 {len(qs)} 道题（跳过歧义 {skipped} 组），其中 value 不在证据里的: {bad} 题")


if __name__ == "__main__":
    main()