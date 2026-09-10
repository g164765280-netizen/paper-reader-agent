# -*- coding: utf-8 -*-
"""无泄漏检索修复 v4：标题→论文ID 解析 + 结构化指标表查表（只解析题目，不读 gold）。"""
from __future__ import annotations

import json
import re
from pathlib import Path

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")

METRIC_RE = re.compile(
    r"(F1|IoU|mIoU|OA|Precision|Recall)(?:-?score)?\s*(?:of)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE)
KNOWN_DATASETS = ["LEVIR-CD", "WHU-CD", "S2Looking", "SYSU-CD", "CDD", "DSIFN", "SECOND",
                  "HRSCD", "CLCD", "OSCD", "LEVIR", "WHU", "xView2"]


def extract_table(text: str) -> list[dict]:
    """从全文提取 (metric, value, dataset, sentence)，取数字周围 ±250 字符找数据集。"""
    out, seen = [], set()
    for m in METRIC_RE.finditer(text):
        field, num = m.group(1).lower(), m.group(2)
        if (field, num) in seen:
            continue
        seen.add((field, num))
        s = max(0, m.start() - 250)
        e = min(len(text), m.end() + 250)
        ctx = text[s:e].replace("\n", " ").strip()
        ds = [d for d in KNOWN_DATASETS if re.search(re.escape(d), ctx, re.IGNORECASE)]
        if ds:
            out.append({"metric": field, "value": num, "dataset": ds[0], "sentence": ctx})
    return out


def main():
    # 1) 建 title→paper_id 映射 + 每篇的指标表
    title2id, metric_table = {}, {}
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        pid = d["arxiv_id"]
        title2id[d.get("title", "").strip()] = pid
        metric_table[pid] = extract_table("\n".join(d.get("pages_text", [])))
    print(f"title→id: {len(title2id)} 篇；指标表总条目: {sum(len(v) for v in metric_table.values())}")

    # 2) 读评测集，无泄漏检索（gold 只用于打分，不用于检索）
    qs = [json.loads(l) for l in (CORPUS / "eval_set_v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(qs)
    doc_ok = ev_ok = val_ok = 0
    for q in qs:
        gold = q["gold"]
        pid = title2id.get(q["title"])          # 标题→论文ID（题目里的标题）
        if pid is None:
            continue
        doc_ok += 1
        hits = [e for e in metric_table.get(pid, [])
                if e["metric"] == gold["metric"] and e["dataset"] == gold["dataset"]]
        if hits:
            ev_ok += 1
        if any(e["value"] == gold["value"] for e in hits):
            val_ok += 1

    print(f"\n题目数: {n}")
    print(f"Document Recall（标题→论文ID）: {doc_ok}/{n} = {doc_ok/n*100:.1f}%")
    print(f"Evidence Recall（表里有该 metric+dataset）: {ev_ok}/{n} = {ev_ok/n*100:.1f}%")
    print(f"Value 命中（表里有该具体数值）: {val_ok}/{n} = {val_ok/n*100:.1f}%")


if __name__ == "__main__":
    main()
