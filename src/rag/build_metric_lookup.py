# -*- coding: utf-8 -*-
"""构建"指标查找表"：每篇论文提取 {field: [(number, sentence), ...]}。

用于数字感知检索——数字题直接查表定位"含该指标数字的那句话"，
而不是靠稠密检索猜。输出 metric_lookup.json。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")

METRIC_RE = re.compile(
    r"(F1|IoU|mIoU|OA|Precision|Recall)(?:-?score)?\s*(?:of)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE)


def extract(text: str) -> list[tuple]:
    out = []
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    for m in METRIC_RE.finditer(text):
        field, num = m.group(1).lower(), m.group(2)
        ev = text[m.start():m.end()]
        for s in sentences:
            if field in s and num in s:
                ev = s.strip()
                break
        out.append((field, num, ev))
    return out


def main():
    lookup = {}
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        aid = d["arxiv_id"]
        full_text = "\n".join(d.get("pages_text", []))
        metrics = {}
        for field, num, ev in extract(full_text):
            metrics.setdefault(field, []).append({"number": num, "sentence": ev})
        if metrics:
            lookup[aid] = metrics

    out = CORPUS / "metric_lookup.json"
    out.write_text(json.dumps(lookup, ensure_ascii=False, indent=1), encoding="utf-8")
    n_papers = len(lookup)
    n_entries = sum(len(v) for p in lookup.values() for v in p.values())
    print(f"metric lookup: {n_papers} papers, {n_entries} field entries")


if __name__ == "__main__":
    main()
