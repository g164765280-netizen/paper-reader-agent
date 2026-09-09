# -*- coding: utf-8 -*-
"""从留出论文生成评测集（问题 + 金证据 + 金答案字段）。

产出自带可验证性：
  - number 类问题带 field/number/unit，用于"字段-数字-单位"校验；
  - name 类问题带 gold_names，用于覆盖度校验；
  - unanswerable 类问题带 should_refuse=True，用于拒答校验；
  - 每题 gold_evidence 即 Oracle 分支注入的正确证据段落。

用法：python3 eval_set_build.py --corpus /path/to/corpus [--questions-per-paper 5]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KNOWN_DATASETS = [
    "LEVIR-CD", "WHU-CD", "S2Looking", "SYSU-CD", "CDD", "DSIFN", "SECOND",
    "HRSCD", "CLCD", "OSCD", "xView2", "3DCD", "LEVIR", "WHU", "S2Looking",
    "BANDON", "EGY-BCD", "Hi-UCD",
]

METRIC_RE = re.compile(
    r"(F1|IoU|mIoU|OA|Precision|Recall)(?:-?score)?\s*(?:of)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE)


def find_numbers(abstract: str) -> list[dict]:
    """只提取干净且高置信的"指标-数字"对，避免伪字段（如 Sek）与噪声数字。"""
    out = []
    seen = set()
    for m in METRIC_RE.finditer(abstract):
        field = m.group(1).lower()
        num = m.group(2)
        key = (field, num)
        if key in seen:
            continue
        seen.add(key)
        out.append({"field": field, "number": num, "unit": "score"})
    return out


def find_datasets(text: str) -> list[str]:
    found = []
    for d in KNOWN_DATASETS:
        if re.search(re.escape(d), text, re.IGNORECASE) and d not in found:
            found.append(d)
    return found


def method_from_title(title: str) -> str:
    # 标题里通常含方法名：去掉常见词
    title = title.replace("\n", " ")
    m = re.search(r"([A-Za-z][A-Za-z0-9\-]+Net|[A-Za-z][A-Za-z0-9\-]+Former|[A-Za-z][A-Za-z0-9]+Net)", title)
    if m:
        return m.group(1)
    return ""


def build(corpus: Path, qpp: int = 5) -> list[dict]:
    split = json.loads((corpus / "split.json").read_text(encoding="utf-8"))
    held = split["heldout_meta"]
    train = split["train_meta"]
    # 留出论文：测拒答/不幻觉；训练论文：测检索召回 + 忠实
    papers = [{"meta": p, "split": "heldout"} for p in held] + \
             [{"meta": p, "split": "train"} for p in train]
    questions = []
    for item in papers:
        p = item["meta"]
        sp = item["split"]
        title = p.get("title", "").strip()
        abstract = p.get("abstract", "").strip()
        aid = p.get("arxiv_id", "")
        if not title or not abstract:
            continue
        ev = abstract[:1200]
        qs_for_paper = []

        # 1) 方法名
        mname = method_from_title(title)
        if mname:
            qs_for_paper.append({
                "type": "name", "paper_id": aid, "title": title,
                "question": f"论文《{title}》提出的方法/网络叫什么名字？",
                "gold_names": [mname], "gold_evidence": ev,
            })

        # 2) 数字题
        nums = find_numbers(abstract)[:3]
        for n in nums:
            qs_for_paper.append({
                "type": "number", "paper_id": aid, "title": title,
                "question": f"论文《{title}》报告的 {n['field']} 是多少？",
                "gold": n, "gold_evidence": ev,
            })

        # 3) 数据集题
        dsets = find_datasets(abstract)
        if dsets:
            qs_for_paper.append({
                "type": "name", "paper_id": aid, "title": title,
                "question": f"论文《{title}》在哪些数据集上做了实验？",
                "gold_names": dsets, "gold_evidence": ev,
            })

        # 4) 拒答题（故意问摘要里没有的内容）
        qs_for_paper.append({
            "type": "unanswerable", "paper_id": aid, "title": title,
            "question": f"论文《{title}》中关于 ZZZQWE123 这一无关主题的结论是什么？",
            "should_refuse": True, "gold_evidence": ev,
        })

        # 控制每题数量 + 打上 split 标签
        for q in qs_for_paper[:qpp]:
            q["split"] = sp
        questions.extend(qs_for_paper[:qpp])

    return questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--questions-per-paper", type=int, default=5)
    args = ap.parse_args()
    corpus = Path(args.corpus)
    qs = build(corpus, args.questions_per_paper)
    out = corpus / "eval_set.jsonl"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    types = {}
    for q in qs:
        types[q["type"]] = types.get(q["type"], 0) + 1
    print(f"total questions: {len(qs)}")
    print(f"by type: {types}")


if __name__ == "__main__":
    main()
