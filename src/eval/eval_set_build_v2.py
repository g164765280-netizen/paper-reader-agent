# -*- coding: utf-8 -*-
"""基于全文文本出题（评测集 v2）：方法名/数据集/全文指标数字/拒答。

从 fulltext/*.json 读留出论文的正文，用 METRIC_RE 在全文里找"指标-数字"对，
比摘要级能找到多得多的可验证数字题。
"""
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


def method_from_title(title: str) -> str:
    m = re.search(r"([A-Za-z][A-Za-z0-9\-]+Net|[A-Za-z][A-Za-z0-9\-]+Former|[A-Za-z][A-Za-z0-9]+Net)", title)
    return m.group(1) if m else ""


def find_numbers(text: str) -> list[dict]:
    """提取指标-数字对，并返回"包含该指标的那句话"作为金证据。"""
    out, seen = [], set()
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    for m in METRIC_RE.finditer(text):
        field, num = m.group(1).lower(), m.group(2)
        if (field, num) in seen:
            continue
        seen.add((field, num))
        # 找包含该匹配的句子
        start, end = m.start(), m.end()
        ev = text[start:end]
        for s in sentences:
            if m.group(0) in s or (field in s and num in s):
                ev = s.strip()
                break
        out.append({"field": field, "number": num, "unit": "score", "evidence": ev})
    return out


def find_datasets(text: str) -> list[str]:
    return [d for d in KNOWN_DATASETS if re.search(re.escape(d), text, re.IGNORECASE)]


def main():
    split = json.loads((CORPUS / "split.json").read_text(encoding="utf-8"))
    held_ids = set(split["heldout_ids"])
    qs = []
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        sp = "heldout" if d["arxiv_id"] in held_ids else "train"
        title = d.get("title", "").strip()
        full_text = "\n".join(d.get("pages_text", []))
        if not title or not full_text:
            continue
        ev = full_text[:1600]

        # 方法名
        mname = method_from_title(title)
        if mname:
            qs.append({"type": "name", "paper_id": d["arxiv_id"], "title": title, "split": sp,
                       "question": f"论文《{title}》提出的方法叫什么？",
                       "gold_names": [mname], "gold_evidence": ev})

        # 数字题（全文，最多 5 个，金证据=包含该指标的那句话）
        for n in find_numbers(full_text)[:5]:
            qs.append({"type": "number", "paper_id": d["arxiv_id"], "title": title, "split": sp,
                       "question": f"论文《{title}》报告的 {n['field']} 是多少？",
                       "gold": {"field": n["field"], "number": n["number"], "unit": "score"},
                       "gold_evidence": n["evidence"]})

        # 数据集
        dsets = find_datasets(full_text)
        if dsets:
            qs.append({"type": "name", "paper_id": d["arxiv_id"], "title": title, "split": sp,
                       "question": f"论文《{title}》在哪些数据集上做了实验？",
                       "gold_names": dsets, "gold_evidence": ev})

        # 拒答
        qs.append({"type": "unanswerable", "paper_id": d["arxiv_id"], "title": title, "split": sp,
                   "question": f"论文《{title}》中关于 ZZZQWE123 这一无关主题的结论是什么？",
                   "should_refuse": True, "gold_evidence": ev})

    # 专家对比题 + 推理题（固定领域知识，gold=应出现的关键词）
    COMPARE = [
        ("BIT 和 ChangeFormer 在 LEVIR-CD 上哪个 F1 通常更高？", "ChangeFormer"),
        ("TinyCD 和 ChangeFormer 哪个参数量更小、推理更快？", "TinyCD"),
        ("BIT 和 SNUNet 哪个是 Transformer 路线？", "BIT"),
        ("ChangeFormer 和 Changer 哪个是层次化 Transformer？", "ChangeFormer"),
        ("LEVIR-CD 和 WHU-CD 哪个分辨率更高？", "WHU-CD"),
        ("S2Looking 和 LEVIR-CD 哪个数据规模更大？", "S2Looking"),
        ("SYSU-CD 和 CDD 哪个样本更多？", "SYSU-CD"),
        ("SECOND 和 HRSCD 哪个是语义变化检测数据集？", "SECOND"),
        ("IoU 和 OA 哪个更适合类不平衡的变化检测？", "IoU"),
        ("F1 和 OA 哪个更能反映变化检测精度？", "F1"),
    ]
    REASON = [
        ("变化检测中 Precision 高 Recall 低说明什么？", "漏检"),
        ("为什么变化检测不以 OA 为主指标？", "不平衡"),
        ("配准误差会导致什么问题？", "伪变化"),
        ("季节变化导致被误判为变化是什么现象？", "伪变化"),
        ("Precision 和 Recall 的调和平均是什么指标？", "F1"),
        ("为什么孪生网络适合变化检测？", "权重共享"),
        ("变化像素占比很小会导致 OA 怎样？", "虚高"),
    ]
    for q, gold in COMPARE:
        qs.append({"type": "compare", "question": q, "split": "expert", "gold_name": gold})
    for q, gold in REASON:
        qs.append({"type": "reasoning", "question": q, "split": "expert", "gold_name": gold})

    out = CORPUS / "eval_set_v2.jsonl"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    types = {}
    for q in qs:
        types[q["type"]] = types.get(q["type"], 0) + 1
    print(f"total questions: {len(qs)}")
    print(f"by type: {types}")


if __name__ == "__main__":
    main()
