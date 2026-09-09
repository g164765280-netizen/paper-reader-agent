# -*- coding: utf-8 -*-
"""V2：更难的 SFT/DPO 数据——长证据 + 多字段干扰，训练模型在长文中定位特定字段-数字。"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)
OUT = Path(__file__).resolve().parents[2] / "data" / "rscd"

METHODS = ["ChangeFormer", "BIT", "ChangerEx", "SNUNet", "TinyCD", "HANet", "DMINet", "ChangeStar"]
DATASETS = ["LEVIR-CD", "WHU-CD", "S2Looking", "SYSU-CD", "CDD", "SECOND", "HRSCD", "CLCD"]
FIELDS = ["F1", "IoU", "mIoU", "OA", "Precision", "Recall"]


def _num():
    return f"{random.uniform(0.70, 0.95):.3f}"


def _long_evidence() -> tuple[str, dict]:
    """生成一段长证据（含多个指标/数据集/方法做干扰），返回证据文本与字段->数字表。"""
    m = random.choice(METHODS)
    d = random.choice(DATASETS)
    d2 = random.choice([x for x in DATASETS if x != d])
    vals = {f: _num() for f in FIELDS}
    ev = (
        f"证据[1]：本文提出 {m} 用于遥感变化检测。"
        f"作者在 {d} 和 {d2} 两个数据集上评估该方法，"
        f"其中在 {d} 上取得 F1={vals['F1']}、IoU={vals['IoU']}、"
        f"Precision={vals['Precision']}、Recall={vals['Recall']}、OA={vals['OA']}。"
        f"在 {d2} 上取得 F1={_num()}、IoU={_num()}。"
        f"与基线方法相比，{m} 在 {d} 上 F1 提升了 {random.uniform(0.01, 0.05):.2f}。"
    )
    return ev, vals


def sft_number_v2(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        ev, vals = _long_evidence()
        field = random.choice(FIELDS)
        q = f"该方法在 LEVIR-CD 上的 {field} 是多少？" if "LEVIR-CD" in ev else \
            f"该方法在其主数据集上的 {field} 是多少？"
        # 用证据里的数据集名重写问题，避免硬编码
        for d in DATASETS:
            if d in ev:
                q = f"该方法在 {d} 上的 {field} 是多少？"
                break
        a = f"{field} 为 {vals[field]}[1]。"
        rows.append({"conversations": [
            {"from": "human", "value": f"{ev}\n\n问题：{q}"},
            {"from": "gpt", "value": a}]})
    return rows


def sft_name_v2(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        m = random.choice(METHODS)
        d, d2 = random.sample(DATASETS, 2)
        ev = (f"证据[1]：本文提出 {m}，并在 {d} 与 {d2} 上验证，"
              f"实验表明其优于现有基线方法。")
        if random.random() < 0.5:
            q = "这篇论文提出的方法叫什么？"
            a = f"方法为 {m}[1]。"
        else:
            q = "这篇论文在哪些数据集上做了实验？"
            a = f"在 {d} 和 {d2} 上验证[1]。"
        rows.append({"conversations": [
            {"from": "human", "value": f"{ev}\n\n问题：{q}"},
            {"from": "gpt", "value": a}]})
    return rows


def sft_refusal_v2(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        ev, _ = _long_evidence()
        q = "这篇论文的训练学习率是多少？"
        a = "根据已有论文无法回答该问题，证据中未提及训练学习率。"
        rows.append({"conversations": [
            {"from": "human", "value": f"{ev}\n\n问题：{q}"},
            {"from": "gpt", "value": a}]})
    return rows


def dpo_v2(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        ev, vals = _long_evidence()
        field = random.choice(FIELDS)
        q = f"该方法在其主数据集上的 {field} 是多少？"
        for d in DATASETS:
            if d in ev:
                q = f"该方法在 {d} 上的 {field} 是多少？"
                break
        chosen = f"{field} 为 {vals[field]}[1]。"
        rkind = random.choice(["hallucinate", "no_cite", "overclaim"])
        if rkind == "hallucinate":
            rejected = f"{field} 为 {float(vals[field])+0.05:.3f}。"
        elif rkind == "no_cite":
            rejected = f"{field} 大概是 {vals[field]} 左右。"
        else:
            rejected = f"{field} 达到全球第一的 {vals[field]}，完美碾压所有方法。"
        rows.append({"conversations": [{"from": "human", "value": f"{ev}\n\n问题：{q}"}],
                     "chosen": chosen, "rejected": rejected})
    return rows


def main() -> None:
    sft = sft_number_v2(600) + sft_name_v2(250) + sft_refusal_v2(150)
    dpo = dpo_v2(500)
    (OUT / "sft").mkdir(parents=True, exist_ok=True)
    (OUT / "reward").mkdir(parents=True, exist_ok=True)
    (OUT / "sft" / "rscd_sft_v2.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sft), encoding="utf-8")
    (OUT / "reward" / "rscd_dpo_v2.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dpo), encoding="utf-8")
    print(f"sft_v2: {len(sft)}")
    print(f"dpo_v2: {len(dpo)}")


if __name__ == "__main__":
    main()
