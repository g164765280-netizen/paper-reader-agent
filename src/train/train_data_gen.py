# -*- coding: utf-8 -*-
"""针对基线短板生成 SFT / DPO 训练数据（格式与 MedicalGPT 训练脚本一致）。

基线短板：数字忠实 5.6%、名称 65%、引用 24%。故数据重点：
  SFT:  ①数字/字段/单位忠实抽取(50%) ②名称/数据集抽取(25%) ③拒答(10%) ④比较(15%)
  DPO:  chosen=忠实+引用+拒答；rejected=数字幻觉/无引用/过度声称/不拒答
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)
OUT = Path(__file__).resolve().parents[2] / "data" / "rscd"

METHODS = ["ChangeFormer", "BIT", "ChangerEx", "SNUNet", "TinyCD", "HANet", "DMINet", "ChangeStar"]
DATASETS = ["LEVIR-CD", "WHU-CD", "S2Looking", "SYSU-CD", "CDD", "SECOND", "HRSCD", "CLCD"]
FIELDS = ["F1", "IoU", "mIoU", "OA", "Precision", "Recall"]


def _num():
    return f"{random.uniform(0.70, 0.95):.3f}"


def _evidence_one() -> str:
    m, d = random.choice(METHODS), random.choice(DATASETS)
    return (f"{m} 在 {d} 上取得 F1={_num()}、IoU={_num()}、"
            f"Precision={_num()}、Recall={_num()}。")


def _evidence_pair() -> str:
    return f"证据[1]：{_evidence_one()}\n证据[2]：{_evidence_one()}"


def sft_number(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        m, d = random.choice(METHODS), random.choice(DATASETS)
        field = random.choice(FIELDS)
        val = _num()
        ev = (f"证据[1]：{m} 在 {d} 数据集上取得 {field}={val}。"
              + ("" if random.random() < 0.5 else f" 其他指标为 F1={_num()}、IoU={_num()}。"))
        q = f"{m} 在 {d} 上的 {field} 是多少？"
        a = f"{field} 为 {val}[1]。"
        rows.append({"conversations": [
            {"from": "human", "value": f"{ev}\n\n问题：{q}"},
            {"from": "gpt", "value": a}]})
    return rows


def sft_name(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        m, d = random.choice(METHODS), random.choice(DATASETS)
        ev = f"证据[1]：本文提出 {m}，并在 {d}、{random.choice(DATASETS)} 上验证。"
        if random.random() < 0.5:
            q = "这篇论文提出的方法叫什么？"
            a = f"方法为 {m}[1]。"
        else:
            q = "这篇论文在哪些数据集上做了实验？"
            a = f"在 {d} 等数据集上验证[1]。"
        rows.append({"conversations": [
            {"from": "human", "value": f"{ev}\n\n问题：{q}"},
            {"from": "gpt", "value": a}]})
    return rows


def sft_refusal(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        ev = f"证据[1]：{_evidence_one()}"
        q = random.choice(["这篇论文的训练学习率是多少？", "该方法在海洋遥感上的表现如何？",
                           "论文中关于无关主题的结论是什么？"])
        a = "根据已有论文无法回答该问题，证据中未提及相关内容。"
        rows.append({"conversations": [
            {"from": "human", "value": f"{ev}\n\n问题：{q}"},
            {"from": "gpt", "value": a}]})
    return rows


def sft_compare(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        m1, m2 = random.sample(METHODS, 2)
        d = random.choice(DATASETS)
        f1a, f1b = _num(), _num()
        ev = (f"证据[1]：{m1} 在 {d} 上 F1={f1a}。\n"
              f"证据[2]：{m2} 在 {d} 上 F1={f1b}。")
        q = f"{m1} 和 {m2} 在 {d} 上哪个 F1 更高？"
        if f1a > f1b:
            a = f"{m1} 的 F1（{f1a}）高于 {m2}（{f1b}）[1][2]。"
        else:
            a = f"{m2} 的 F1（{f1b}）高于 {m1}（{f1a}）[1][2]。"
        rows.append({"conversations": [
            {"from": "human", "value": f"{ev}\n\n问题：{q}"},
            {"from": "gpt", "value": a}]})
    return rows


def dpo_pairs(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        m, d = random.choice(METHODS), random.choice(DATASETS)
        field = random.choice(FIELDS)
        val = _num()
        ev = f"证据[1]：{m} 在 {d} 上 {field}={val}。"
        q = f"{m} 在 {d} 上的 {field} 是多少？"
        chosen = f"{field} 为 {val}[1]。"
        # rejected 变体
        rkind = random.choice(["hallucinate", "no_cite", "overclaim"])
        if rkind == "hallucinate":
            rejected = f"{field} 为 {float(val)+0.06:.3f}，达到了最高水平。"
        elif rkind == "no_cite":
            rejected = f"{field} 大概是 {val} 左右，差不多是这个数。"
        else:
            rejected = f"{field} 达到了全球第一的 {val}，完全碾压所有方法，没有任何缺点。"
        rows.append({"conversations": [{"from": "human", "value": f"{ev}\n\n问题：{q}"}],
                     "chosen": chosen, "rejected": rejected})
    return rows


def main() -> None:
    sft = (sft_number(500) + sft_name(250) + sft_refusal(100) + sft_compare(150))
    dpo = dpo_pairs(500)
    (OUT / "sft").mkdir(parents=True, exist_ok=True)
    (OUT / "reward").mkdir(parents=True, exist_ok=True)
    (OUT / "sft" / "rscd_sft.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sft), encoding="utf-8")
    (OUT / "reward" / "rscd_dpo.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dpo), encoding="utf-8")
    print(f"sft: {len(sft)}")
    print(f"dpo: {len(dpo)}")


if __name__ == "__main__":
    main()
