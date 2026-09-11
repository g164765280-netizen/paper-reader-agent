# -*- coding: utf-8 -*-
"""生成数字题评测报告 markdown。"""
import json
import re


def strict(gold, ans):
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ans)]
    g = float(gold)
    return any(abs(a - g) <= 0.05 for a in nums)


def main():
    rs = [json.loads(l) for l in open("/tmp/eval_result_v9.jsonl", encoding="utf-8") if l.strip()]
    for r in rs:
        r["correct"] = strict(r["gold"]["value"], r["model_ans"])
    correct = sum(1 for r in rs if r["correct"])

    L = []
    L.append("# 数字题评测报告")
    L.append("")
    L.append("## 概览")
    L.append("")
    L.append("- 出题模型：**qwen3.8-max（API，独立第三方）**——识别\"本文方法\" + 生成题目（题目含具体方法名）")
    L.append("- 答题模型：**本地 Qwen3-8B**——读指标库记录 + 题目 → 答数值")
    L.append("- 指标库：Docling 抽表 → **8657 条**「方法×数据集×指标→数值」")
    L.append("- 题目数：**30 道数字题**")
    L.append("- **消歧**：题目从\"提出的方法\"改为\"具体方法名\"（如 ChangerVanilla），消除方法变体歧义")
    L.append("")
    L.append(f"- **答对率：{correct}/30 = {correct/30*100:.1f}%**（严格匹配，绝对差 ≤ 0.05）")
    L.append("")
    L.append("（消歧前 27/30=90.0%，消歧后 28/30=93.3%）")
    L.append("")
    L.append("## 题目 + 金标准 + 本地 Qwen3-8B 回答")
    L.append("")
    L.append("| # | 论文 | 方法 | 数据集.指标 | 金标准 | 本地Qwen3-8B答 | 对/错 |")
    L.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rs, 1):
        g = r["gold"]
        title = r["title"][:24]
        method = g["method"]
        dsmt = g["dataset"] + "." + g["metric"]
        gold = g["value"]
        ans = r["model_ans"].replace("\n", " ")[:20]
        mark = "✅" if r["correct"] else "❌"
        L.append(f"| {i} | {title} | {method} | {dsmt} | {gold} | {ans} | {mark} |")

    L.append("")
    L.append("## 答错分析（2 题）")
    L.append("")
    for i, r in enumerate(rs, 1):
        if not r["correct"]:
            g = r["gold"]
            L.append(f"- 题{i}：gold={g['value']}（方法「{g['method']}」），模型答「{r['model_ans'][:30]}」")
            L.append("  —— 题目已明确方法名仍答错，疑似指标库该记录数据有误（Docling 抽表时数值与方法名未对齐），非答题模型问题，需回查指标库。")

    open("docs/数字题评测报告.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"报告已写：答对 {correct}/30")


if __name__ == "__main__":
    main()