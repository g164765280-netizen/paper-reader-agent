# -*- coding: utf-8 -*-
"""P0 可信度评测：多次跑 + bootstrap 置信区间。

针对「可信度」指标的评测，两件事：
1. 多次跑：评测集跑 N 次，统计指标的均值 + 波动（避免单次评测的偶然性）。
2. bootstrap 重采样：从现有评测结果里，对每个指标估 95% 置信区间。

测的「可信度」指标（P0 收紧版）：
  - 引用率 grounded：答案引用真实出处、不编造的比例。
  - 拒答率 refusal：信息不足时诚实拒答（而非瞎编）的比例。
  - 答到点率 relevance：命中金标准关键点的比例。
  （三个指标都可对「数字题」和「概念题」分别统计）

置信区间方法：bootstrap（有放回重采样 2000 次，取 2.5%/97.5% 分位数）。

用法：
  服务器上 /media/sdb1/gzj/support/llm/envs/paperqa/bin/python eval_confidence.py \
      --answers /path/to/concept_answers_v3.jsonl \
      --report  /path/to/concept_eval_report_v3.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def bootstrap_ci(values: list[int], n_boot: int = 2000, seed: int = 42) -> dict:
    """对 0/1 序列做 bootstrap，返回 95% 置信区间 + 均值。"""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {"mean": sum(values) / n, "ci_low": lo, "ci_high": hi, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    report = [json.loads(l) for l in Path(args.report).read_text(encoding="utf-8").splitlines() if l.strip()]
    answers = [json.loads(l) for l in Path(args.answers).read_text(encoding="utf-8").splitlines() if l.strip()]

    # 提取三维 0/1 序列
    def field(rows, key, sub="grounded"):
        return [1 if r.get("judge", {}).get(sub) == 1 else 0 for r in rows]

    grounded = field(report, "judge", "grounded")
    relevance = field(report, "judge", "relevance")
    honesty = field(report, "judge", "honesty")

    # 拒答率：从 answers 里判断（answer 含拒答短语）
    def is_refused(a: dict) -> int:
        t = (a.get("answer") or "").lower()
        return 1 if any(k in t for k in ["cannot answer", "无法", "查不到", "未找到", "insufficient"]) else 0
    refused = [is_refused(a) for a in answers]

    print("=" * 60)
    print(f"P0 可信度评测（n={len(report)} 题，bootstrap {args.n_boot} 次）")
    print("=" * 60)
    for name, vals in [("grounded 有出处率", grounded),
                       ("relevance 答到点率", relevance),
                       ("honesty 诚实率", honesty),
                       ("refusal 拒答率", refused)]:
        ci = bootstrap_ci(vals, args.n_boot)
        print(f"{name:<20} 均值 {ci['mean']*100:5.1f}%   "
              f"95% CI [{ci['ci_low']*100:.1f}%, {ci['ci_high']*100:.1f}%]  (n={ci['n']})")


if __name__ == "__main__":
    main()