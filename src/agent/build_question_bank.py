# -*- coding: utf-8 -*-
"""出题 v9：用 kimi-k2.6（独立第三方）识别"本文方法" + 生成 30 道数字题。

- 从指标库按论文分组 → 列方法名 → kimi 判断哪行是"本文方法"
- 题目只含键(论文/数据集/指标)，数值放 gold（不泄漏）
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import litellm

LIB = Path("/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl")
FULLTEXT = Path("/media/sdb1/gzj/data/rscd/corpus/fulltext")
OUT = Path("/media/sdb1/gzj/data/rscd/corpus/eval_set_v9.jsonl")

# API 凭据从环境变量读取，不硬编码
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE", "")
MODEL = "openai/qwen3.8-max"

TITLE = {}
for ft in FULLTEXT.glob("*.json"):
    d = json.loads(ft.read_text(encoding="utf-8"))
    TITLE[d["arxiv_id"]] = d.get("title", "").strip()


def identify_proposed(batch: list[dict]) -> dict[str, str]:
    """给 kimi 一批论文(标题+方法名列表)，返回 {paper_id: 本文方法名}。"""
    lines = []
    for b in batch:
        lines.append(f"[{b['paper_id']}] 标题: {b['title']}\n  方法名: {', '.join(b['methods'])}")
    prompt = ("下面是若干篇遥感变化检测论文。每篇列出标题和它实验结果表里出现的方法名。\n"
              "对每篇论文，指出哪【一个】方法名是「本文提出的方法」（不是基线、不是对比方法）。\n"
              "判断依据：方法名含 Ours/Proposed/our，或与论文标题里的方法缩写一致。\n\n"
              + "\n\n".join(lines) +
              "\n\n只输出 JSON：{\"paper_id\": \"本文方法名\", ...}，无法判断的 paper_id 不要写进 JSON。")
    r = litellm.completion(model=MODEL, api_key=API_KEY, api_base=API_BASE,
                           messages=[{"role": "user", "content": prompt}],
                           temperature=0.0, max_tokens=1000, extra_body={"enable_thinking": False})
    raw = r.choices[0].message.content or ""
    try:
        return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return {}


def main():
    # 1. 分组：paper → 方法名列表
    by_paper: dict[str, dict] = {}
    for l in LIB.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        p = by_paper.setdefault(r["paper_id"], {"methods": set(), "recs": []})
        p["methods"].add(r["method"])
        p["recs"].append(r)

    papers = [(pid, TITLE.get(pid, ""), sorted(v["methods"])) for pid, v in by_paper.items()]
    papers = [p for p in papers if p[1]]  # 有标题的
    print(f"共 {len(papers)} 篇论文，开始识别本文方法...", flush=True)

    # 2. 批量调 kimi（每批 5 篇）
    proposed = {}
    t0 = time.time()
    for i in range(0, len(papers), 3):
        batch = [{"paper_id": p[0], "title": p[1], "methods": p[2][:20]} for p in papers[i:i+5]]
        try:
            res = identify_proposed(batch)
            proposed.update(res)
        except Exception as e:
            print(f"  批 {i//5+1} 失败: {str(e)[:80]}", flush=True)
        print(f"  进度 {min(i+5, len(papers))}/{len(papers)}，已识别 {len(proposed)} 篇（{time.time()-t0:.0f}s）", flush=True)
        time.sleep(1)

    # 3. 生成题目：每篇 pick (dataset=LEVIR-CD 优先, metric=f1 优先) 的一条
    FAVOR_METRIC = ["f1", "iou", "oa", "miou", "precision", "recall"]
    qs = []
    for pid, v in by_paper.items():
        m = proposed.get(pid)
        if not m:
            continue
        cands = [r for r in v["recs"] if r["method"] == m and r["value"] not in ("-", "", "nan")]
        if not cands:
            continue
        # 优先 LEVIR-CD + f1
        pref = [r for r in cands if r["dataset"] == "LEVIR-CD" and r["metric"] == "f1"] or \
               [r for r in cands if r["metric"] == "f1"] or cands
        r = pref[0]
        qs.append({
            "paper_id": pid, "title": TITLE.get(pid, ""),
            "question": f"论文《{TITLE.get(pid, '')}》的 {m} 方法在 {r['dataset']} 上的 {r['metric']} 是多少？",
            "gold": {"method": m, "dataset": r["dataset"], "metric": r["metric"], "value": r["value"]},
        })

    # 限 30 道
    qs = qs[:30]
    OUT.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    print(f"\n生成 {len(qs)} 道题，存至 {OUT}", flush=True)


if __name__ == "__main__":
    main()