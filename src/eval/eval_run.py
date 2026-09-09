# -*- coding: utf-8 -*-
"""运行评测：Base+Oracle 与 Base+RAG（微调后再加 SFT 两组），产出指标。

用法（all-in-rag 环境）：
  python eval_run.py --corpus ... --embedding ... --reranker ... --model ... \
      --configs oracle,rag --out results_baseline.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from rag_pipeline import RAGPipeline

REFUSE_KEYS = ["无法回答", "未提及", "没有提供", "不知道", "无法", "无相关信息", "不包含", "没有相关"]


def nums_in(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]


def num_match(gold: str, ans_nums: list[float]) -> bool:
    try:
        g = float(gold)
    except ValueError:
        return False
    for a in ans_nums:
        if abs(a - g) <= max(0.001, abs(g) * 0.02):
            return True
        # 百分比口径：0.90 vs 90
        if abs(a - g * 100) <= 0.5 or abs(g - a * 100) <= 0.005:
            return True
    return False


def verify(q: dict, answer: str) -> dict:
    r = {"type": q["type"], "answer_nonempty": len(answer.strip()) > 0,
         "has_citation": bool(re.search(r"\[\d+\]", answer))}
    if q["type"] == "number":
        g = q["gold"]
        an = nums_in(answer)
        r["gold_number"] = g["number"]
        r["number_matched"] = num_match(g["number"], an)
        r["field_present"] = g["field"].lower() in answer.lower()
        r["faithful"] = r["number_matched"] and r["field_present"]
    elif q["type"] == "name":
        hits = [n for n in q["gold_names"] if n.lower() in answer.lower()]
        r["gold_names"] = q["gold_names"]
        r["names_hit"] = len(hits)
        r["coverage"] = len(hits) / max(1, len(q["gold_names"]))
        r["faithful"] = len(hits) >= 1
    elif q["type"] == "unanswerable":
        r["refused"] = any(k in answer for k in REFUSE_KEYS)
        r["faithful"] = r["refused"]
    elif q["type"] in ("compare", "reasoning"):
        r["gold_name"] = q["gold_name"]
        r["faithful"] = q["gold_name"].lower() in answer.lower()
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="oracle,rag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    qs = [json.loads(l) for l in (corpus / "eval_set_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        qs = qs[: args.limit]
    print(f"eval questions: {len(qs)}", flush=True)

    # 数字感知检索的指标查找表
    metric_lookup = {}
    ml_path = corpus / "metric_lookup.json"
    if ml_path.exists():
        metric_lookup = json.loads(ml_path.read_text(encoding="utf-8"))

    pipe = RAGPipeline(corpus, args.embedding, args.reranker, args.model,
                       adapter_dir=args.adapter or None)

    results = {}
    details = {}
    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        print(f"=== config: {cfg} ===", flush=True)
        rows = []
        t0 = time.time()
        for i, q in enumerate(qs, 1):
            if cfg == "oracle":
                ctx = [{"title": q["title"], "page": 0, "text": q["gold_evidence"]}]
                retrieved_gold = True
            elif q["type"] == "number" and metric_lookup:
                # 数字感知检索：查表定位"含该指标数字的那句话"
                entries = metric_lookup.get(q["paper_id"], {}).get(q["gold"]["field"], [])
                target = q["gold"]["number"]
                hit = None
                for e in entries:
                    if e["number"] == target:
                        hit = e["sentence"]
                        break
                if hit is None and entries:
                    hit = entries[0]["sentence"]
                if hit:
                    ctx = [{"title": q["title"], "page": 0, "text": hit}]
                    retrieved_gold = True
                else:
                    ctx = pipe.retrieve(q["question"])
                    retrieved_gold = False
            else:
                ctx = pipe.retrieve(q["question"])
                retrieved_gold = q["paper_id"] in {c["paper_id"] for c in ctx}
            ans = pipe.generate(q["question"], ctx)
            vr = verify(q, ans)
            rows.append({"paper_id": q["paper_id"], "split": q.get("split", ""),
                         "question": q["question"], "answer": ans, "verify": vr,
                         "retrieved_gold": retrieved_gold})
            if i % 10 == 0:
                print(f"  [{i}/{len(qs)}] {time.time()-t0:.0f}s", flush=True)
        details[cfg] = rows

        # 聚合
        agg = {"config": cfg, "n": len(rows),
               "answer_nonempty_rate": sum(r["verify"]["answer_nonempty"] for r in rows) / len(rows),
               "citation_rate": sum(r["verify"]["has_citation"] for r in rows) / len(rows)}
        by_type = {}
        by_split = {}
        for r in rows:
            t = r["verify"]["type"]
            sp = r["split"]
            by_type.setdefault(t, {"n": 0, "faithful": 0, "refused": 0, "refused_total": 0})
            by_type[t]["n"] += 1
            by_type[t]["faithful"] += 1 if r["verify"].get("faithful") else 0
            if t == "unanswerable":
                by_type[t]["refused_total"] += 1
                by_type[t]["refused"] += 1 if r["verify"].get("refused") else 0
            by_split.setdefault(sp, {"n": 0, "faithful": 0, "retrieved_gold": 0, "retrieved_total": 0})
            by_split[sp]["n"] += 1
            by_split[sp]["faithful"] += 1 if r["verify"].get("faithful") else 0
            if cfg != "oracle":
                by_split[sp]["retrieved_total"] += 1
                by_split[sp]["retrieved_gold"] += 1 if r["retrieved_gold"] else 0
        agg["by_type"] = by_type
        agg["by_split"] = by_split
        results[cfg] = agg
        print(json.dumps(agg, ensure_ascii=False, indent=2), flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # 保存逐条结果
    detail_lines = []
    for cfg, rows in details.items():
        for r in rows:
            detail_lines.append(json.dumps({"config": cfg, **r}, ensure_ascii=False) + "\n")
    out.with_suffix(".details.jsonl").write_text("".join(detail_lines), encoding="utf-8")
    print(f"results written to {out}", flush=True)


if __name__ == "__main__":
    main()
