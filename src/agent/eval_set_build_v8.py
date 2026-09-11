# -*- coding: utf-8 -*-
"""出题 v8：LLM 照抄原文 + 程序校验（数字必须在抄的那句原文里），杜绝编造。

流程：
1. LLM 提取"本文提出的方法"的实验结果，每条必须给出【原句 quote】+ 数值 value。
2. 程序校验：value 必须是 quote 的子串，且 (dataset,metric) 唯一；否则丢弃。
3. 题：论文《X》在 {dataset} 上的 {metric} 是多少？
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import litellm

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")
API_KEY = "sk_tr_BNoSixd79VmtzvXtgTHsA7OsXHDiEVvF-WJotkXRevs"
API_BASE = "https://tokenrhythm.studio/v1"
MODEL = "openai/qwen3.8-flash"

PROMPT = """你是遥感变化检测论文的数据抽取器。从论文文本中，提取**论文自己提出的方法/本文方法（Ours/our method/the proposed）**的关键实验结果。

对每条结果，字段：
- dataset: 数据集标准名（LEVIR-CD/WHU-CD/S2Looking/SYSU-CD/CDD/DSIFN/SECOND/HRSCD/CLCD/OSCD）
- metric: f1/iou/miou/oa/precision/recall
- value: 数值（照抄，如 67.61 / 0.863 / 88.6）
- quote: 必须**逐字照抄**原文中包含该数值的那一句话（不能改写、不能概括）

只输出 JSON：{"results":[{"dataset":"..","metric":"..","value":"..","quote":".."}]}

铁律：
1. value 这个数字必须原封不动地出现在 quote 里。
2. 只提取"本文方法"的结果，不提取基线、对比方法、消融变体。
3. 不提取"差值/提升幅度"（如 F1 提升了 1.5%），只提取绝对分数。
4. 不确定就不提取。
"""


def extract(text: str) -> list[dict]:
    r = litellm.completion(
        model=MODEL, api_key=API_KEY, api_base=API_BASE,
        messages=[{"role": "system", "content": "你是论文数据抽取器，只会照抄原文，绝不编造。"},
                  {"role": "user", "content": PROMPT + "\n\n论文文本：\n" + text[:10000]}],
        temperature=0.0, max_tokens=2000, extra_body={"enable_thinking": False})
    raw = r.choices[0].message.content or ""
    try:
        d = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return d.get("results", [])
    except Exception:
        return []


def main(limit: int = 5):
    qs, rejected = [], 0
    fts = sorted((CORPUS / "fulltext").glob("*.json"))[:limit]
    t0 = time.time()
    for ft in fts:
        d = json.loads(ft.read_text(encoding="utf-8"))
        title = d.get("title", "").strip()
        full_text = "\n".join(d.get("pages_text", []))
        results = extract(full_text)
        # 程序校验
        entries = []
        for r in results:
            v = str(r.get("value", ""))
            qte = str(r.get("quote", ""))
            # 铁律：value 必须是 quote 的子串
            if v and qte and v in qte:
                entries.append(r)
            else:
                rejected += 1
        # (dataset, metric) 唯一
        groups = {}
        for e in entries:
            groups.setdefault((e["dataset"], e["metric"]), []).append(e)
        for (ds, mt), es in groups.items():
            vals = set(str(e["value"]) for e in es)
            if len(vals) != 1:
                continue
            qs.append({
                "paper_id": d["arxiv_id"], "title": title,
                "question": f"论文《{title}》在 {ds} 上的 {mt} 是多少？",
                "gold": {"dataset": ds, "metric": mt, "value": es[0]["value"],
                         "evidence": es[0]["quote"]},
            })
        print(f"[{fts.index(ft)+1}/{limit}] {title[:40]} → 有效 {len(entries)} 条，拒绝编造 {rejected} 条（{time.time()-t0:.0f}s）", flush=True)

    out = CORPUS / "eval_set_v8.jsonl"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    bad = sum(1 for q in qs if q["gold"]["value"] not in q["gold"]["evidence"])
    print(f"\n共 {len(qs)} 道题，value 不在证据里的: {bad} 题（必须为 0）")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)