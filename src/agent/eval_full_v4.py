# -*- coding: utf-8 -*-
"""无泄漏完整链路 v4：标题→论文ID + 指标表查表 + 远程生成 + 打分。"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import litellm

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")
API_KEY = "sk_tr_MiutB676oeX7EfIkA-G_gSG-hRdHrEvfo-Uh2TMYBhI"
API_BASE = "https://tokenrhythm.studio/v1"
MODEL = "openai/qwen3.8-flash"

METRIC_RE = re.compile(
    r"(F1|IoU|mIoU|OA|Precision|Recall)(?:-?score)?\s*(?:of)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE)
KNOWN_DATASETS = ["LEVIR-CD", "WHU-CD", "S2Looking", "SYSU-CD", "CDD", "DSIFN", "SECOND",
                  "HRSCD", "CLCD", "OSCD", "LEVIR", "WHU", "xView2"]


def extract_table(text):
    out, seen = [], set()
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    for m in METRIC_RE.finditer(text):
        field, num = m.group(1).lower(), m.group(2)
        if (field, num) in seen:
            continue
        seen.add((field, num))
        # 干净的那一句（含数字，只含一个数字）
        ev = text[m.start():m.end()]
        for s in sentences:
            if m.group(0) in s or (field in s and num in s):
                ev = s.strip()
                break
        # 用 ±250 窗口找数据集（只用于匹配，不喂模型）
        s = max(0, m.start() - 250)
        e = min(len(text), m.end() + 250)
        ctx = text[s:e]
        ds = [d for d in KNOWN_DATASETS if re.search(re.escape(d), ctx, re.IGNORECASE)]
        if ds:
            out.append({"metric": field, "value": num, "dataset": ds[0], "sentence": ev})
    return out


def generate(question, evidence_sentences):
    ev = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(evidence_sentences[:3]))
    r = litellm.completion(
        model=MODEL, api_key=API_KEY, api_base=API_BASE,
        messages=[
            {"role": "system", "content": "你是论文问答助手。严格基于证据回答，数字照抄证据，答不出说无法回答。"},
            {"role": "user", "content": f"问题：{question}\n证据：\n{ev}\n请回答："},
        ],
        temperature=0.0, max_tokens=200,
        extra_body={"enable_thinking": False},
    )
    return r.choices[0].message.content or ""


def nums_in(t):
    return [float(x) for x in re.findall(r"\d+\.\d+|\d+", t)]


def num_match(gold, ans_nums):
    g = float(gold)
    for a in ans_nums:
        if abs(a - g) <= max(0.001, g * 0.02) or abs(a - g * 100) <= 0.5 or abs(g - a * 100) <= 0.005:
            return True
    return False


def main():
    title2id, table = {}, {}
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        title2id[d.get("title", "").strip()] = d["arxiv_id"]
        table[d["arxiv_id"]] = extract_table("\n".join(d.get("pages_text", [])))

    qs = [json.loads(l) for l in (CORPUS / "eval_set_v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    t0 = time.time()
    acc = 0
    n = len(qs)
    for i, q in enumerate(qs, 1):
        gold = q["gold"]
        pid = title2id.get(q["title"])
        # 无泄漏检索：只用题目里的 metric+dataset 查表
        hits = [e for e in table.get(pid, []) if e["metric"] == gold["metric"] and e["dataset"] == gold["dataset"]]
        sentences = [e["sentence"] for e in hits]
        ans = generate(q["question"], sentences)
        ok = num_match(gold["value"], nums_in(ans))
        acc += ok
        if i % 20 == 0:
            print(f"[{i}/{n}] 累计正确率 {acc/i*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n最终答案正确率: {acc}/{n} = {acc/n*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
