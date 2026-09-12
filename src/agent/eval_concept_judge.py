# -*- coding: utf-8 -*-
"""概念题 LLM-as-judge 评分：有出处率 + 答到点率 + 诚实。

用 API（qwen3.8-max）做独立评测（本地模型答题，第三方 API 评分，避免自评）。
三个维度：
  1. 有出处（grounded）：回答是否引用了真实论文段落，不编造。
  2. 答到点（relevance）：是否命中金标准关键点。
  3. 诚实（honesty）：不知道时是否说"查不到"，而不是瞎编。

用法：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python eval_concept_judge.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

# API 凭据从环境变量读取，不硬编码（仓库是 public，禁止提交明文 key）。
# 服务器上 key 已存 /media/sdb1/gzj/support/llm/rscd/.mentor_env（chmod 600），
# 运行前 export OPENAI_API_KEY / OPENAI_API_BASE。
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://tokenrhythm.studio/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3.8-max")

ANSWERS = Path(os.environ.get("ANSWERS_FILE",
                               "/media/sdb1/gzj/support/llm/rscd/concept_answers.jsonl"))
OUT = Path(os.environ.get("OUT_FILE",
                          "/media/sdb1/gzj/support/llm/rscd/concept_eval_report.jsonl"))

SYS = "你是严格的评测裁判。只输出 JSON，不要解释。"

PROMPT = """请对下面这个「遥感变化检测概念题」的 AI 回答做三维评分。

【题目】
{question}

【金标准关键点】
{key_points}

【AI 回答（含引用标记和文末 References）】
{answer}

【AI 回答引用的出处列表（每条含 id=pqac引用id、name=chunk名、citation=论文ID、context=引用原文；可能为空）】
{contexts}

【文末参考文献（References）】
{references}

请逐维度判断并打分，每个维度 0 或 1：
1. grounded（有出处）：回答的结论是否有真实论文段落支撑、而非凭空编造。判断标准：回答正文里的引用标记（形如 pqac-xxxx 或 "arxiv_id chunk N"）能否在"出处列表"或"References"里找到对应条目，且该条目内容确实与所述相关。若回答无任何引用，或引用标记在列表里找不到（编造引用），判 0。
2. relevance（答到点）：回答是否命中上述金标准的关键点（至少命中核心要点）。
3. honesty（诚实）：如果回答承认"查不到/不确定/信息不足"且未瞎编，判 1；若信息不足却硬编内容或编造引用，判 0。若回答合理、有真实出处支撑、未瞎编，也判 1。

严格按此 JSON 格式输出（不要有任何多余文字）：
{{"grounded": 0或1, "relevance": 0或1, "honesty": 0或1, "reason": "一句话理由"}}"""


def parse_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    rows = [json.loads(l) for l in ANSWERS.read_text(encoding="utf-8").splitlines() if l.strip()]

    results = []
    for i, r in enumerate(rows, 1):
        kp = "；".join(r.get("key_points", []))
        # 用 formatted_answer（paperqa 官方：已把 pqac-id 替换成 chunk 名 + 附 References 列表）
        # 若没有则回退到 answer。
        ans_for_judge = (r.get("formatted_answer") or r.get("answer") or "")[:2500]
        # 提取答案里实际引用的 pqac-id 和 chunk 名，只传"被引用到的" contexts（避免截断漏传）
        ans_raw = r.get("answer") or ""
        cited_ids = set(re.findall(r"pqac-[0-9a-f]+", ans_raw))
        cited_names = set(re.findall(r"\d{4}\.\d{4,5}v\d+\s+chunk\s+\d+", ans_raw))
        ctxs = []
        for c in r.get("contexts", []):
            if not isinstance(c, dict):
                continue
            hit = (c.get("id", "") in cited_ids) or (c.get("name", "") in cited_names)
            if hit:
                ctxs.append({
                    "id": c.get("id", ""),
                    "name": c.get("name", ""),
                    "citation": c.get("citation", ""),
                    "context": (c.get("context", "") or "")[:300],
                })
        # 若没匹配到任何引用，退而传前 8 条（让 judge 判断是否根本无出处）
        if not ctxs:
            for c in r.get("contexts", [])[:8]:
                if not isinstance(c, dict):
                    continue
                ctxs.append({
                    "id": c.get("id", ""),
                    "name": c.get("name", ""),
                    "citation": c.get("citation", ""),
                    "context": (c.get("context", "") or "")[:300],
                })
        ctx = json.dumps(ctxs[:15], ensure_ascii=False)
        refs = (r.get("references") or "")[:1500]
        prompt = PROMPT.format(
            question=r["question"], key_points=kp,
            answer=ans_for_judge, contexts=ctx, references=refs,
        )
        verdict = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=600,
                    extra_body={"enable_thinking": False},
                )
                raw = resp.choices[0].message.content or ""
                # 兜底：若 content 为空但 reasoning_content 有 JSON，则取 reasoning
                if not raw:
                    raw = (getattr(resp.choices[0].message, "reasoning_content", None)
                           or getattr(resp.choices[0].message, "reasoning", None) or "")
                verdict = parse_json(raw)
                if verdict is not None:
                    break
                print(f"  [{i}] 解析失败，重试 {attempt+1}: {raw[:150]}", flush=True)
            except Exception as e:
                print(f"  [{i}] judge API 错误: {str(e)[:150]}", flush=True)

        if verdict is None:
            verdict = {"grounded": 0, "relevance": 0, "honesty": 0, "reason": "judge 失败"}
        out = {**r, "judge": verdict}
        results.append(out)
        print(f"[{i}/{len(rows)}] {r['id']} grounded={verdict.get('grounded')} "
              f"relevance={verdict.get('relevance')} honesty={verdict.get('honesty')} "
              f"| {verdict.get('reason','')[:60]}", flush=True)

    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results),
                   encoding="utf-8")

    n = len(results)
    g = sum(1 for x in results if x["judge"].get("grounded") == 1)
    rel = sum(1 for x in results if x["judge"].get("relevance") == 1)
    h = sum(1 for x in results if x["judge"].get("honesty") == 1)
    print(f"\n=== 评测统计（{n} 题）===", flush=True)
    print(f"有出处率 grounded: {g}/{n} = {g/n*100:.1f}%", flush=True)
    print(f"答到点率 relevance: {rel}/{n} = {rel/n*100:.1f}%", flush=True)
    print(f"诚实 honesty: {h}/{n} = {h/n*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()