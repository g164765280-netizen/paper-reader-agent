# -*- coding: utf-8 -*-
"""概念题 PaperQA2 通路：本地 Qwen3-8B(vLLM) + 本地 bge-m3 embedding。

流程：
  1. 109 篇 fulltext 正文（用 pages_text，避免 PDF 重解析）→ Docs.aadd
  2. 概念题（concept_questions.jsonl）→ Docs.aquery
  3. 保存带出处的回答（answer + answer.contexts 引用的论文段落）

避坑（见交接文档）：
  - aadd 也要传 settings（默认 gpt-4o 会报"模型不可用"）
  - Settings.parsing.use_doc_details=False（跳过 S2/Crossref 429）
  - embedding 用 st- 前缀本地 bge-m3
  - LLM 用 hosted_vllm 指向本机 :8000 的 Qwen3-8B（免费）

用法（在 paperqa venv 下）：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python paperqa_concept.py
"""
from __future__ import annotations

import asyncio
import json
import glob
import sys
from pathlib import Path

from paperqa import Docs, Settings

VLLM_BASE = "http://localhost:8000/v1"
MODEL = "Qwen3-8B"

LLM_CONFIG = {
    "model_list": [
        {
            "model_name": MODEL,
            "litellm_params": {
                "model": f"hosted_vllm/{MODEL}",
                "api_base": VLLM_BASE,
                "api_key": "sk-no-key-required",
                "temperature": 0.1,
                "max_tokens": 1024,
                # 关闭 Qwen3 内建思考，避免输出被 thinking 污染
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        }
    ]
}

SETTINGS = Settings(
    llm=MODEL,
    llm_config=LLM_CONFIG,
    summary_llm=MODEL,
    summary_llm_config=LLM_CONFIG,
    embedding="st-/media/sdb1/gzj/weights/llm/rscd/bge-m3",
)
SETTINGS.parsing.use_doc_details = False
# agent 用默认 gpt-4o 时，非 agentic aquery 用到它只在个别分支；但显式改成本地模型最稳
SETTINGS.agent.agent_llm = MODEL
SETTINGS.agent.agent_llm_config = LLM_CONFIG

FULLTEXT_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/fulltext")
QUESTIONS = Path("/media/sdb1/gzj/support/llm/rscd/concept_questions.jsonl")
OUT = Path("/media/sdb1/gzj/support/llm/rscd/concept_answers.jsonl")
CHUNK_DIR = Path("/media/sdb1/gzj/support/llm/rscd/chunks_concept")

# 是否全量还是冒烟（第一批：全部 109）
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = 全量

def fulltext_to_json(pdf_json: Path) -> str:
    """把 fulltext JSON 的 pages_text 拼成纯文本。"""
    d = json.loads(pdf_json.read_text(encoding="utf-8"))
    pt = d.get("pages_text", [])
    if isinstance(pt, list):
        return "\n".join(str(p) for p in pt)
    return str(pt)

TXT_DIR = Path("/media/sdb1/gzj/support/llm/rscd/fulltext_txt")


async def main():
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"概念题 {len(qs)} 道", flush=True)

    fts = sorted(FULLTEXT_DIR.glob("*.json"))
    if LIMIT:
        fts = fts[:LIMIT]
    print(f"论文 {len(fts)} 篇", flush=True)

    docs = Docs()
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    # aadd 全量论文（先用正文文本写 .txt，再 aadd 文件路径，避免 Errno36 文件名过长）
    added = 0
    for i, f in enumerate(fts, 1):
        d = json.loads(f.read_text(encoding="utf-8"))
        # 用短的 arxiv_id 当 citation/docname，避免标题太长触发 Errno36
        paper_id = d.get("arxiv_id", f.stem)
        title = " ".join((d.get("title", "unknown") or "unknown").split())[:120]
        text = fulltext_to_json(f)
        txt_path = TXT_DIR / f"{paper_id}.txt"
        try:
            txt_path.write_text(text, encoding="utf-8")
            await docs.aadd(str(txt_path), settings=SETTINGS, citation=paper_id,
                            title=title, docname=paper_id)
            added += 1
            print(f"[{i}/{len(fts)}] ok  {paper_id}  {title[:40]}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(fts)}] ERR {paper_id}: {str(e)[:160]}", flush=True)
    print(f"\naadd 完成：{added}/{len(fts)} 篇。语料文本数={len(docs.texts)}", flush=True)

    # 依次 aquery 概念题
    results = []
    for qi, q in enumerate(qs, 1):
        print(f"\n=== [{qi}/{len(qs)}] 提问：{q['question']} ===", flush=True)
        try:
            s = await docs.aquery(q["question"], settings=SETTINGS)
            ans = getattr(s, "answer", None)
            ans_str = str(ans) if ans is not None else ""
            # 提取引用的出处（contexts）—— 存全字段：id(pqac引用id)/citation/name/context/score
            contexts = []
            try:
                for ctx in getattr(s, "contexts", []):
                    t = getattr(ctx, "text", None)  # Text 对象
                    doc = getattr(t, "doc", None)   # Doc 对象
                    contexts.append({
                        "id": getattr(ctx, "id", ""),       # pqac-xxx 引用id
                        "citation": getattr(doc, "citation", "") if doc else "",
                        "docname": getattr(doc, "docname", "") if doc else "",
                        "name": getattr(t, "name", "") if t else "",   # arxiv_id chunk N
                        "context": getattr(ctx, "context", ""),
                        "score": getattr(ctx, "score", None),
                    })
            except Exception:
                contexts = []
            results.append({
                "id": q["id"],
                "question": q["question"],
                "key_points": q["key_points"],
                "answer": ans_str,
                "contexts": contexts,
                "references": getattr(s, "references", ""),
                "formatted_answer": getattr(s, "formatted_answer", None),
            })
            print("  回答:", (ans_str[:300] + "...") if len(ans_str) > 300 else ans_str, flush=True)
            print("  出处条数:", len(contexts), flush=True)
        except Exception as e:
            print(f"  aquery ERROR: {str(e)[:200]}", flush=True)
            results.append({
                "id": q["id"], "question": q["question"], "key_points": q["key_points"],
                "answer": "", "contexts": [], "error": str(e)[:200],
            })

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results),
                   encoding="utf-8")
    print(f"\n已保存 {len(results)} 条回答到 {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())