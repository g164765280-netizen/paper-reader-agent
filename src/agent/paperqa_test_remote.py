# -*- coding: utf-8 -*-
"""PaperQA2 证据层测试：远程 LLM(qwen3.8-flash) + 本地 bge-m3 embedding。

验证：PaperQA2 能否对我们的 120 篇论文做带出处的问答（替换手写 RAG）。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from paperqa import Docs, Settings

API_KEY = "sk_tr_MiutB676oeX7EfIkA-G_gSG-hRdHrEvfo-Uh2TMYBhI"
API_BASE = "https://tokenrhythm.studio/v1"

LLM_CONFIG = {
    "model_list": [
        {
            "model_name": "qwen3.8-flash",
            "litellm_params": {
                "model": "openai/qwen3.8-flash",
                "api_base": API_BASE,
                "api_key": API_KEY,
                "temperature": 0.1,
                "max_tokens": 512,
                "extra_body": {"enable_thinking": False},
            },
        }
    ]
}

SETTINGS = Settings(
    llm="qwen3.8-flash",
    llm_config=LLM_CONFIG,
    embedding="st-/media/sdb1/gzj/weights/llm/rscd/bge-m3",
)

PAPERS_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/pdf_fulltext")


async def main():
    docs = Docs()
    papers = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"共 {len(papers)} 篇论文，先加前 8 篇测试", flush=True)
    for p in papers[:8]:
        print("  adding", p.name, flush=True)
        await docs.aadd(str(p))

    for q in ["变化检测中常用的评价指标有哪些？", "BIT 方法的核心思想是什么？"]:
        print(f"\n=== 提问：{q} ===", flush=True)
        try:
            session = await docs.aquery(q, settings=SETTINGS)
            ans = getattr(session, "answer", None)
            if ans is None:
                print("answer 为空，session 属性:", [a for a in dir(session) if not a.startswith("_")], flush=True)
            else:
                print("=== 回答 ===", flush=True)
                print(str(ans)[:700], flush=True)
        except Exception as e:
            print(f"ERROR: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
