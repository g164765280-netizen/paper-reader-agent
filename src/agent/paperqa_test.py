# -*- coding: utf-8 -*-
"""PaperQA2 对比测试：验证它是否比手写 RAG（无泄漏 45.1%）更优。

用法（需 GPU + vLLM 服务 Qwen3-8B:8000）：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python paperqa_test.py

配置：LLM=Qwen3-8B(vLLM) + embedding=bge-m3(本地 sentence-transformers)。
先跑 10 篇论文 + 1 个问题冒烟，通过后再扩到全量 + v3 评测集。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from paperqa import Docs, Settings

LLM_CONFIG = {
    "model_list": [
        {
            "model_name": "qwen3-8b",
            "litellm_params": {
                "model": "hosted_vllm/Qwen3-8B",
                "api_base": "http://localhost:8000/v1",
                "api_key": "sk-no-key-required",
                "temperature": 0.1,
                "max_tokens": 512,
            },
        }
    ]
}

SETTINGS = Settings(
    llm="qwen3-8b",
    llm_config=LLM_CONFIG,
    embedding="st-/media/sdb1/gzj/weights/llm/rscd/bge-m3",
)

PAPERS_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/pdf_fulltext")


async def main():
    docs = Docs()
    papers = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"共 {len(papers)} 篇论文，先加前 10 篇测试", flush=True)
    for p in papers[:10]:
        print("  adding", p.name, flush=True)
        await docs.aadd(str(p))

    q = "变化检测中常用的评价指标有哪些？"
    print(f"\n=== 提问：{q} ===", flush=True)
    session = await docs.aquery(q, settings=SETTINGS)
    ans = getattr(session, "answer", None)
    if ans is None:
        print("answer 为空，session 属性:", [a for a in dir(session) if not a.startswith("_")], flush=True)
    else:
        print("=== 回答 ===", flush=True)
        print(str(ans)[:800], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
