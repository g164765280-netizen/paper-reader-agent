# -*- coding: utf-8 -*-
"""遥感 RAG Agent demo：演示路由式多模态融合。"""
import sys
from pathlib import Path

from agent import RS_Agent, VLM, load_metric_lookup
from rag_pipeline import RAGPipeline

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")
EMB = "/media/sdb1/gzj/weights/llm/rscd/bge-m3"
RERANK = "/media/sdb1/gzj/weights/llm/rscd/bge-reranker-v2-m3"
TEXT_MODEL = "/media/sdb1/yjl/ollama-models/Qwen3-8B"


def main():
    print("loading RAG + VLM ...", flush=True)
    rag = RAGPipeline(CORPUS, EMB, RERANK, TEXT_MODEL)
    vlm = VLM()
    lookup = load_metric_lookup(CORPUS)
    agent = RS_Agent(rag, vlm, lookup)
    print("agent ready. 工具:", [t["name"] for t in agent.list_tools()], flush=True)

    # 演示：文本问答
    r1 = agent.answer("变化检测中常用的评价指标有哪些？")
    print("\n=== 文本问答 ===")
    print(r1["answer"][:400])

    # 演示：单张影像描述
    img = "/media/sdb1/gzj/data/LEVIR-CD/test/A/test_103.png"
    r2 = agent.answer("这张遥感影像有什么？", images=[img])
    print("\n=== 影像描述 ===")
    print(r2["answer"][:400])

    # 演示：双时相变化检测
    img_b = "/media/sdb1/gzj/data/LEVIR-CD/test/B/test_103.png"
    r3 = agent.answer("", images=[img, img_b])
    print("\n=== 变化检测 ===")
    print(r3["answer"][:400])


if __name__ == "__main__":
    main()
