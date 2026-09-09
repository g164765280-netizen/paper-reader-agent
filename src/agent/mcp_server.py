# -*- coding: utf-8 -*-
"""遥感 RAG Agent 的 MCP server：把 4 个工具通过 MCP 协议暴露。

工具：search_papers / lookup_metric / analyze_image / detect_change
运行：python mcp_server.py  （默认 stdio 传输，供 MCP 客户端调用）
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from agent import RS_Agent, VLM, load_metric_lookup
from rag_pipeline import RAGPipeline

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")
EMB = "/media/sdb1/gzj/weights/llm/rscd/bge-m3"
RERANK = "/media/sdb1/gzj/weights/llm/rscd/bge-reranker-v2-m3"
TEXT_MODEL = "/media/sdb1/yjl/ollama-models/Qwen3-8B"

mcp = FastMCP("rs-change-detection-agent")

# 全局惰性加载（首次调用工具时加载，避免启动即吃显存）
_agent = None


def _get_agent() -> RS_Agent:
    global _agent
    if _agent is None:
        rag = RAGPipeline(CORPUS, EMB, RERANK, TEXT_MODEL)
        vlm = VLM()
        lookup = load_metric_lookup(CORPUS)
        _agent = RS_Agent(rag, vlm, lookup)
    return _agent


@mcp.tool()
def search_papers(question: str) -> str:
    """检索变化检测论文并回答问题（纯文本）。"""
    return _get_agent().search_papers(question)


@mcp.tool()
def lookup_metric(paper_id: str, field: str) -> str:
    """查询某篇论文的某个指标（F1/IoU/OA/Precision/Recall）的数值。"""
    return _get_agent().lookup_metric(paper_id, field)


@mcp.tool()
def analyze_image(image_path: str, prompt: str = "") -> str:
    """描述/解读单张遥感影像。prompt 为空则默认描述地表类型。"""
    return _get_agent().analyze_image(image_path, prompt or None)


@mcp.tool()
def detect_change(image_a: str, image_b: str) -> str:
    """对比两张同区域不同时间的遥感影像，说明发生了什么变化。"""
    return _get_agent().detect_change(image_a, image_b)


@mcp.tool()
def fuse(question: str, image_path: str) -> str:
    """深度融合：检索论文知识 + 看影像，综合解读（多模态 RAG）。"""
    return _get_agent().fuse(question, image_path)


@mcp.tool()
def generate_report(method: str, dataset: str, metrics: str, change_map: str) -> str:
    """变化图 → 自动实验报告：输入方法/数据集/指标 + 变化图路径，输出规范结果分析报告。"""
    return _get_agent().generate_report(method, dataset, metrics, change_map)


if __name__ == "__main__":
    mcp.run()
