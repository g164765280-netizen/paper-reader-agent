# -*- coding: utf-8 -*-
"""能力目录（Catalog）：给调度器看的插件清单。

对应 DSH 的 skill catalog 思想：只列「名称 + 描述」，不暴露实现。
新增能力 = 在这里注册一条，调度器无需改动。

当前四大能力（终极大专家的四层）：
  1. lookup_metric      数字题查指标（层1 检索）
  2. answer_concept     概念题问答 RAG（层1 检索）
  3. summarize_compare  归纳对比（层2 归纳）
  4. （层3 propose_idea 创见 / 层4 debate 多agent —— 待实现）
"""
from __future__ import annotations

from plugin_metric import MetricPlugin
from plugin_concept import ConceptPlugin
from plugin_summarize_compare import SummarizeComparePlugin


def build_plugins() -> list:
    """构建插件列表（单一入口，新增插件在这里注册）。"""
    return [
        MetricPlugin(),
        ConceptPlugin(),
        SummarizeComparePlugin(),
    ]


def build_catalog(plugins: list) -> list[dict]:
    """生成能力目录：list[{name, description}]。"""
    return [
        {"name": p.name, "description": p.description}
        for p in plugins
    ]