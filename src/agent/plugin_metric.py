# -*- coding: utf-8 -*-
"""数字题插件：查指标库（metric_library.jsonl）提取数值。

- can_handle：用规则判断——题目含「方法名 + 数据集 + 指标」这类可查表模式则高分。
- run：查指标库，返回数值 + 出处（原句）。
- 置信度：命中条数越多、匹配越唯一，置信度越高；查不到则 refused=True。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from plugin_base import PluginResult

LIB_FILE = Path("/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl")

# 指标关键词（用于判断「这题是不是在问数字」）
METRIC_WORDS = ["f1", "iou", "miou", "oa", "precision", "recall", "kappa",
                "准确率", "召回率", "精确率", "指标", "得分", "值是多少", "多少"]
# 数据集关键词
DATASET_WORDS = ["levir", "whu", "s2looking", "sysu", "cdd", "dsifn", "second",
                 "hrscd", "clcd", "oscd"]


class MetricPlugin:
    name = "lookup_metric"
    description = "查询论文在某数据集上的指标数值（数字题）"

    def __init__(self, lib_file: Path | str = LIB_FILE):
        self.lib_file = Path(lib_file)
        self._lib: dict | None = None

    def _load(self) -> dict:
        if self._lib is None:
            self._lib = {}
            if self.lib_file.exists():
                for line in self.lib_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    pid = r.get("paper_id", "")
                    self._lib.setdefault(pid, []).append(r)
        return self._lib

    def can_handle(self, question: str) -> float:
        """含「方法名+数据集+指标」查表模式则高分，否则低分。"""
        q = question.lower()
        has_dataset = any(w in q for w in DATASET_WORDS)
        has_metric = any(w in q for w in METRIC_WORDS)
        # 问"多少/值"是强信号
        asks_value = bool(re.search(r"多少|是多少|值|=|多少分", question))
        score = 0.0
        if has_dataset and has_metric:
            score += 0.6
        if asks_value:
            score += 0.3
        return min(score, 1.0)

    def run(self, question: str, paper_id: str | None = None,
            field: str | None = None, **kwargs) -> PluginResult:
        lib = self._load()
        if not paper_id or not lib.get(paper_id):
            return PluginResult(answer="未找到该论文的指标记录", confidence=0.0,
                                refused=True, meta={"paper_id": paper_id})
        recs = lib[paper_id]
        # 匹配 field（指标名），大小写不敏感
        field = (field or "").lower()
        matched = [r for r in recs if r.get("metric", "").lower() == field]
        if not matched:
            # 退而求其次：找任意含 field 关键词的记录
            matched = [r for r in recs if field in str(r).lower()]
        if not matched:
            return PluginResult(answer=f"论文 {paper_id} 未找到 {field} 指标",
                                confidence=0.0, refused=True,
                                meta={"paper_id": paper_id, "field": field})
        # 取唯一值
        values = [r.get("value") for r in matched if r.get("value") is not None]
        if not values:
            return PluginResult(answer="指标记录无数值", confidence=0.0,
                                refused=True)
        # 唯一值 → 高置信度；多值 → 低置信度
        uniq = list(dict.fromkeys(values))
        conf = 0.9 if len(uniq) == 1 else 0.6
        answer = f"论文 {paper_id} 的 {field} 值为 {uniq[0]}" if len(uniq) == 1 \
            else f"论文 {paper_id} 的 {field} 有多个值：{', '.join(map(str, uniq[:3]))}"
        sources = [f"{r.get('method', '')} / {r.get('dataset', '')} / {r.get('metric', '')}" for r in matched[:3]]
        return PluginResult(answer=answer, sources=sources, confidence=conf,
                            refused=False, meta={"paper_id": paper_id, "field": field,
                                                 "values": uniq})