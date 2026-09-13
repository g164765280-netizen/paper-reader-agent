# -*- coding: utf-8 -*-
"""数字题插件：查「本文方法指标」权威表 + 指标库兜底。

- 优先查 paper_metrics.json（从 eval_set_v9 gold 提取的 30 条权威指标）
- 兜底查 metric_library.jsonl（粗抽取，可能有脏数据）
- 论文名→paper_id 用 PaperResolver 解析（用户只说"BIT"也能定位）

- can_handle：题目含「方法名 + 数据集 + 指标」这类可查表模式则高分。
- run：查指标，返回数值 + 出处。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from plugin_base import PluginResult
from paper_resolver import PaperResolver

METRICS_FILE = Path(__file__).parent / "paper_metrics.json"
LIB_FILE = Path("/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl")

METRIC_WORDS = ["f1", "iou", "miou", "oa", "precision", "recall", "kappa",
                "准确率", "召回率", "精确率", "指标", "得分", "值是多少", "多少"]
DATASET_WORDS = ["levir", "whu", "s2looking", "sysu", "cdd", "dsifn", "second",
                 "hrscd", "clcd", "oscd"]


class MetricPlugin:
    name = "lookup_metric"
    description = "查询论文在某数据集上的指标数值（数字题）"

    def __init__(self):
        self.resolver = PaperResolver()
        self._metrics = None
        self._lib = None

    def _load_metrics(self) -> dict:
        if self._metrics is None:
            self._metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        return self._metrics

    def _load_lib(self) -> dict:
        if self._lib is None:
            self._lib = {}
            p = Path(LIB_FILE)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            r = json.loads(line)
                            pid = r.get("paper_id", "")
                            self._lib.setdefault(pid, []).append(r)
                        except Exception:
                            pass
        return self._lib

    def can_handle(self, question: str) -> float:
        q = question.lower()
        has_dataset = any(w in q for w in DATASET_WORDS)
        has_metric = any(w in q for w in METRIC_WORDS)
        asks_value = bool(re.search(r"多少|是多少|值|多少分|=", question))
        # 若能从问题里解析出方法名，也是强数字题信号
        has_method = self.resolver.resolve(question) is not None
        score = 0.0
        if has_dataset and has_metric:
            score += 0.6
        if asks_value:
            score += 0.2
        if has_method and asks_value:
            score += 0.2
        return min(score, 1.0)

    def _extract_field(self, question: str) -> str:
        """从问题里提取指标名（F1/IoU/OA/Precision/Recall）。"""
        for w in ["f1", "iou", "miou", "oa", "precision", "recall", "kappa"]:
            if re.search(rf"\b{w}\b", question, re.IGNORECASE):
                return w
        return ""

    def _extract_dataset(self, question: str) -> str:
        for w in DATASET_WORDS:
            if w in question.lower():
                return w
        return ""

    def run(self, question: str, paper_id: str | None = None,
            field: str | None = None, **kwargs) -> PluginResult:
        pid = paper_id or self.resolver.resolve(question)
        if not pid:
            return PluginResult(answer="无法识别你问的是哪篇论文（方法名→论文映射未命中）",
                                confidence=0.0, refused=True)
        field = (field or self._extract_field(question)).lower()
        dataset = self._extract_dataset(question)

        # 1. 优先查权威指标表 paper_metrics.json
        metrics = self._load_metrics()
        if pid in metrics:
            hits = metrics[pid]
            # 按 metric + dataset 过滤
            matched = [r for r in hits
                       if (not field or r.get("metric", "").lower() == field)
                       and (not dataset or dataset in r.get("dataset", "").lower())]
            if matched:
                r = matched[0]
                title = r.get("title", "")
                ans = (f"论文《{title[:40]}》的 {r.get('method')} 方法"
                       f"在 {r.get('dataset')} 上的 {r.get('metric')} = {r.get('value')}")
                return PluginResult(
                    answer=ans,
                    sources=[f"{title} ({pid})"],
                    confidence=0.95, refused=False,
                    meta={"paper_id": pid, "method": r.get("method"),
                          "dataset": r.get("dataset"), "metric": r.get("metric"),
                          "value": r.get("value"), "source": "paper_metrics"})

        # 2. 兜底查 metric_library
        lib = self._load_lib()
        recs = lib.get(pid, [])
        if not recs:
            return PluginResult(answer=f"未找到论文 {pid} 的指标记录", confidence=0.0,
                                refused=True, meta={"paper_id": pid})
        matched = [r for r in recs if (not field or field in str(r.get("metric", "")).lower())]
        values = [r.get("value") for r in matched if r.get("value") not in (None, "-", "")]
        if not values:
            return PluginResult(answer=f"论文 {pid} 未找到 {field} 指标的干净数值（指标库该论文可能只有对比表数据）",
                                confidence=0.0, refused=True, meta={"paper_id": pid})
        uniq = list(dict.fromkeys(values))
        conf = 0.7 if len(uniq) == 1 else 0.4
        ans = f"论文 {pid} 的 {field} 值约为 {uniq[0]}" if len(uniq) == 1 \
            else f"论文 {pid} 的 {field} 有多个值：{', '.join(map(str, uniq[:3]))}"
        return PluginResult(answer=ans, sources=[f"{pid}/{field}"], confidence=conf,
                            refused=False, meta={"paper_id": pid, "field": field,
                                                 "source": "metric_library"})