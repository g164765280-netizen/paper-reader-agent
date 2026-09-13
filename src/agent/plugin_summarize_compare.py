# -*- coding: utf-8 -*-
"""层2 归纳对比插件：同语境多篇论文的方案 + 实验结果横向归纳。

能力：用户问「XX 类方法各自思路/结果对比」时，召回多篇论文 + 查指标库，
把「方法/思路/数据集/指标」结构化，再让（可插拔的）LLM 归纳成对比 + 总结。

流程：
  1. 多文档召回：用 PaperQA2/embedding 找同语境 top-N 篇（非 top1）
  2. 结构化：结合指标库 metric_library.jsonl 拿到各方法实验结果
  3. 归纳生成：LLMBackend 输出「共同范式 vs 差异点」对比

可插拔：归纳用 LLMBackend（本地 8B / API 大模型一键切）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from plugin_base import PluginResult
from llm_backend import LLMBackend, get_backend

METRIC_LIB = "/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl"

COMPARE_PROMPT = """你是遥感变化检测专家。下面是从多篇论文里提取的方法与实验结果，请做横向归纳对比。

【用户问题】
{question}

【各方法信息】
{methods_info}

请输出：
1. 【对比表】方法 | 核心思路 | 主要数据集 | 代表性指标（F1/IoU）
2. 【共同范式】这些方法的共同点
3. 【关键差异】各自的创新点和差异
4. 【趋势总结】一句话归纳这类方法的演进趋势

要求：只基于给出的信息，不要编造；信息不足就说明不足。"""


class SummarizeComparePlugin:
    name = "summarize_compare"
    description = "同语境多篇论文的方案与实验结果归纳对比"

    def __init__(self, backend: LLMBackend | None = None):
        self.backend = backend or get_backend("local")
        self._lib = None

    def can_handle(self, question: str) -> float:
        kw = ["对比", "比较", "区别", "哪些方法", "归纳", "总结", "各自", "演进", "综述", "几种",
              "相比", "优势", "缺点", "异同", "横向"]
        hits = sum(1 for k in kw if k in question)
        score = 0.3 * hits
        if any(k in question for k in ["对比", "比较", "区别", "各自", "异同"]):
            score += 0.4
        # 检测"多个方法专有名词"（大写驼峰词 ≥2 个）→ 强归纳信号
        methods = re.findall(r"[A-Z][a-zA-Z]+(?:\s?[-+]\s?[A-Z][a-zA-Z]+)*", question)
        if len(set(methods)) >= 2:
            score += 0.4
        return min(score, 1.0)

    def _load_lib(self) -> list[dict]:
        if self._lib is None:
            self._lib = []
            p = Path(METRIC_LIB)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            self._lib.append(json.loads(line))
                        except Exception:
                            pass
        return self._lib

    def _load_concepts(self) -> dict:
        """加载权威词条里的方法定义（method category），用于补充「核心思路」。"""
        if getattr(self, "_concepts", None) is None:
            self._concepts = {}
            p = Path("/media/sdb1/gzj/support/llm/rscd/concepts_authoritative.json")
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                for c in d.get("concepts", []):
                    if c.get("category") == "method":
                        # key: 小写英文名/中文名都映射到定义
                        self._concepts[c.get("term", "").lower()] = c.get("definition", "")
                        self._concepts[c.get("term_zh", "")] = c.get("definition", "")
        return self._concepts

    def _is_noise_method(self, m: str) -> bool:
        """消融噪声判断：纯数字、Base/+/ablation 变体都算噪声。"""
        m = m.strip()
        if re.fullmatch(r"\d+", m):
            return True
        if re.search(r"^base\b|\+\s*|ablation|s[0-9]\b", m, re.I):
            return True
        return False

    def _format_methods(self, rows: list[dict], question: str) -> str:
        """结构化：方法定义(权威词条) + 指标数据，按问题里的方法名过滤，清洗噪声。"""
        concepts = self._load_concepts()
        # 提取问题里提到的方法名（大写驼峰词）
        q_methods = set(re.findall(r"[A-Z][A-Za-z]+(?:\s?[-+]\s?[A-Z][A-Za-z]+)*", question))
        q_methods = {m.lower() for m in q_methods if len(m) > 1}

        by_method = {}
        for r in rows:
            m = r.get("method", "?")
            if self._is_noise_method(m):
                continue
            v = r.get("value")
            metric = str(r.get("metric", ""))
            if v in (None, "-", "", "None") or "." in metric[:30]:
                continue
            if not re.search(r"f1|iou|oa|precision|recall|kappa", metric, re.I):
                continue
            by_method.setdefault(m, []).append(f"{r.get('dataset')}.{metric}={v}")

        # 若问题里指明了方法名，只保留这些方法
        if q_methods:
            filtered = {m: k for m, k in by_method.items() if m.lower() in q_methods}
            if filtered:
                by_method = filtered

        lines = []
        for m, kpis in list(by_method.items())[:15]:
            # 补「核心思路」：从权威词条查定义
            idea = concepts.get(m.lower(), concepts.get(m, ""))
            idea_str = f"（思路：{idea}）" if idea else "（思路：未知）"
            kpi_str = ", ".join(kpis[:6])
            lines.append(f"- {m} {idea_str}：{kpi_str}")
        return "\n".join(lines) if lines else "（指标库无干净数据）"

    def run(self, question: str, paper_rows: list[dict] | None = None) -> PluginResult:
        lib = paper_rows if paper_rows else self._load_lib()
        methods_info = self._format_methods(lib, question)
        prompt = COMPARE_PROMPT.format(question=question, methods_info=methods_info)
        answer = self.backend.generate([
            {"role": "system", "content": "你是遥感变化检测专家，严格基于给定信息归纳对比。"},
            {"role": "user", "content": prompt},
        ], max_tokens=1500)
        # 简化版本：无需 PaperQA2 全文，直接用指标库聚合（可选注入全文后续再加）
        return PluginResult(
            answer=answer,
            sources=[f"{r.get('method')}/{r.get('dataset')}" for r in lib[:10]],
            confidence=0.6 if methods_info else 0.2,
            refused=not methods_info,
            meta={"n_methods": methods_info.count("\n- ")}
        )