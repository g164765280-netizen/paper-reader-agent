# -*- coding: utf-8 -*-
"""拆合并表头：列名"LEVIR-CD Pre. / Rec. / F1 / IoU / OA" → 拆成 (数据集, [指标列表])，
值"89.24 / 89.37 / 89.31 / 80.68 / 98.92" → 拆成 [数值列表]，追加进指标库。"""
from __future__ import annotations

import json
import re
from pathlib import Path

TABLES_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/tables_docling")
LIB_FILE = Path("/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl")

DATASETS = ["LEVIR-CD", "WHU-CD", "DSIFN-CD", "S2Looking", "SYSU-CD", "CDD", "SECOND",
            "HRSCD", "CLCD", "OSCD", "LEVIR+CD", "WHU", "DOTA", "DIOR", "xView2"]
METRIC_ALIAS = {"pre.": "precision", "rec.": "recall", "pr.": "precision", "p": "precision",
                "r": "recall", "pre": "precision", "rec": "recall", "f1": "f1", "iou": "iou",
                "oa": "oa", "miou": "miou", "precision": "precision", "recall": "recall"}


def norm_metric(m):
    m = re.sub(r"[↑↓*]", "", str(m)).strip().lower()
    return METRIC_ALIAS.get(m, m)


def parse_header(col: str):
    """'LEVIR-CD Pre. / Rec. / F1 / IoU / OA' → (dataset, [metric,...]) 或 None"""
    parts = [p.strip() for p in col.split("/")]
    if len(parts) < 2:
        return None
    first = parts[0]
    ds = None
    for d in sorted(DATASETS, key=len, reverse=True):
        if first.lower().startswith(d.lower()):
            ds = d
            first_metric = first[len(d):].strip()
            break
    if ds is None or not first_metric:
        return None
    metrics = [norm_metric(first_metric)] + [norm_metric(p) for p in parts[1:]]
    return ds, metrics


def main():
    # 读现有库，去重
    existing = set()
    for l in LIB_FILE.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            existing.add((r["paper_id"], r["method"], r["dataset"], r["metric"], r["value"]))

    added = 0
    out = open(LIB_FILE, "a", encoding="utf-8")
    for f in sorted(TABLES_DIR.glob("*.json")):
        pid = f.stem
        tables = json.loads(f.read_text(encoding="utf-8"))
        for t in tables:
            cols = [str(c) for c in t.get("columns", [])]
            # 找合并表头列
            for i, c in enumerate(cols):
                parsed = parse_header(c)
                if not parsed:
                    continue
                ds, metrics = parsed
                for row in t.get("df", []):
                    method = str(row.get(cols[0], "")).strip()
                    if not method:
                        continue
                    cell = str(row.get(cols[i], ""))
                    vals = [v.strip() for v in cell.split("/")]
                    if len(vals) != len(metrics):
                        continue
                    for m, v in zip(metrics, vals):
                        if v in ("", "-", "nan", "none"):
                            continue
                        key = (pid, method, ds, m, v)
                        if key in existing:
                            continue
                        existing.add(key)
                        out.write(json.dumps({"paper_id": pid, "method": method,
                                              "dataset": ds, "metric": m, "value": v},
                                             ensure_ascii=False) + "\n")
                        added += 1
    out.close()
    print(f"追加 {added} 条合并表头记录")
    total = sum(1 for _ in LIB_FILE.read_text(encoding="utf-8").splitlines() if _.strip())
    print(f"指标库总记录数: {total}")


if __name__ == "__main__":
    main()