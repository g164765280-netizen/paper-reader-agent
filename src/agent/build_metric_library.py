# -*- coding: utf-8 -*-
"""建指标库：Docling 抽的表 → {paper_id, method, dataset, metric, value} 长表记录。

只处理"列名 = 数据集.指标"（如 LEVIR-CD.F1）这类干净表格；合并表头（Pre./Rec./F1 挤一格）暂跳过。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TABLES_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/tables_docling")
OUT_FILE = Path("/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl")

DATASETS = ["LEVIR-CD", "WHU-CD", "DSIFN-CD", "S2Looking", "SYSU-CD", "CDD", "SECOND",
            "HRSCD", "CLCD", "OSCD", "LEVIR", "WHU", "DOTA", "DIOR", "xView2"]


def main():
    records = []
    for f in sorted(TABLES_DIR.glob("*.json")):
        pid = f.stem
        try:
            tables = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for t in tables:
            cols = t.get("columns", [])
            # 找"数据集.指标"列 + 方法列（第一列）
            metric_idx = {}
            for i, c in enumerate(cols):
                c = str(c)
                for ds in DATASETS:
                    if c.startswith(ds + "."):
                        metric_idx[i] = (ds, c.split(".", 1)[1].lower().strip())
                        break
            if not metric_idx:
                continue
            method_col = cols[0] if cols else ""
            for row in t.get("df", []):
                method = str(row.get(method_col, "")).strip()
                if not method:
                    continue
                for i, (ds, metric) in metric_idx.items():
                    v = row.get(cols[i])
                    if v is None or v == "" or str(v).lower() in ("nan", "none"):
                        continue
                    records.append({"paper_id": pid, "method": method,
                                    "dataset": ds, "metric": metric, "value": str(v)})
    OUT_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"共 {len(records)} 条指标记录，存至 {OUT_FILE}")


if __name__ == "__main__":
    main()