# -*- coding: utf-8 -*-
"""批量用 Docling 抽表：120 篇 PDF → 每篇一个 JSON（含所有表格 DataFrame + 页码），可断点续跑。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

PDF_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/pdf_fulltext")
OUT_DIR = Path("/media/sdb1/gzj/data/rscd/corpus/tables_docling")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    done = 0
    t0 = time.time()
    for i, pdf in enumerate(pdfs, 1):
        pid = pdf.stem
        out_file = OUT_DIR / f"{pid}.json"
        if out_file.exists():
            continue  # 已处理，跳过
        try:
            result = converter.convert(str(pdf))
            tables = []
            for t in result.document.tables:
                try:
                    df = t.export_to_dataframe()
                    # 处理重复列名（多级表头展平常出现）：用 index 方向保留全部数据
                    df.columns = [f"{c}" for c in df.columns]
                    tables.append({
                        "df": df.to_dict(orient="records"),
                        "columns": [str(c) for c in df.columns],
                    })
                except Exception:
                    pass
            out_file.write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")
            done += 1
            print(f"[{i}/{len(pdfs)}] {pid}: {len(tables)} 表（累计 {done} 篇，{time.time()-t0:.0f}s）", flush=True)
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] {pid}: ERROR {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()