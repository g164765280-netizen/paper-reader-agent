# -*- coding: utf-8 -*-
"""Docling 抽表测试：处理 BIT PDF，导出表格 DataFrame，看结构是否保住。"""
from __future__ import annotations

import sys

from docling.document_converter import DocumentConverter

PDF = "/media/sdb1/gzj/data/rscd/corpus/pdf_fulltext/2103.00208v3.pdf"


def main():
    print("加载 Docling 转换器（首次会下载模型，走 hf-mirror）...", flush=True)
    converter = DocumentConverter()
    print("转换 PDF ...", flush=True)
    result = converter.convert(PDF)

    tables = result.document.tables
    print(f"共识别 {len(tables)} 张表", flush=True)
    for i, table in enumerate(tables):
        try:
            df = table.export_to_dataframe()
        except Exception as e:
            print(f"表{i} 导出失败: {str(e)[:100]}", flush=True)
            continue
        # 只看含指标的（F1/IoU/OA/LEVIR 等）
        txt = df.to_string() if hasattr(df, "to_string") else str(df)
        if any(k in txt for k in ["F1", "IoU", "OA", "LEVIR", "WHU"]):
            print(f"\n===== 表{i}（含指标）=====", flush=True)
            print(df.to_string()[:2000], flush=True)


if __name__ == "__main__":
    main()