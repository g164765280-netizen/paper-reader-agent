# -*- coding: utf-8 -*-
"""Qwen3-VL-8B 看图抽表：把 PDF 页渲染成图片，让 VLM 抽出"行=方法、列=数据集×指标"的表格。"""
from __future__ import annotations

import sys

import fitz  # pymupdf
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "/media/sdb1/yjl/ollama-models/Qwen3-VL-8B-Instruct"
PDF = "/media/sdb1/gzj/data/rscd/corpus/pdf_fulltext/2103.00208v3.pdf"
PAGE = 8

PROMPT = """这张图片是一篇论文的实验结果表。请把它还原成"行=方法名、列=数据集×指标"的表格。
对每一行（每个方法），列出它对应的数值。如果表头有"LEVIR-CD / WHU-CD / DSIFN-CD"等数据集、每列又分 F1/IoU 等指标，请明确写出"方法 X 在数据集 Y 的指标 Z = 数值"。

逐条输出，格式（只输出这个，不要别的）：
方法名；数据集；指标；数值
（每行一条）

如果看不清楚表头，就按你能看出的结构如实写，不要编造没写出的数字。"""


def main():
    doc = fitz.open(PDF)
    page = doc[PAGE - 1]
    pix = page.get_pixmap(dpi=150)
    img_path = "/tmp/bit_table_page.png"
    pix.save(img_path)
    print(f"已渲染第 {PAGE} 页到 {img_path}（{pix.width}x{pix.height}）", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    img = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
    ans = processor.decode(out[0], skip_special_tokens=True)
    print("=== VLM 抽出结果 ===", flush=True)
    print(ans, flush=True)


if __name__ == "__main__":
    main()