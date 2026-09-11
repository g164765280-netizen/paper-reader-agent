# -*- coding: utf-8 -*-
"""离线抽表（本地 Qwen3-8B，免费）：喂全文，抽「方法×数据集×指标→数值+原文行」，校验 value 在 quote 里。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")
MODEL_PATH = "/media/sdb1/yjl/ollama-models/Qwen3-8B"

PROMPT = """下面是一篇变化检测论文的全文。论文里有实验结果的"竖排表格"：方法名单独一行，下一行/几行是用 / 分隔或空格分隔的指标数值（F1、IoU、OA 等，对应表头的各数据集）。

请理解表头与列，抽出每个方法在 LEVIR-CD / WHU-CD / DSIFN-CD 等数据集上的 F1 和 IoU 数值。

只输出 JSON，格式：
{"results":[{"method":"方法名","dataset":"数据集","metric":"f1或iou","value":"数值","quote":"包含该数值的原文那一行"}]}

铁律：
1. value 这个数字必须原样出现在 quote 里，否则不要这条。
2. 方法名从原文找（如 Base、BIT、SNUNet、FC-EF 等）。
3. 数值照抄，不要换算、不要取整。
4. 只抽明确能对应上的，不确定就跳过。"""


def main(paper_file: str):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    d = json.loads(open(CORPUS / "fulltext" / paper_file, encoding="utf-8").read())
    text = "\n".join(d.get("pages_text", []))
    msgs = [{"role": "system", "content": "你是论文实验数据抽取器，只输出 JSON，照抄原文不编造。"},
            {"role": "user", "content": PROMPT + "\n\n论文全文：\n" + text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=2000, do_sample=False)
    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    try:
        j = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        results = j.get("results", [])
    except Exception:
        results = []
    # 校验：value 必须在 quote 里
    valid = [r for r in results if r.get("value", "") and str(r["value"]) in str(r.get("quote", ""))]
    print(f"原始抽 {len(results)} 条，校验后 {len(valid)} 条")
    for r in valid:
        print(f"  {r.get('method')} | {r.get('dataset')} | {r.get('metric')}={r.get('value')} | quote: {str(r.get('quote'))[:70]}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "2103.00208v3.json")